"""
T5歌单生成模型推理脚本
用于加载训练好的T5模型并根据输入文本生成歌曲推荐
"""
import os
import sys
import torch
import json
import logging
import random
import argparse
from typing import Dict, Tuple, List
from collections import defaultdict

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config
from src.generator.tiger_model import TIGERModel
from src.common.utils import setup_logging

logger = logging.getLogger(__name__)


class PlaylistGenerator:
    """处理模型加载和根据文本提示生成歌单"""

    def __init__(self, config: Config, model_path: str = None):
        """
        初始化歌单生成器
        
        Args:
            config: 配置对象
            model_path: 模型路径，如果为None则使用默认路径
        """
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 使用提供的模型路径或默认路径
        if model_path is None:
            model_path = os.path.join(self.config.model_dir, "generator", "final_model")
        self.model_path = model_path
        
        self.model = self._load_model()
        self.semantic_to_song_cluster = self._create_reverse_map()
        self.song_info_map = self._load_song_info()

    def _load_model(self) -> TIGERModel:
        """
        智能加载TIGER模型。
        - 如果是最终模型目录，则使用 TIGERModel.from_pretrained。
        - 如果是检查点目录，则通过 TIGERModel.__init__ 加载。
        """
        if not os.path.exists(self.model_path):
            logger.error(f"错误: 模型未找到 {self.model_path}")
            logger.error("请确保路径正确")
            sys.exit(1)

        # 判断是否为检查点目录
        is_checkpoint = "checkpoint" in os.path.basename(os.path.normpath(self.model_path))

        try:
            if is_checkpoint:
                logger.info(f"检测到检查点目录，使用 __init__ 方法加载: {self.model_path}")
                # 从主配置重新构建 layer_vocab_sizes
                rq_config = self.config.h_rqkmeans
                layer_vocab_sizes = {
                    'l1': rq_config.need_clusters[0],
                    'l2': rq_config.need_clusters[1],
                    'l3': rq_config.need_clusters[2],
                }
                # 直接实例化TIGERModel。这将从检查点加载基础T5模型，
                # 然后重新应用自定义token和嵌入层大小调整。
                model = TIGERModel(base_model=self.model_path, layer_vocab_sizes=layer_vocab_sizes)
            else:
                logger.info(f"正在从 {self.model_path} 加载最终模型 (使用 from_pretrained)...")
                # 对最终保存的模型使用自定义的 from_pretrained 方法
                model = TIGERModel.from_pretrained(self.model_path)
            
            model.model.to(self.device)
            model.model.eval()
            logger.info(f"模型加载成功。词汇表大小: {len(model.tokenizer)}")
            return model
        except Exception as e:
            logger.error(f"模型加载失败: {e}", exc_info=True)
            if is_checkpoint:
                logger.error("加载检查点失败。请确保检查点目录完整，并且Hugging Face模型文件存在。")
            else:
                logger.error("加载最终模型失败。请确保模型是使用 TIGERModel.save_pretrained 保存的。")
            sys.exit(1)

    def _create_reverse_map(self) -> Dict[Tuple[int, ...], List[str]]:
        """创建从语义ID到歌曲ID列表的反向映射"""
        mapping = defaultdict(list)
        semantic_ids_file = os.path.join(self.config.output_dir, "semantic_id", "song_semantic_ids.jsonl")
        if not os.path.exists(semantic_ids_file):
            logger.error(f"错误: song_semantic_ids.jsonl 未找到于 {semantic_ids_file}")
            logger.error("请先运行语义ID生成步骤")
            sys.exit(1)

        logger.info("正在创建语义ID到歌曲簇的反向映射...")
        with open(semantic_ids_file, 'r', encoding='utf-8') as f:
            for line in f: 
                item = json.loads(line)
                mapping[tuple(item['semantic_ids'])].append(item['song_id'])
        logger.info(f"已加载 {len(mapping)} 个唯一的语义ID簇")
        return mapping

    def _load_song_info(self) -> Dict[str, Dict[str, str]]:
        """加载歌曲信息（歌名、歌手）"""
        import csv
        mapping = {}
        try:
            with open(self.config.data.song_info_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                #next(reader, None)  # 跳过表头
                for row in reader:
                    if len(row) >= 3: 
                        mapping[row[0]] = {"name": row[1], "singer": row[2]}
            logger.info(f"已加载 {len(mapping)} 首歌曲的信息")
        except FileNotFoundError: 
            logger.warning(f"歌曲信息文件未找到: {self.config.data.song_info_file}")
        return mapping

    def generate(self, title: str, tags: str = "", max_songs: int = 20, temperature: float = 0.8) -> List[str]:
        """
        根据标题和标签生成歌单
        
        Args:
            title: 歌单标题/描述
            tags: 可选标签（当前未在生成中使用）
            max_songs: 最大生成歌曲数量
            temperature: 采样温度（越高越多样化）
            
        Returns:
            歌曲ID列表
        """
        # 格式化输入提示以匹配训练格式
        prompt = title
        logger.info(f"正在生成歌单，提示: '{prompt}'")

        # 对输入进行分词
        input_ids = self.model.tokenizer.base_tokenizer(
            prompt, 
            return_tensors="pt",
            max_length=self.config.generator_t5.max_input_length,
            truncation=True
        ).input_ids.to(self.device)

        # 生成语义ID
        with torch.no_grad():
            generated_ids = self.model.model.generate(
                input_ids,
                max_new_tokens=self.config.generator_t5.max_target_length,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=temperature,
                num_return_sequences=1,
                pad_token_id=self.model.tokenizer.pad_token_id
            )
        
        # 解码生成的token
        decoded_tokens = self.model.tokenizer.base_tokenizer.convert_ids_to_tokens(
            generated_ids[0], 
            skip_special_tokens=False
        )
        
        logger.debug(f"生成的token (前50个): {decoded_tokens[:50]}...")

        # 从层级特定的token中提取语义ID
        # 格式: <id_l1_X>, <id_l2_Y>, <id_l3_Z>
        semantic_id_tuples = []
        i = 0
        while i < len(decoded_tokens):
            token = decoded_tokens[i]
            
            # 检查是否为第1层token
            if token.startswith("<id_l1_"):
                # 尝试提取完整的3层语义ID
                if i + 2 < len(decoded_tokens):
                    l1_token = decoded_tokens[i]
                    l2_token = decoded_tokens[i + 1]
                    l3_token = decoded_tokens[i + 2]
                    
                    # 验证三个都是语义ID token
                    if (l1_token.startswith("<id_l1_") and 
                        l2_token.startswith("<id_l2_") and 
                        l3_token.startswith("<id_l3_")):
                        try:
                            l1_id = int(l1_token.split('_')[2].rstrip('>'))
                            l2_id = int(l2_token.split('_')[2].rstrip('>'))
                            l3_id = int(l3_token.split('_')[2].rstrip('>'))
                            semantic_id_tuples.append((l1_id, l2_id, l3_id))
                            i += 3
                            continue
                        except (ValueError, IndexError):
                            pass
            i += 1
        
        logger.info(f"提取了 {len(semantic_id_tuples)} 个语义ID元组")
        
        # 去重同时保持顺序
        unique_semantic_ids = list(dict.fromkeys(semantic_id_tuples))
        logger.info(f"唯一语义ID: {len(unique_semantic_ids)}")

        # 对每个唯一的语义ID，从其簇中随机采样一首歌
        reconstructed_song_ids = []
        for id_tuple in unique_semantic_ids:
            if id_tuple in self.semantic_to_song_cluster:
                song_cluster = self.semantic_to_song_cluster[id_tuple]
                # 从簇中随机采样一首歌
                sampled_song = random.choice(song_cluster)
                reconstructed_song_ids.append(sampled_song)
                
                # 如果达到最大歌曲数则停止
                if len(reconstructed_song_ids) >= max_songs:
                    break
            else:
                logger.debug(f"语义ID {id_tuple} 在簇映射中未找到")
        
        logger.info(f"生成了 {len(reconstructed_song_ids)} 首歌曲")
        return reconstructed_song_ids

    def interactive_demo(self):
        """启动交互式命令行演示"""
        print("\n" + "="*60)
        print("  🎵 T5歌单生成模型 - 交互式演示 🎵")
        print("="*60)
        print("  输入歌单标题或描述，模型会为您生成个性化歌单。")
        print("  模型会生成语义ID序列，然后从相似歌曲簇中随机采样。")
        print("  每次生成的歌单可能不同，体现了多样性！")
        print("  ")
        print("  命令:")
        print("    - 直接输入文本: 生成歌单")
        print("    - 'exit' 或 'quit': 退出程序")
        print("-"*60)

        while True:
            try:
                prompt = input("\n请输入歌单标题/描述 > ")
                if prompt.lower() in ['exit', 'quit']:
                    print("\n感谢使用，再见！👋")
                    break
                
                if not prompt.strip(): 
                    continue

                print("\n🎼 生成中，请稍候...")
                song_ids = self.generate(prompt.strip())

                if not song_ids:
                    print("❌ 模型未能生成有效的歌曲列表，请尝试更换标题或描述。")
                    continue
                
                print(f"\n✨ 为您推荐的歌单 (共{len(song_ids)}首): ✨")
                print("-"*60)
                for i, song_id in enumerate(song_ids, 1):
                    info = self.song_info_map.get(song_id, {"name": "未知歌曲", "singer": "未知歌手"})
                    sem_id = self._get_sem_id_for_song(song_id)
                    cluster_size = len(self.semantic_to_song_cluster.get(sem_id, [])) if sem_id else 0
                    print(f"  {i:2d}. {info['name']} - {info['singer']}")
                    if cluster_size > 1:
                        print(f"      (来自包含{cluster_size}首相似歌曲的簇)")
                print("-"*60)

            except KeyboardInterrupt:
                print("\n\n感谢使用，再见！👋")
                break
            except Exception as e:
                logger.error(f"生成过程中出错: {e}", exc_info=True)
                print(f"\n❌ 生成过程中出现错误: {e}")
                print("请重试或输入 'exit' 退出。")

    def _get_sem_id_for_song(self, song_id_to_find: str) -> Tuple[int, ...]:
        """辅助函数：查找给定歌曲ID的语义ID（用于显示）"""
        for sem_id, song_list in self.semantic_to_song_cluster.items():
            if song_id_to_find in song_list:
                return sem_id
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T5歌单生成模型推理")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=None,
        help="模型路径 (默认: models/generator/final_model)"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        default=None,
        help="直接生成歌单的提示文本 (如果不提供则进入交互模式)"
    )
    parser.add_argument(
        "--max_songs", 
        type=int, 
        default=20,
        help="最大生成歌曲数量 (默认: 20)"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.8,
        help="采样温度，越高越多样化 (默认: 0.8)"
    )
    parser.add_argument(
        "--log_level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别 (默认: INFO)"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    log_level = getattr(logging, args.log_level)
    setup_logging(level=log_level)
    logger = logging.getLogger(__name__)
    
    # 加载配置
    config = Config()
    
    # 创建生成器
    logger.info("正在初始化歌单生成器...")
    generator = PlaylistGenerator(config, model_path=args.model_path)
    
    # 生成或启动交互模式
    if args.prompt:
        # 单次生成模式
        logger.info(f"正在为以下内容生成歌单: '{args.prompt}'")
        song_ids = generator.generate(
            args.prompt, 
            max_songs=args.max_songs,
            temperature=args.temperature
        )
        
        if song_ids:
            print(f"\n生成的歌单 (共{len(song_ids)}首):")
            print("="*60)
            for i, song_id in enumerate(song_ids, 1):
                info = generator.song_info_map.get(song_id, {"name": "未知歌曲", "singer": "未知歌手"})
                print(f"{i:2d}. {info['name']} - {info['singer']}")
            print("="*60)
        else:
            print("未能生成有效的歌单，请尝试其他提示文本。")
    else:
        # 交互模式
        generator.interactive_demo()