"""
T5歌单生成模型推理脚本       
用于加载训练好的T5模型并根据输入文本生成歌曲推荐

支持两种模型加载方式:
1. 从训练检查点加载 (checkpoint目录)
   - 包含文件: config.json, model.safetensors, generation_config.json等
   - 自动检测并加载检查点
   
2. 从最终保存的模型加载 (final_model目录)
   - 使用 TIGERModel.save_pretrained() 保存的模型
   - 包含完整的模型配置和权重

使用示例:
---------
1. 从检查点加载并进入交互模式:
   python src/generator/inference_t5.py --model_path models/generator/checkpoint-1000

2. 从最终模型加载并生成单个歌单:
   python src/generator/inference_t5.py --model_path models/generator/final_model --prompt "适合运动的歌曲"

3. 调整生成参数:
   python src/generator/inference_t5.py --model_path models/generator/checkpoint-1000 --max_songs 30 --temperature 1.0

4. 启用调试日志:
   python src/generator/inference_t5.py --model_path models/generator/checkpoint-1000 --log_level DEBUG
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

from config_optimized import Config
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
        - 如果是检查点目录（包含model.safetensors），则从检查点加载。
        - 如果是最终模型目录，则使用 TIGERModel.from_pretrained。
        """
        if not os.path.exists(self.model_path):
            logger.error(f"错误: 模型路径不存在 {self.model_path}")
            logger.error("请确保路径正确")
            sys.exit(1)

        # 检查目录中的关键文件以判断模型类型
        is_checkpoint = self._is_checkpoint_dir(self.model_path)

        try:
            if is_checkpoint:
                logger.info(f"检测到检查点目录，正在加载: {self.model_path}")
                logger.info(f"检查点包含文件: {os.listdir(self.model_path)}")
                
                # 从主配置重新构建 layer_vocab_sizes
                rq_config = self.config.h_rqkmeans
                layer_vocab_sizes = {
                    'l1': rq_config.need_clusters[0],
                    'l2': rq_config.need_clusters[1],
                    'l3': rq_config.need_clusters[2],
                }
                logger.info(f"使用层级词汇表大小: {layer_vocab_sizes}")
                
                # 检查checkpoint是否包含tokenizer文件
                has_tokenizer = self._has_tokenizer_files(self.model_path)
                
                if not has_tokenizer:
                    logger.warning("检查点目录缺少tokenizer文件（spiece.model等）")
                    logger.info(f"将使用基础模型路径加载tokenizer: {self.config.generator_t5.model_name}")
                    # 使用修改后的加载方式
                    model = self._load_from_checkpoint_without_tokenizer(
                        self.model_path, 
                        self.config.generator_t5.model_name,
                        layer_vocab_sizes
                    )
                else:
                    # 直接实例化TIGERModel，这将从检查点加载基础T5模型
                    # 然后重新应用自定义token和嵌入层大小调整
                    model = TIGERModel(base_model=self.model_path, layer_vocab_sizes=layer_vocab_sizes)
                
                logger.info("检查点加载成功")
            else:
                logger.info(f"正在从 {self.model_path} 加载最终模型 (使用 from_pretrained)...")
                # 对最终保存的模型使用自定义的 from_pretrained 方法
                model = TIGERModel.from_pretrained(self.model_path)
                logger.info("最终模型加载成功")
            
            model.model.to(self.device)
            model.model.eval()
            logger.info(f"模型已移至设备 {self.device} 并设置为评估模式")
            logger.info(f"词汇表大小: {len(model.tokenizer)}")
            return model
        except Exception as e:
            logger.error(f"模型加载失败: {e}", exc_info=True)
            if is_checkpoint:
                logger.error("加载检查点失败。可能的原因:")
                logger.error("  1. 缺少必需文件: config.json, model.safetensors")
                logger.error("  2. 缺少tokenizer文件: spiece.model, tokenizer.json等")
                logger.error("  3. protobuf库未安装: pip install protobuf sentencepiece")
                logger.error(f"\n请检查检查点目录: {self.model_path}")
            else:
                logger.error("加载最终模型失败。请确保模型是使用 TIGERModel.save_pretrained 保存的。")
            sys.exit(1)
    
    def _has_tokenizer_files(self, path: str) -> bool:
        """
        检查目录是否包含tokenizer文件。
        
        Args:
            path: 要检查的目录路径
            
        Returns:
            如果包含tokenizer文件返回True，否则返回False
        """
        if not os.path.isdir(path):
            return False
        
        files = os.listdir(path)
        
        # T5 tokenizer需要的文件
        tokenizer_files = [
            'spiece.model',           # SentencePiece模型文件（必需）
            'tokenizer.json',         # 或者tokenizer配置
            'tokenizer_config.json',  # tokenizer配置
        ]
        
        # 至少需要spiece.model
        has_spiece = 'spiece.model' in files
        
        if has_spiece:
            logger.debug(f"目录 {path} 包含tokenizer文件")
        else:
            logger.debug(f"目录 {path} 缺少tokenizer文件")
        
        return has_spiece
    
    def _load_from_checkpoint_without_tokenizer(self, checkpoint_path: str, 
                                                base_model_path: str,
                                                layer_vocab_sizes: dict) -> TIGERModel:
        """
        从缺少tokenizer文件的checkpoint加载模型。
        使用基础模型的tokenizer，然后加载checkpoint的权重。
        
        Args:
            checkpoint_path: checkpoint目录路径
            base_model_path: 基础模型路径（用于加载tokenizer）
            layer_vocab_sizes: 层级词汇表大小
            
        Returns:
            加载好的TIGERModel
        """
        from transformers import T5ForConditionalGeneration
        from src.generator.tiger_model import TIGERTokenizer
        
        logger.info(f"从基础模型加载tokenizer: {base_model_path}")
        
        # 创建TIGER模型实例，使用基础模型的tokenizer
        tiger_model = TIGERModel.__new__(TIGERModel)
        super(TIGERModel, tiger_model).__init__()
        
        # 初始化tokenizer（从基础模型）
        tiger_model.tokenizer = TIGERTokenizer(base_model_path, layer_vocab_sizes)
        tiger_model.layer_vocab_sizes = layer_vocab_sizes
        tiger_model.base_model_path = base_model_path
        
        # 从checkpoint加载T5模型
        logger.info(f"从checkpoint加载模型权重: {checkpoint_path}")
        tiger_model.model = T5ForConditionalGeneration.from_pretrained(checkpoint_path)
        tiger_model.config = tiger_model.model.config
        
        # 验证词汇表大小
        expected_vocab_size = len(tiger_model.tokenizer)
        actual_vocab_size = tiger_model.model.config.vocab_size
        
        if actual_vocab_size != expected_vocab_size:
            logger.warning(
                f"词汇表大小不匹配: 模型={actual_vocab_size}, tokenizer={expected_vocab_size}"
            )
            logger.info("调整模型嵌入层大小以匹配tokenizer...")
            tiger_model.model.resize_token_embeddings(expected_vocab_size)
        
        logger.info(f"成功加载checkpoint，词汇表大小: {len(tiger_model.tokenizer)}")
        
        return tiger_model
    
    def _is_checkpoint_dir(self, path: str) -> bool:
        """
        判断给定路径是否为训练检查点目录。
        检查点目录通常包含: model.safetensors, config.json, optimizer.pt, scheduler.pt 等。
        
        Args:
            path: 要检查的目录路径
            
        Returns:
            如果是检查点目录返回True，否则返回False
        """
        if not os.path.isdir(path):
            return False
        
        files = os.listdir(path)
        
        # 检查点目录的特征文件
        checkpoint_indicators = [
            'model.safetensors',      # Hugging Face safetensors格式
            'pytorch_model.bin',      # 或传统的PyTorch格式
            'optimizer.pt',           # 优化器状态
            'scheduler.pt',           # 调度器状态
            'trainer_state.json',     # 训练器状态
        ]
        
        # 如果包含任何检查点特征文件，则认为是检查点目录
        has_checkpoint_files = any(f in files for f in checkpoint_indicators)
        
        # 同时检查是否包含必需的模型配置文件
        has_config = 'config.json' in files
        
        # 检查是否包含模型权重文件
        has_model_weights = 'model.safetensors' in files or 'pytorch_model.bin' in files
        
        is_checkpoint = has_checkpoint_files and has_config and has_model_weights
        
        if is_checkpoint:
            logger.debug(f"目录 {path} 被识别为检查点目录")
            logger.debug(f"包含文件: {files}")
        
        return is_checkpoint

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

    def generate(self, title: str, tags: str = "", max_songs: int = 20, temperature: float = 0.8) -> List[Dict]:
        """
        根据标题和标签生成歌单
        
        Args:
            title: 歌单标题/描述
            tags: 可选标签（当前未在生成中使用）
            max_songs: 最大生成歌曲数量
            temperature: 采样温度（越高越多样化）
            
        Returns:
            歌曲信息字典列表，每个字典包含:
            - song_id: 歌曲ID
            - semantic_id: 语义ID元组
            - cluster_songs: 簇中的所有歌曲ID列表
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
        
        # DEBUG: 打印全部唯一语义ID及其生成次数
        if logger.isEnabledFor(logging.DEBUG):
            from collections import Counter
            semantic_id_counts = Counter(semantic_id_tuples)
            
            logger.debug("\n" + "="*100)
            logger.debug("全部唯一语义ID序列及其生成次数:")
            logger.debug("="*100)
            
            # 按生成次数从高到低排序
            sorted_ids = sorted(semantic_id_counts.items(), key=lambda x: x[1], reverse=True)
            
            for rank, (sem_id, count) in enumerate(sorted_ids, 1):
                cluster_size = len(self.semantic_to_song_cluster.get(sem_id, []))
                status = "✓" if sem_id in self.semantic_to_song_cluster else "✗"
                logger.debug(
                    f"{rank:3d}. 语义ID: ({sem_id[0]:3d}, {sem_id[1]:3d}, {sem_id[2]:3d}) | "
                    f"生成次数: {count:3d} | 簇大小: {cluster_size:3d} | {status}"
                )
            
            logger.debug("="*100)
            logger.debug(f"统计信息:")
            logger.debug(f"  - 总生成次数: {len(semantic_id_tuples)}")
            logger.debug(f"  - 唯一语义ID数: {len(unique_semantic_ids)}")
            logger.debug(f"  - 有效语义ID数: {sum(1 for sem_id in unique_semantic_ids if sem_id in self.semantic_to_song_cluster)}")
            logger.debug(f"  - 无效语义ID数: {sum(1 for sem_id in unique_semantic_ids if sem_id not in self.semantic_to_song_cluster)}")
            logger.debug("="*100 + "\n")

        # 对每个唯一的语义ID，从其簇中随机采样一首歌，并保存完整信息
        reconstructed_songs = []
        for id_tuple in unique_semantic_ids:
            if id_tuple in self.semantic_to_song_cluster:
                song_cluster = self.semantic_to_song_cluster[id_tuple]
                # 从簇中随机采样一首歌
                sampled_song = random.choice(song_cluster)
                
                # 保存歌曲信息
                song_info = {
                    'song_id': sampled_song,
                    'semantic_id': id_tuple,
                    'cluster_songs': song_cluster
                }
                reconstructed_songs.append(song_info)
                
                # 如果达到最大歌曲数则停止
                if len(reconstructed_songs) >= max_songs:
                    break
            else:
                logger.debug(f"语义ID {id_tuple} 在簇映射中未找到")
        
        logger.info(f"生成了 {len(reconstructed_songs)} 首歌曲")
        return reconstructed_songs

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
                songs = self.generate(prompt.strip())

                if not songs:
                    print("❌ 模型未能生成有效的歌曲列表，请尝试更换标题或描述。")
                    continue
                
                print(f"\n✨ 为您推荐的歌单 (共{len(songs)}首): ✨")
                print("="*100)
                
                for i, song_data in enumerate(songs, 1):
                    song_id = song_data['song_id']
                    sem_id = song_data['semantic_id']
                    cluster_songs = song_data['cluster_songs']
                    
                    info = self.song_info_map.get(song_id, {"name": "未知歌曲", "singer": "未知歌手"})
                    
                    # 构建主歌曲信息（紧凑格式）
                    main_song = f"{i:2d}. {info['name']} - {info['singer']} - {song_id} - {sem_id[0]}, {sem_id[1]}, {sem_id[2]}"
                    
                    # 如果簇中有多首歌曲，添加簇信息（最多显示4首其他歌曲）
                    if len(cluster_songs) > 1:
                        other_songs = [s for s in cluster_songs if s != song_id]
                        cluster_info_parts = []
                        
                        for other_song_id in other_songs[:4]:
                            other_info = self.song_info_map.get(other_song_id, {"name": "未知", "singer": "未知"})
                            # 获取该歌曲的语义ID（应该和主歌曲相同）
                            cluster_info_parts.append(f"{other_info['name']} - {other_info['singer']} - {other_song_id}")
                        
                        if cluster_info_parts:
                            cluster_str = "; ".join(cluster_info_parts)
                            if len(other_songs) > 4:
                                cluster_str += f"; ... 还有{len(other_songs)-4}首"
                            print(f"{main_song} ({cluster_str})")
                        else:
                            print(main_song)
                    else:
                        print(main_song)
                
                print("="*100)

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
        songs = generator.generate(
            args.prompt, 
            max_songs=args.max_songs,
            temperature=args.temperature
        )
        
        if songs:
            print(f"\n生成的歌单 (共{len(songs)}首):")
            print("="*100)
            
            for i, song_data in enumerate(songs, 1):
                song_id = song_data['song_id']
                sem_id = song_data['semantic_id']
                cluster_songs = song_data['cluster_songs']
                
                info = generator.song_info_map.get(song_id, {"name": "未知歌曲", "singer": "未知歌手"})
                
                # 构建主歌曲信息（紧凑格式）
                main_song = f"{i:2d}. {info['name']} - {info['singer']} - {song_id} - {sem_id[0]}, {sem_id[1]}, {sem_id[2]}"
                
                # 如果簇中有多首歌曲，添加簇信息（最多显示4首其他歌曲）
                if len(cluster_songs) > 1:
                    other_songs = [s for s in cluster_songs if s != song_id]
                    cluster_info_parts = []
                    
                    for other_song_id in other_songs[:4]:
                        other_info = generator.song_info_map.get(other_song_id, {"name": "未知", "singer": "未知"})
                        # 获取该歌曲的语义ID（应该和主歌曲相同）
                        cluster_info_parts.append(f"{other_info['name']} - {other_info['singer']} - {other_song_id}")
                    
                    if cluster_info_parts:
                        cluster_str = "; ".join(cluster_info_parts)
                        if len(other_songs) > 4:
                            cluster_str += f"; ... 还有{len(other_songs)-4}首"
                        print(f"{main_song} ({cluster_str})")
                    else:
                        print(main_song)
                else:
                    print(main_song)
            
            print("="*100)
        else:
            print("未能生成有效的歌单，请尝试其他提示文本。")
    else:
        # 交互模式
        generator.interactive_demo()