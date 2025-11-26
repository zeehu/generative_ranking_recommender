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

from config_optimized import Config
from src.generator.tiger_model import TIGERModel
from src.generator.semantic_id_trie import SemanticIDTrie, ConstrainedLogitsProcessor
from src.common.utils import setup_logging

logger = logging.getLogger(__name__)


class PlaylistGenerator:
    """处理模型加载和根据文本提示生成歌单"""

    def __init__(self, config: Config, model_path: str = None, use_trie_constraint: bool = True):
        """
        初始化歌单生成器
        
        Args:
            config: 配置对象
            model_path: 模型路径，如果为None则使用默认路径
            use_trie_constraint: 是否使用Trie树约束生成（默认True）
        """
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        if model_path is None:
            model_path = os.path.join(self.config.model_dir, "generator", "final_model")
        self.model_path = model_path
        
        self.model = self._load_model()
        self.semantic_to_song_cluster = self._create_reverse_map()
        self.song_info_map = self._load_song_info()
        
        # 初始化Trie树约束（如果启用）
        self.use_trie_constraint = use_trie_constraint
        self.trie = None
        self.constrained_processor = None
        
        if self.use_trie_constraint:
            self._init_trie_constraint()

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

        is_checkpoint = "checkpoint" in os.path.basename(os.path.normpath(self.model_path))

        try:
            if is_checkpoint:
                logger.info(f"检测到检查点目录，使用 __init__ 方法加载: {self.model_path}")
                rq_config = self.config.h_rqkmeans
                layer_vocab_sizes = {
                    'l1': rq_config.need_clusters[0],
                    'l2': rq_config.need_clusters[1],
                    'l3': rq_config.need_clusters[2],
                }
                model = TIGERModel(base_model=self.model_path, layer_vocab_sizes=layer_vocab_sizes)
            else:
                logger.info(f"正在从 {self.model_path} 加载最终模型 (使用 from_pretrained)...")
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
                for row in reader:
                    if len(row) >= 3: 
                        mapping[row[0]] = {"name": row[1], "singer": row[2]}
            logger.info(f"已加载 {len(mapping)} 首歌曲的信息")
        except FileNotFoundError: 
            logger.warning(f"歌曲信息文件未找到: {self.config.data.song_info_file}")
        return mapping
    
    def _init_trie_constraint(self):
        """初始化Trie树约束"""
        try:
            semantic_ids_file = os.path.join(self.config.output_dir, "semantic_id", "song_semantic_ids.jsonl")
            logger.info("正在初始化Trie树约束...")
            
            # 构建Trie树
            self.trie = SemanticIDTrie(self.model.tokenizer, semantic_ids_file)
            
            # 创建约束处理器
            self.constrained_processor = ConstrainedLogitsProcessor(
                self.trie, 
                self.model.tokenizer, 
                self.model.tokenizer.eos_token_id
            )
            
            # 打印统计信息
            stats = self.trie.get_statistics()
            logger.info(f"Trie树统计信息:")
            logger.info(f"  - 有效语义ID序列总数: {stats['total_valid_sequences']}")
            logger.info(f"  - 唯一L1 token数量: {stats['unique_l1_tokens']}")
            logger.info(f"  - L2分布: {dict(stats['l2_distribution'])}")
            logger.info(f"  - L3分布: {dict(stats['l3_distribution'])}")
            logger.info("Trie树约束初始化成功！")
            
        except Exception as e:
            logger.warning(f"初始化Trie树约束失败: {e}")
            logger.warning("将使用无约束生成模式")
            self.use_trie_constraint = False
            self.trie = None
            self.constrained_processor = None

    def generate(self, title: str, tags: str = "", max_songs: int = 20, 
                 do_sample: bool = False, num_beams: int = 1, temperature: float = 1.0, 
                 top_k: int = 50, top_p: float = 1.0) -> List[Dict]:
        """
        根据标题和标签生成歌单，并返回结构化的推荐信息。
        
        Args:
            title: 歌单标题/描述
            tags: 可选标签
            max_songs: 最大生成歌曲数量
            do_sample: 是否使用采样
            num_beams: Beam search数量
            temperature: 采样温度
            top_k: Top-k采样
            top_p: Top-p采样
            
        Returns:
            一个字典列表，每个字典包含主歌曲、同簇歌曲、语义ID和生成次数等信息。
        """
        prompt = title
        logger.info(f"正在生成歌单，提示: '{prompt}'")

        # --- 使用配置文件中的种子固定随机性以保证可复现性 ---
        seed = self.config.seed
        logger.info(f"使用固定随机种子: {seed}")
        torch.manual_seed(seed)
        # --- 结束 ---

        input_ids = self.model.tokenizer.base_tokenizer(
            prompt, 
            return_tensors="pt",
            max_length=self.config.generator_t5.max_input_length,
            truncation=True
        ).input_ids.to(self.device)

        gen_kwargs = {
            "max_new_tokens": self.config.generator_t5.max_target_length,
            "pad_token_id": self.model.tokenizer.pad_token_id,
            "num_return_sequences": 1,
        }

        # --- Dynamically build generation arguments ---
        if do_sample:
            gen_kwargs["do_sample"] = True
            gen_kwargs["top_k"] = top_k
            gen_kwargs["top_p"] = top_p
            gen_kwargs["temperature"] = temperature
            logger.info(f"使用采样解码策略 (Top-K: {top_k}, Top-P: {top_p}, Temp: {temperature})")
        else:
            gen_kwargs["do_sample"] = False
            gen_kwargs["num_beams"] = num_beams
            if num_beams > 1:
                logger.info(f"使用Beam Search解码策略 (Beams: {num_beams})")
            else:
                logger.info("使用Greedy Search解码策略")
        
        # 添加Trie树约束（如果启用）
        if self.use_trie_constraint and self.constrained_processor is not None:
            gen_kwargs["logits_processor"] = [self.constrained_processor]
            logger.info(f"使用Trie树约束")
        else:
            logger.info(f"使用无约束生成")

        with torch.no_grad():
            generated_ids = self.model.model.generate(input_ids, **gen_kwargs)
        
        decoded_tokens = self.model.tokenizer.base_tokenizer.convert_ids_to_tokens(
            generated_ids[0], 
            skip_special_tokens=False
        )
        
        logger.debug(f"生成的token (前50个): {decoded_tokens[:50]}...")

        results = self._process_generated_tokens(decoded_tokens, max_songs)
        return results

    def generate_batch(self, titles: List[str], max_songs: int = 20, 
                       do_sample: bool = False, num_beams: int = 1, temperature: float = 1.0, 
                       top_k: int = 50, top_p: float = 1.0) -> List[List[Dict]]:
        """
        批量生成歌单。
        """
        logger.info(f"正在批量生成歌单，批次大小: {len(titles)}")

        # Tokenize batch
        input_encodings = self.model.tokenizer.base_tokenizer(
            titles, 
            return_tensors="pt",
            max_length=self.config.generator_t5.max_input_length,
            truncation=True,
            padding=True
        ).to(self.device)

        input_ids = input_encodings.input_ids
        attention_mask = input_encodings.attention_mask

        gen_kwargs = {
            "max_new_tokens": self.config.generator_t5.max_target_length,
            "pad_token_id": self.model.tokenizer.pad_token_id,
            "num_return_sequences": 1,
        }

        if do_sample:
            gen_kwargs["do_sample"] = True
            gen_kwargs["top_k"] = top_k
            gen_kwargs["top_p"] = top_p
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False
            gen_kwargs["num_beams"] = num_beams
        
        if self.use_trie_constraint and self.constrained_processor is not None:
            gen_kwargs["logits_processor"] = [self.constrained_processor]

        with torch.no_grad():
            generated_ids = self.model.model.generate(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                **gen_kwargs
            )
        
        batch_results = []
        for i in range(len(titles)):
            decoded_tokens = self.model.tokenizer.base_tokenizer.convert_ids_to_tokens(
                generated_ids[i], 
                skip_special_tokens=False
            )
            results = self._process_generated_tokens(decoded_tokens, max_songs)
            batch_results.append(results)
            
        return batch_results

    def _process_generated_tokens(self, decoded_tokens: List[str], max_songs: int) -> List[Dict]:
        """处理生成的token序列，提取语义ID并映射为歌曲"""
        semantic_id_tuples = []
        i = 0
        while i < len(decoded_tokens):
            token = decoded_tokens[i]
            if token.startswith("<id_l1_"):
                if i + 2 < len(decoded_tokens):
                    l1_token, l2_token, l3_token = decoded_tokens[i], decoded_tokens[i+1], decoded_tokens[i+2]
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
        
        # 如果使用了Trie树约束，验证生成的语义ID是否都有效
        if self.use_trie_constraint and self.trie is not None:
            invalid_count = 0
            for id_tuple in semantic_id_tuples:
                if id_tuple not in self.trie.valid_semantic_ids:
                    invalid_count += 1
            
            if invalid_count > 0:
                logger.warning(f"生成了 {invalid_count} 个无效的语义ID（共{len(semantic_id_tuples)}个）")

        id_stats = {}
        for i, id_tuple in enumerate(semantic_id_tuples):
            if id_tuple not in id_stats:
                id_stats[id_tuple] = {"count": 1, "first_index": i}
            else:
                id_stats[id_tuple]["count"] += 1
        
        sorted_stats = sorted(
            id_stats.items(), 
            key=lambda item: (-item[1]['count'], item[1]['first_index'])
        )
        
        results = []
        for id_tuple, stats in sorted_stats:
            if id_tuple in self.semantic_to_song_cluster:
                song_cluster = self.semantic_to_song_cluster[id_tuple]
                
                sorted_cluster = sorted(song_cluster)
                primary_song_id = sorted_cluster[0]
                similar_song_ids = sorted_cluster[1:6]

                primary_song_info = self.song_info_map.get(primary_song_id, {"name": "未知歌曲", "singer": "未知歌手"})
                similar_songs_info = [
                    {"id": song_id, "info": self.song_info_map.get(song_id, {"name": "未知歌曲", "singer": "未知歌手"})}
                    for song_id in similar_song_ids
                ]

                results.append({
                    "primary_song_id": primary_song_id,
                    "primary_song_info": primary_song_info,
                    "semantic_id": id_tuple,
                    "cluster_size": len(song_cluster),
                    "generation_count": stats['count'],
                    "similar_songs": similar_songs_info
                })

                if len(results) >= max_songs:
                    break
        
        return results

    def _format_song_string(self, song_id: str, song_info: dict) -> str:
        """辅助函数，格式化单曲的输出字符串"""
        name = song_info.get("name", "未知歌曲")
        singer = song_info.get("singer", "未知歌手")
        return f"{song_id}-{name}-{singer}"

    def interactive_demo(self, **kwargs):
        """
        启动交互式命令行演示。
        会使用命令行传入的解码参数。
        """
        print("\n" + "="*80)
        print("  🎵 T5歌单生成模型 - 交互式演示 🎵")
        print("="*80)
        constraint_mode = "Trie树约束" if self.use_trie_constraint else "无约束"
        
        # 从kwargs获取解码设置用于显示
        do_sample = kwargs.get('do_sample', False)
        if do_sample:
            decode_strategy = f"采样 (Temp: {kwargs.get('temperature', 1.0)}, Top-K: {kwargs.get('top_k', 50)}, Top-P: {kwargs.get('top_p', 1.0)})"
        else:
            num_beams = kwargs.get('num_beams', 1)
            decode_strategy = f"Beam Search (Beams: {num_beams})" if num_beams > 1 else "Greedy Search"

        print(f"  推理模式: {constraint_mode} + {decode_strategy}")
        print("  命令: 'exit' 或 'quit' 退出程序")
        print("-"*80)

        while True:
            try:
                prompt = input("\n请输入歌单标题/描述 > ").strip()
                if prompt.lower() in ['exit', 'quit']:
                    print("\n感谢使用，再见！👋")
                    break

                if not prompt: 
                    continue

                print("\n🎼 生成中，请稍候...")
                results = self.generate(prompt, **kwargs)

                if not results:
                    print("❌ 模型未能生成有效的歌曲列表，请尝试更换标题或描述。")
                    continue
                
                print(f"\n✨ 为您推荐的歌单 (共{len(results)}首): ✨")
                print("-"*80)
                for i, item in enumerate(results, 1):
                    primary_str = self._format_song_string(item['primary_song_id'], item['primary_song_info'])
                    
                    similar_list = [self._format_song_string(s['id'], s['info']) for s in item['similar_songs']]
                    similar_str = "; ".join(similar_list)
                    
                    line = f"{i:2d}. {str(item['semantic_id']):<18} - {primary_str}"
                    if similar_str:
                        line += f" ({similar_str})"
                    print(line)
                print("-"*80)

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
        "-m", "--model_path", 
        type=str, 
        default=None,
        help="模型路径 (默认: models/generator/final_model)"
    )
    parser.add_argument(
        "-p", "--prompt", 
        type=str, 
        default=None,
        help="直接生成歌单的提示文本 (不提供则进入交互模式)"
    )
    parser.add_argument(
        "--max_songs", 
        type=int, 
        default=20,
        help="最大生成歌曲数量 (默认: 20)"
    )
    parser.add_argument(
        "-l", "--log_level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别 (默认: INFO)"
    )
    parser.add_argument(
        "--no_trie_constraint",
        action="store_true",
        help="禁用Trie树约束生成（默认启用）"
    )

    # --- New Decoding Strategy Arguments ---
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="启用采样模式 (默认禁用，与beam search互斥)"
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="Beam search的beam数量 (默认: 1, 表示Greedy Search)"
    )
    parser.add_argument(
        "-t", "--temperature", 
        type=float, 
        default=1.0,
        help="采样温度 (仅在--do_sample时生效, 默认: 1.0)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k采样 (仅在--do_sample时生效, 默认: 50)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p (nucleus)采样 (仅在--do_sample时生效, 默认: 1.0)"
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
    use_trie = not args.no_trie_constraint
    generator = PlaylistGenerator(config, model_path=args.model_path, use_trie_constraint=use_trie)
    
    # 生成或启动交互模式
    if args.prompt:
        # 单次生成模式
        logger.info(f"正在为以下内容生成歌单: '{args.prompt}'")
        results = generator.generate(
            args.prompt, 
            max_songs=args.max_songs,
            do_sample=args.do_sample,
            num_beams=args.num_beams,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        
        if results:
            print(f"\n生成的歌单 (共{len(results)}首):")
            print("="*80)
            for i, item in enumerate(results, 1):
                primary_str = generator._format_song_string(item['primary_song_id'], item['primary_song_info'])
                
                similar_list = [generator._format_song_string(s['id'], s['info']) for s in item['similar_songs']]
                similar_str = "; ".join(similar_list)
                
                line = f"{i:2d}. {str(item['semantic_id']):<18} - {primary_str}"
                if similar_str:
                    line += f" ({similar_str})"
                print(line)
            print("="*80)
        else:
            print("未能生成有效的歌单，请尝试其他提示文本。")
    else:
        # 交互模式 - 注意：交互模式将使用命令行传入的解码参数
        logger.info("启动交互模式...")
        logger.info(f"交互会话解码参数: do_sample={args.do_sample}, num_beams={args.num_beams}, temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")
        generator.interactive_demo(
            do_sample=args.do_sample,
            num_beams=args.num_beams,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )


    