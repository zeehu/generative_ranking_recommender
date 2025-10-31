"""
Step G1: 使用HierarchicalRQKMeans训练语义ID    

这个脚本从config.py中读取参数，使用改造后的HierarchicalRQKMeans类
训练层次化的RQ-KMeans模型，生成歌曲的语义ID。

支持两种模式：
1. 生产模式（PROD）：使用完整数据集
2. 测试模式（TEST）：只加载10w条数据用于快速测试
"""

import os
import sys
import json
import logging
import numpy as np
import csv
import torch
from pathlib import Path
from typing import Tuple, Dict, List
from tqdm import tqdm

# 添加项目根目录到sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config
from src.common.utils import setup_logging
from src.semantic_id_generator.hierarchical_rq_kmeans import HierarchicalRQKMeans, HierarchicalRQKMeansConfig

logger = logging.getLogger(__name__)


class SemanticIDTrainer:
    """语义ID训练器"""
    
    def __init__(self, config: Config, use_test_config: bool = False):
        """
        初始化语义ID训练器
        
        Args:
            config: 主配置对象
            use_test_config: 是否使用测试配置
        """
        self.config = config
        self.use_test_config = use_test_config
        
        # 选择配置
        if use_test_config:
            self.rqkmeans_config = config.h_rqkmeans_test
            logger.info("Using TEST configuration")
        else:
            self.rqkmeans_config = config.h_rqkmeans
            logger.info("Using PROD configuration")
        
        # 设置输出目录
        self.output_dir = Path(config.output_dir) / "semantic_id"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置模型和检查点目录
        self.model_dir = Path(config.model_dir) / "semantic_id"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Model directory: {self.model_dir}")
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
    
    def load_song_vectors(self, max_samples: int = None) -> Tuple[List[str], torch.Tensor]:
        """
        加载歌曲向量，参考simplified_semantic_id_generator.py的_load_data方法
        格式：song_id,dim_1,dim_2,...,dim_N
        
        Args:
            max_samples: 最大加载样本数，如果为None则加载全部
        
        Returns:
            (歌曲ID列表, 向量张量)
        """
        vector_file = self.config.data.song_vectors_file
        
        logger.info(f"Loading song vectors from {vector_file}...")
        if max_samples:
            logger.info(f"NOTE: Loading only the first {max_samples} rows for testing.")
        
        if not os.path.isfile(vector_file):
            logger.error(f"FATAL: Song vector file not found at {vector_file}")
            logger.error("Please run Step G0 (train_word2vec.py) first.")
            raise FileNotFoundError(f"Song vector file not found: {vector_file}")
        
        try:
            song_ids, embeddings = [], []
            
            with open(vector_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for i, row in enumerate(tqdm(reader, desc="Reading rows from CSV")):
                    if max_samples and i >= max_samples:
                        logger.info(f"Data loading limit of {max_samples} rows reached.")
                        break
                    
                    if len(row) < 2:
                        continue
                    
                    song_id = row[0]
                    try:
                        embed = np.array(row[1:], dtype=np.float32)
                    except ValueError:
                        logger.warning(f"Skipping row for song_id {song_id} due to non-numeric vector data.")
                        continue
                    
                    if embed.shape[0] == self.rqkmeans_config.embedding_dim:
                        song_ids.append(song_id)
                        embeddings.append(embed)
            
            if not song_ids:
                raise ValueError("No valid data with the correct embedding dimension found in the CSV file.")
            
            logger.info(f"Successfully loaded {len(song_ids)} song vectors")
            logger.info(f"Vector shape: ({len(song_ids)}, {self.rqkmeans_config.embedding_dim})")
            
            # 转换为张量
            use_half = any(n > 512 for n in self.rqkmeans_config.layer_clusters)
            tensor_embeddings = torch.from_numpy(np.vstack(embeddings))
            return song_ids, tensor_embeddings.half() if use_half else tensor_embeddings.float()
            
        except Exception as e:
            logger.error(f"Error loading song vectors: {str(e)}")
            raise
    
    def train(self, resume: bool = True) -> Dict:
        """
        训练语义ID模型
        
        Args:
            resume: 是否从上次中断的地方继续训练
        
        Returns:
            训练结果字典
        """
        logger.info("="*80)
        logger.info("Starting Semantic ID Training")
        logger.info("="*80)
        
        # 加载数据
        max_samples = 100000 if self.use_test_config else None
        song_ids, vectors = self.load_song_vectors(max_samples=max_samples)
        
        # 转换为numpy数组用于训练
        vectors_np = vectors.cpu().numpy() if isinstance(vectors, torch.Tensor) else vectors
        
        # 创建模型
        logger.info("Creating HierarchicalRQKMeans model...")
        model = HierarchicalRQKMeans(
            config=self.rqkmeans_config,
            checkpoint_dir=str(self.checkpoint_dir),
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        )
        
        # 检查训练状态
        status = model.get_training_status()
        logger.info(f"Training status: {status}")
        
        # 训练模型
        logger.info("Starting model training...")
        try:
            train_result = model.train(vectors_np, resume=resume)
            logger.info("Model training completed successfully")
        except Exception as e:
            import traceback
            logger.error(f"Error during training: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            logger.info("Checkpoint saved. You can resume training later.")
            raise
        
        # 保存模型
        logger.info(f"Saving model to {self.model_dir}...")
        model.save_model(str(self.model_dir))
        
        # 保存配置
        self._save_config(model)
        
        # 生成语义ID
        logger.info("Generating semantic IDs...")
        semantic_ids = self._generate_semantic_ids(song_ids, train_result)
        
        # 保存语义ID
        logger.info("Saving semantic IDs...")
        self._save_semantic_ids(semantic_ids)
        
        # 生成统计信息
        logger.info("Generating statistics...")
        stats = self._generate_statistics(semantic_ids)
        self._save_statistics(stats)
        
        logger.info("="*80)
        logger.info("Semantic ID Training Completed Successfully")
        logger.info("="*80)
        
        return {
            'model': model,
            'semantic_ids': semantic_ids,
            'statistics': stats,
            'song_ids': song_ids
        }
    
    def _generate_semantic_ids(self, song_ids: list, train_result: Dict) -> Dict:
        """
        生成语义ID
        
        Args:
            song_ids: 歌曲ID列表
            train_result: 训练结果
        
        Returns:
            语义ID字典 {song_id: [id1, id2, id3]}
        """
        semantic_ids = {}
        
        # 获取每层的聚类ID
        cluster_ids_list = train_result['cluster_ids']
        
        # 组合每层的聚类ID
        for i, song_id in enumerate(song_ids):
            semantic_id = []
            for layer in range(len(cluster_ids_list)):
                cluster_id = cluster_ids_list[layer][i]
                if isinstance(cluster_id, torch.Tensor):
                    cluster_id = cluster_id.item()
                semantic_id.append(int(cluster_id))
            
            semantic_ids[song_id] = semantic_id
        
        logger.info(f"Generated semantic IDs for {len(semantic_ids)} songs")
        return semantic_ids
    
    def _save_semantic_ids(self, semantic_ids: Dict):
        """
        保存语义ID到JSONL文件，参考simplified_semantic_id_generator.py的save_semantic_ids方法
        格式：{"song_id": "xxx", "semantic_ids": [id1, id2, id3]}
        
        Args:
            semantic_ids: 语义ID字典
        """
        # 保存到config指定的路径
        output_file = self.config.data.semantic_ids_file
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Saving semantic IDs to {output_file}...")
        
        unique_ids = set()
        with open(output_file, 'w', encoding='utf-8') as f:
            for song_id, semantic_id in semantic_ids.items():
                # 创建JSON对象
                data = {"song_id": song_id, "semantic_ids": semantic_id}
                # 写入JSON字符串，后跟换行符
                f.write(json.dumps(data) + '\n')
                unique_ids.add(tuple(semantic_id))
        
        logger.info(f"Saved {len(semantic_ids)} total IDs.")
        logger.info(f"Found {len(unique_ids)} unique semantic IDs.")
    
    def _save_config(self, model: HierarchicalRQKMeans):
        """
        保存配置信息
        
        Args:
            model: 训练好的模型
        """
        config_file = self.output_dir / "training_config.json"
        
        config_dict = {
            'layer_clusters': model.config.layer_clusters,
            'need_clusters': model.config.need_clusters,
            'embedding_dim': model.config.embedding_dim,
            'group_dims': model.config.group_dims,
            'hierarchical_weights': model.config.hierarchical_weights,
            'iter_limit': model.config.iter_limit,
            'use_test_config': self.use_test_config,
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Training config saved to {config_file}")
    
    def _generate_statistics(self, semantic_ids: Dict) -> Dict:
        """
        生成统计信息
        
        Args:
            semantic_ids: 语义ID字典
        
        Returns:
            统计信息字典
        """
        stats = {
            'total_songs': len(semantic_ids),
            'unique_semantic_ids': len(set(tuple(sid) for sid in semantic_ids.values())),
            'layer_statistics': []
        }
        
        # 统计每层的聚类分布
        for layer in range(len(self.rqkmeans_config.layer_clusters)):
            layer_ids = [sid[layer] for sid in semantic_ids.values()]
            unique_count = len(set(layer_ids))
            
            layer_stat = {
                'layer': layer + 1,
                'unique_clusters': unique_count,
                'expected_clusters': self.rqkmeans_config.need_clusters[layer],
                'min_cluster_id': min(layer_ids),
                'max_cluster_id': max(layer_ids),
            }
            
            # 计算聚类分布的统计信息
            cluster_counts = np.bincount(layer_ids)
            layer_stat['cluster_distribution'] = {
                'min': int(cluster_counts.min()),
                'max': int(cluster_counts.max()),
                'mean': float(cluster_counts.mean()),
                'std': float(cluster_counts.std()),
            }
            
            stats['layer_statistics'].append(layer_stat)
        
        logger.info(f"Total unique semantic IDs: {stats['unique_semantic_ids']}")
        logger.info(f"Total songs: {stats['total_songs']}")
        
        return stats
    
    def _save_statistics(self, stats: Dict):
        """
        保存统计信息
        
        Args:
            stats: 统计信息字典
        """
        stats_file = self.output_dir / "training_statistics.json"
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Statistics saved to {stats_file}")
        
        # 打印统计信息
        logger.info("\n" + "="*80)
        logger.info("Training Statistics")
        logger.info("="*80)
        logger.info(f"Total songs: {stats['total_songs']}")
        logger.info(f"Unique semantic IDs: {stats['unique_semantic_ids']}")
        logger.info(f"Semantic ID coverage: {stats['unique_semantic_ids'] / stats['total_songs'] * 100:.2f}%")
        
        for layer_stat in stats['layer_statistics']:
            logger.info(f"\nLayer {layer_stat['layer']}:")
            logger.info(f"  Unique clusters: {layer_stat['unique_clusters']}/{layer_stat['expected_clusters']}")
            logger.info(f"  Cluster ID range: [{layer_stat['min_cluster_id']}, {layer_stat['max_cluster_id']}]")
            logger.info(f"  Cluster distribution:")
            logger.info(f"    Min: {layer_stat['cluster_distribution']['min']}")
            logger.info(f"    Max: {layer_stat['cluster_distribution']['max']}")
            logger.info(f"    Mean: {layer_stat['cluster_distribution']['mean']:.2f}")
            logger.info(f"    Std: {layer_stat['cluster_distribution']['std']:.2f}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train semantic IDs using HierarchicalRQKMeans'
    )
    parser.add_argument(
        '--mode',
        choices=['prod', 'test'],
        default='prod',
        help='Training mode: prod (full data) or test (10k samples)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        default=True,
        help='Resume training from last checkpoint'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Do not resume training, start from scratch'
    )
    parser.add_argument(
        '--clear-checkpoints',
        action='store_true',
        help='Clear all checkpoints before training'
    )
    
    args = parser.parse_args()
    
    # 设置日志
    config = Config()
    log_file_path = os.path.join(config.log_dir, "g1_train_semantic_ids.log")
    setup_logging(log_file=log_file_path)
    
    logger.info("="*80)
    logger.info("Semantic ID Training Script")
    logger.info("="*80)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Resume: {not args.no_resume}")
    
    # 创建训练器
    use_test_config = args.mode == 'test'
    trainer = SemanticIDTrainer(config, use_test_config=use_test_config)
    
    # 清除检查点（如果指定）
    if args.clear_checkpoints:
        logger.info("Clearing all checkpoints...")
        trainer.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for file in trainer.checkpoint_dir.glob("layer_*_checkpoint.pkl"):
            file.unlink()
        if (trainer.checkpoint_dir / "checkpoint_metadata.json").exists():
            (trainer.checkpoint_dir / "checkpoint_metadata.json").unlink()
        logger.info("Checkpoints cleared")
    
    # 训练
    resume = not args.no_resume
    try:
        result = trainer.train(resume=resume)
        logger.info("Training completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.error("You can resume training later using --resume flag")
        return 1


if __name__ == "__main__":
    sys.exit(main())
