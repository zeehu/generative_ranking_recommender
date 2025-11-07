"""
改造后的BalanceRqKMeans核心类                                                                                                                                          
支持：
1. 训练和预测的统一接口
2. 单个维度的group_dims和单一均匀权重hierarchical_weights
3. 训练中断恢复，支持从失败层继续训练

核心策略与extreme_hierarchical_rq_cluster.py一致：
- 第1层：直接训练n_clusters个聚类中心
- 第2层：递归聚类，在第1层的每个簇内进行聚类
- 第3层：特殊处理，使用两个KMeans模型，然后通过match_matrix筛选
"""

import os
import json
import pickle
import logging
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass, asdict, field
from tqdm import tqdm
from functools import partial
import time

from .balancekmeans import KMeans, pairwise_distance_full

logger = logging.getLogger(__name__)


@dataclass
class HierarchicalRQKMeansConfig:
    """层次化RQ-KMeans配置"""
    layer_clusters: List[int]  # 每层的聚类中心数量
    need_clusters: List[int]   # 每层实际使用的聚类中心数量
    embedding_dim: int         # 嵌入维度
    group_dims: Union[int, List[int]] = None  # 维度分组，支持单个int或list
    hierarchical_weights: Union[float, List[List[float]]] = None  # 权重系数，支持单个float或list
    iter_limit: int = 100      # 最大迭代次数
    
    def __post_init__(self):
        """初始化和验证配置"""
        # 处理group_dims
        if self.group_dims is None or (isinstance(self.group_dims, list) and len(self.group_dims) == 0):
            self.group_dims = [self.embedding_dim]
        elif isinstance(self.group_dims, int):
            self.group_dims = [self.group_dims]
        
        # 验证group_dims的和等于embedding_dim
        if sum(self.group_dims) != self.embedding_dim:
            raise ValueError(
                f"Sum of group_dims {sum(self.group_dims)} must equal embedding_dim {self.embedding_dim}"
            )
        
        # 处理hierarchical_weights
        if self.hierarchical_weights is None or (isinstance(self.hierarchical_weights, list) and len(self.hierarchical_weights) == 0):
            # 默认使用均匀权重
            self.hierarchical_weights = [
                [1.0 / len(self.group_dims)] * len(self.group_dims)
                for _ in range(len(self.layer_clusters))
            ]
        elif isinstance(self.hierarchical_weights, (int, float)):
            # 单一权重，转换为均匀权重
            self.hierarchical_weights = [
                [1.0 / len(self.group_dims)] * len(self.group_dims)
                for _ in range(len(self.layer_clusters))
            ]
        
        # 验证hierarchical_weights的维度
        if len(self.hierarchical_weights) != len(self.layer_clusters):
            raise ValueError(
                f"Length of hierarchical_weights {len(self.hierarchical_weights)} "
                f"must equal length of layer_clusters {len(self.layer_clusters)}"
            )
        
        for i, weights in enumerate(self.hierarchical_weights):
            if len(weights) != len(self.group_dims):
                raise ValueError(
                    f"Length of hierarchical_weights[{i}] {len(weights)} "
                    f"must equal length of group_dims {len(self.group_dims)}"
                )


class CheckpointManager:
    """管理训练检查点，支持中断恢复"""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"
    
    def save_layer_checkpoint(self, layer: int,
                             cluster_ids: torch.Tensor, residual_data: torch.Tensor,
                             cluster_centers: Optional[torch.Tensor] = None,
                             match_matrix: Optional[List] = None):
        """保存单层的检查点"""
        checkpoint = {
            'layer': layer,
            'cluster_ids': cluster_ids.cpu().numpy() if isinstance(cluster_ids, torch.Tensor) else cluster_ids,
            'residual_data': residual_data.cpu().numpy() if isinstance(residual_data, torch.Tensor) else residual_data,
            'cluster_centers': cluster_centers.cpu().numpy() if isinstance(cluster_centers, torch.Tensor) else cluster_centers,
            'match_matrix': match_matrix,
        }
        
        checkpoint_file = self.checkpoint_dir / f"layer_{layer}_checkpoint.pkl"
        temp_checkpoint_file = self.checkpoint_dir / f"layer_{layer}_checkpoint.tmp"
        
        try:
            with open(temp_checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint, f)
            
            with open(temp_checkpoint_file, 'rb') as f:
                loaded_checkpoint = pickle.load(f)
            
            required_keys = ['cluster_ids', 'cluster_centers']
            for key in required_keys:
                if key not in loaded_checkpoint or loaded_checkpoint[key] is None:
                    raise ValueError(f"Checkpoint validation failed: missing or None key '{key}'")
            
            temp_checkpoint_file.replace(checkpoint_file)
            
        except Exception as e:
            if temp_checkpoint_file.exists():
                try:
                    temp_checkpoint_file.unlink()
                except Exception:
                    pass
            logger.error(f"Failed to save checkpoint for layer {layer}: {str(e)}")
            raise
    
    def load_layer_checkpoint(self, layer: int, device: torch.device) -> Optional[Dict]:
        """加载单层的检查点，并将所有数据转换为torch.Tensor"""
        checkpoint_file = self.checkpoint_dir / f"layer_{layer}_checkpoint.pkl"
        if not checkpoint_file.exists():
            return None
        
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)
        
        if 'cluster_ids' in checkpoint and checkpoint['cluster_ids'] is not None:
            if isinstance(checkpoint['cluster_ids'], np.ndarray):
                checkpoint['cluster_ids'] = torch.from_numpy(checkpoint['cluster_ids']).to(device)
        
        if 'residual_data' in checkpoint and checkpoint['residual_data'] is not None:
            if isinstance(checkpoint['residual_data'], np.ndarray):
                checkpoint['residual_data'] = torch.from_numpy(checkpoint['residual_data']).to(device)
        
        if 'cluster_centers' in checkpoint and checkpoint['cluster_centers'] is not None:
            if isinstance(checkpoint['cluster_centers'], np.ndarray):
                checkpoint['cluster_centers'] = torch.from_numpy(checkpoint['cluster_centers']).to(device)
        
        return checkpoint
    
    def get_last_completed_layer(self) -> int:
        """获取最后完成的层数"""
        completed_layers = []
        for i in range(100):  # 假设最多100层
            if (self.checkpoint_dir / f"layer_{i}_checkpoint.pkl").exists():
                completed_layers.append(i)
            else:
                break
        
        return max(completed_layers) if completed_layers else -1
    
    def save_metadata(self, metadata: Dict):
        """保存元数据"""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def load_metadata(self) -> Optional[Dict]:
        """加载元数据"""
        if not self.metadata_file.exists():
            return None
        
        with open(self.metadata_file, 'r') as f:
            return json.load(f)
    
    def clear_checkpoints(self):
        """清除所有检查点"""
        for file in self.checkpoint_dir.glob("layer_*_checkpoint.pkl"):
            file.unlink()
        if self.metadata_file.exists():
            self.metadata_file.unlink()


class HierarchicalRQKMeans:
    """
    改造后的BalanceRqKMeans核心类
    
    支持：
    1. 统一的训练和预测接口
    2. 单个维度的group_dims和单一均匀权重hierarchical_weights
    3. 训练中断恢复
    
    核心策略与extreme_hierarchical_rq_cluster.py一致
    """
    
    def __init__(self, config: HierarchicalRQKMeansConfig, 
                 checkpoint_dir: Optional[str] = None,
                 device: Optional[torch.device] = None):
        """
        初始化HierarchicalRQKMeans
        
        Args:
            config: HierarchicalRQKMeansConfig配置对象
            checkpoint_dir: 检查点保存目录，如果为None则不保存检查点
            device: 计算设备，默认为cuda:0或cpu
        """
        self.config = config
        self.device = device or self._get_device()
        self.checkpoint_manager = CheckpointManager(checkpoint_dir) if checkpoint_dir else None
        
        # 训练状态
        self.is_trained = False
        self.cluster_centers_list = []  # 保存每层的聚类中心
        self.match_matrices = []  # 保存匹配矩阵
        self.result_cluster_ids = []  # 保存每层的聚类ID

    
    @staticmethod
    def _get_device() -> torch.device:
        """获取计算设备"""
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        return torch.device('cpu')
    
    @staticmethod
    def _calculate_safe_batch_size(X: torch.Tensor, num_centers: int, device: torch.device, 
                                   initial_batch_size: int = 200000) -> int:
        """
        动态计算安全的batch_size，防止OOM
        
        参考extreme版本的逻辑：
        safe_batch = min(100000, int(0.8 * free_memory / (element_size * embedding_dim * 3)))
        
        关键修复：
        1. 使用空闲内存而不是总内存
        2. 精确计算每个样本的内存需求（distance矩阵 + mask矩阵 + 输入数据）
        3. 根据聚类中心数量调整内存使用率
        
        Args:
            X: 输入数据张量
            num_centers: 聚类中心数量
            device: 计算设备
            initial_batch_size: 初始batch_size上限，默认100000
        
        Returns:
            安全的batch_size
        """
        # 获取GPU空闲内存
        free_memory, total_memory = torch.cuda.mem_get_info()
        
        element_size = X.element_size()  # 通常是4 (float32)
        embedding_dim = X.shape[1]
        
        # 精确计算每个样本的内存需求：
        # 1. distance矩阵: num_centers × element_size
        # 2. mask矩阵: num_centers × element_size
        # 3. 输入batch数据: embedding_dim × element_size
        # 4. 临时变量（cluster_assignments等）: 约0.5倍的distance矩阵
        # 总计: (num_centers × 2.5 + embedding_dim) × element_size
        memory_per_sample = (num_centers * 2.5 + embedding_dim) * element_size
        
        # 根据聚类中心数量调整内存使用率
        if num_centers > 10000:
            # 大规模聚类中心（如第2层：16384），使用60%的空闲内存
            memory_usage_ratio = 0.6
        elif num_centers > 5000:
            memory_usage_ratio = 0.7
        else:
            # 参考extreme版本：使用80%的空闲内存
            memory_usage_ratio = 0.8
        
        safe_batch = int(memory_usage_ratio * free_memory / memory_per_sample)
        safe_batch = min(initial_batch_size, safe_batch)
        safe_batch = max(1, safe_batch)
        
        logger.info(
            f"GPU memory: free={free_memory / 1e9:.2f}GB, total={total_memory / 1e9:.2f}GB, "
            f"num_centers={num_centers:,}, embedding_dim={embedding_dim}, "
            f"memory_per_sample={memory_per_sample / 1024:.2f}KB, "
            f"memory_ratio={memory_usage_ratio:.1f}, safe_batch={safe_batch:,}"
        )
        
        return safe_batch
    
    @staticmethod
    def _calculate_adaptive_iter_limit(num_samples: int, n_clusters: int, 
                                       layer: int, base_iter_limit: int = 100,
                                       is_sub_cluster: bool = False) -> int:
        """
        根据数据量和聚类中心数动态计算iter_limit，平衡效果和计算量
        
        优化：
        1. 第2层子簇内聚类使用更少的迭代次数（数据已聚类过，更易收敛）
        2. 提高大数据集的迭代次数
        3. 考虑样本/聚类比例
        
        Args:
            num_samples: 当前层的样本数
            n_clusters: 当前层的聚类中心数
            layer: 层索引（0表示第1层）
            base_iter_limit: 基础迭代次数，默认100
            is_sub_cluster: 是否是子簇内聚类（第2层特有）
        
        Returns:
            自适应的iter_limit
        """
        samples_per_cluster = num_samples / max(n_clusters, 1)
        
        # 优化：第2层子簇内聚类的特殊处理
        if is_sub_cluster:
            # 子簇内聚类：数据已经过第1层聚类，分布更集中，更容易收敛
            if num_samples < 5000:
                iter_limit = 15
            elif num_samples < 10000:
                iter_limit = 20
            elif num_samples < 20000:
                iter_limit = 25
            else:
                iter_limit = 30
            
            # 根据样本/聚类比例微调
            if samples_per_cluster < 50:
                iter_limit = max(10, int(iter_limit * 0.8))  # 样本太少，减少迭代
            elif samples_per_cluster > 200:
                iter_limit = int(iter_limit * 1.2)  # 样本充足，稍微增加
            
            return max(10, iter_limit)
        
        # 第1层和第3层的逻辑
        # 优化：提高大数据集的迭代次数
        if num_samples < 5000:
            iter_limit = max(10, int(base_iter_limit * 0.2))
        elif num_samples < 10000:
            iter_limit = max(15, int(base_iter_limit * 0.3))
        elif num_samples < 50000:
            iter_limit = max(30, int(base_iter_limit * 0.5))
        elif num_samples < 100000:
            iter_limit = max(50, int(base_iter_limit * 0.7))
        elif num_samples < 500000:
            iter_limit = base_iter_limit
        elif num_samples < 1000000:
            iter_limit = int(base_iter_limit * 1.2)  # 新增：中大数据集
        else:
            iter_limit = int(base_iter_limit * 1.5)  # 新增：大数据集需要更多迭代
        
        # 根据聚类中心数调整
        if n_clusters > 512:
            iter_limit = int(iter_limit * 1.3)
        elif n_clusters > 256:
            iter_limit = int(iter_limit * 1.15)
        
        # 优化：减缓深层衰减（从0.8改为0.9）
        if layer > 1:
            iter_limit = max(10, int(iter_limit * 0.9))
        
        # 优化：考虑样本/聚类比例
        if samples_per_cluster < 50:
            iter_limit = int(iter_limit * 1.2)  # 样本太少，需要更多迭代
        
        # 确保最小迭代次数
        iter_limit = max(10, iter_limit)
        
        return iter_limit
    
    def train(self, X: np.ndarray, resume: bool = True) -> Dict:
        """
        训练模型
        
        Args:
            X: 输入数据，shape为(N, embedding_dim)
            resume: 是否从上次中断的地方继续训练
        
        Returns:
            训练结果字典，包含每层的聚类ID和聚类中心
        """
        if X.shape[1] != self.config.embedding_dim:
            raise ValueError(
                f"Input dimension {X.shape[1]} does not match config embedding_dim {self.config.embedding_dim}"
            )
        
        # 记录总体训练开始时间
        total_start_time = time.time()
        logger.info(f"{'='*80}")
        logger.info(f"Starting training with {len(X)} samples")
        logger.info(f"Total layers: {len(self.config.layer_clusters)}")
        logger.info(f"Layer clusters: {self.config.layer_clusters}")
        logger.info(f"Need clusters: {self.config.need_clusters}")
        logger.info(f"Device: {self.device}")
        logger.info(f"{'='*80}")
        
        # 确定起始层
        start_layer = 0
        if resume and self.checkpoint_manager:
            start_layer = self.checkpoint_manager.get_last_completed_layer() + 1
            if start_layer > 0:
                logger.info(f"[RESUME] Resuming training from layer {start_layer}")
                # 加载之前的检查点
                self._load_previous_checkpoints(start_layer)
        
        # 初始化当前数据
        current_data = torch.from_numpy(X.astype('float32')).to(self.device)
        
        # 如果是恢复训练，加载上一层的残差数据
        if start_layer > 0 and self.checkpoint_manager:
            checkpoint = self.checkpoint_manager.load_layer_checkpoint(start_layer - 1, self.device)
            if checkpoint and 'residual_data' in checkpoint:
                current_data = checkpoint['residual_data']
                logger.info(f"[RESUME] Loaded residual data from layer {start_layer - 1}")
        
        # 逐层训练
        for layer in range(start_layer, len(self.config.layer_clusters)):
            layer_start_time = time.time()
            n_clusters = self.config.layer_clusters[layer]
            need_clusters = self.config.need_clusters[layer]
            num_samples = len(current_data)
            
            logger.info(f"\n[LAYER {layer + 1}/{len(self.config.layer_clusters)}] Starting layer training")
            logger.info(f"  - Samples: {num_samples:,}")
            logger.info(f"  - Clusters: {n_clusters} (need: {need_clusters})")
            logger.info(f"  - Data shape: {current_data.shape}")
            
            try:
                # 应用权重
                weight_start = time.time()
                weighted_data = self._apply_weights(current_data, layer)
                weight_time = time.time() - weight_start
                logger.info(f"  - Applied weights in {weight_time:.2f}s")
                
                # 临时存储本层的训练结果（只有成功才会保存到正式列表）
                layer_cluster_centers = None
                layer_cluster_ids = None
                layer_match_matrix = None
                
                # 根据extreme_hierarchical_rq_cluster.py的判断逻辑进行训练
                if n_clusters == need_clusters:
                    # 策略1：直接聚类（通常是第1层）
                    logger.info(f"  - Strategy: Direct clustering (Layer 0)")
                    train_start = time.time()
                    layer_cluster_centers, layer_cluster_ids, layer_residual_data = self._train_layer_0(weighted_data, layer)
                    train_time = time.time() - train_start
                    logger.info(f"  - Layer 0 training completed in {train_time:.2f}s")
                    # 更新残差数据
                    if layer < len(self.config.layer_clusters) - 1:
                        residual_data = layer_residual_data
                    
                elif layer == len(self.config.layer_clusters) - 1:
                    # 策略2：最后一层（两个KMeans + match_matrix）
                    logger.info(f"  - Strategy: Last layer (2 KMeans + match matrix)")
                    train_start = time.time()
                    layer_cluster_centers, layer_cluster_ids, layer_residual_data = self._train_last_layer(
                        weighted_data, layer
                    )
                    train_time = time.time() - train_start
                    if len(self.match_matrices) > 0:
                        layer_match_matrix = self.match_matrices[-1]
                    logger.info(f"  - Last layer training completed in {train_time:.2f}s")
                    # 最后一层也计算残差（虽然不会用到）
                    if layer < len(self.config.layer_clusters) - 1:
                        residual_data = layer_residual_data
                    
                else:
                    # 策略3：中间层（递归聚类）
                    logger.info(f"  - Strategy: Middle layer (recursive clustering)")
                    train_start = time.time()
                    layer_cluster_centers, layer_cluster_ids, layer_residual_data = self._train_middle_layer(
                        weighted_data, layer
                    )
                    train_time = time.time() - train_start
                    logger.info(f"  - Middle layer training completed in {train_time:.2f}s")
                    # 更新残差数据
                    if layer < len(self.config.layer_clusters) - 1:
                        residual_data = layer_residual_data
                
                # 更新内部状态
                self.cluster_centers_list.append(layer_cluster_centers)
                self.result_cluster_ids.append(layer_cluster_ids)
                
                # 只有当整层训练完全成功时，才保存检查点
                if self.checkpoint_manager:
                    try:
                        checkpoint_start = time.time()
                        # 保存checkpoint时使用residual_data（如果是最后一层则使用current_data）
                        checkpoint_residual = residual_data if layer < len(self.config.layer_clusters) - 1 else current_data
                        self.checkpoint_manager.save_layer_checkpoint(
                            layer,
                            layer_cluster_ids,
                            checkpoint_residual,
                            layer_cluster_centers,
                            layer_match_matrix
                        )
                        checkpoint_time = time.time() - checkpoint_start
                        logger.info(f"  - Saved checkpoint in {checkpoint_time:.2f}s")
                    except Exception as e:
                        logger.error(f"Failed to save checkpoint for layer {layer}: {str(e)}")
                        raise
                
                # 更新当前数据为残差数据（用于下一层训练）
                if layer < len(self.config.layer_clusters) - 1:
                    current_data = residual_data
                    logger.info(f"  - Updated current_data to residual_data for next layer")
                
                # 统计本层耗时
                layer_total_time = time.time() - layer_start_time
                logger.info(f"[LAYER {layer + 1}] Completed in {layer_total_time:.2f}s")
                logger.info(f"{'-'*80}")
                
            except Exception as e:
                logger.error(f"Error training layer {layer + 1}: {str(e)}")
                raise
        
        self.is_trained = True
        
        # 保存元数据
        if self.checkpoint_manager:
            metadata = {
                'num_layers': len(self.config.layer_clusters),
                'embedding_dim': self.config.embedding_dim,
                'group_dims': self.config.group_dims,
                'hierarchical_weights': self.config.hierarchical_weights,
                'num_samples': len(X),
            }
            self.checkpoint_manager.save_metadata(metadata)
        
        # 统计总体耗时
        total_time = time.time() - total_start_time
        logger.info(f"\n{'='*80}")
        logger.info(f"[TRAINING COMPLETE] Total time: {total_time:.2f}s ({total_time/60:.2f}m)")
        logger.info(f"Average time per layer: {total_time/len(self.config.layer_clusters):.2f}s")
        logger.info(f"{'='*80}\n")
        
        return {
            'cluster_ids': self.result_cluster_ids,
            'cluster_centers': self.cluster_centers_list,
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测数据的聚类ID
        
        Args:
            X: 输入数据，shape为(N, embedding_dim)
        
        Returns:
            聚类ID数组，shape为(N, num_layers)
        """
        if not self.is_trained or not self.cluster_centers_list:
            raise RuntimeError("Model not trained. Call train() first or load a trained model.")
        
        if X.shape[1] != self.config.embedding_dim:
            raise ValueError(
                f"Input dimension {X.shape[1]} does not match config embedding_dim {self.config.embedding_dim}"
            )
        
        current_data = torch.from_numpy(X.astype('float32')).to(self.device)
        all_cluster_ids = []
        
        for layer in range(len(self.config.layer_clusters)):
            
            # 应用权重
            weighted_data = self._apply_weights(current_data, layer)
            
            # 预测当前层（传入之前层的预测结果）
            if layer == 0:
                cluster_ids = self._predict_layer_0(weighted_data, layer)
            elif layer == len(self.config.layer_clusters) - 1:
                cluster_ids = self._predict_last_layer(weighted_data, layer, all_cluster_ids)
            else:
                cluster_ids = self._predict_middle_layer(weighted_data, layer, all_cluster_ids)
            
            all_cluster_ids.append(cluster_ids)
            
            # 计算残差用于下一层
            if layer < len(self.config.layer_clusters) - 1:
                current_data = self._compute_residuals(current_data, cluster_ids, layer)
        
        # 转换为numpy数组（all_cluster_ids中的元素已是torch.Tensor）
        result = np.column_stack([ids.cpu().numpy() for ids in all_cluster_ids])
        return result
    
    def _apply_weights(self, data: torch.Tensor, layer: int) -> torch.Tensor:
        """
        应用层级权重到数据（优化：避免克隆，使用向量化操作）
        
        Args:
            data: 输入数据
            layer: 层索引
        
        Returns:
            加权后的数据
        """
        weights = self.config.hierarchical_weights[layer]
        
        # 创建权重张量，避免克隆整个数据
        weight_tensor = torch.ones(data.shape[1], device=data.device, dtype=data.dtype)
        cur_idx = 0
        for i, dim in enumerate(self.config.group_dims):
            weight_tensor[cur_idx:cur_idx+dim] = weights[i]
            cur_idx += dim
        
        # 使用广播乘法，避免克隆
        return data * weight_tensor.unsqueeze(0)
    
    def _train_layer_0(self, X: torch.Tensor, layer: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        训练第1层：直接训练n_clusters个聚类中心，并在内部计算残差
        
        严格按照extreme版本的逻辑：
        1. 训练KMeans
        2. 预测聚类ID
        3. 使用相同的聚类ID计算残差（避免重新预测）
        
        Args:
            X: 输入数据张量，shape为(N, embedding_dim)
            layer: 层索引
        
        Returns:
            (cluster_centers, cluster_ids, residual_data): 聚类中心、聚类ID和残差数据
        """
        n_clusters = self.config.layer_clusters[layer]
        
        # 计算目标节点数
        target_nodes_num = 1
        for idx, x in enumerate(self.config.need_clusters):
            if idx != layer:
                target_nodes_num *= x
        
        # 动态计算自适应iter_limit
        adaptive_iter_limit = self._calculate_adaptive_iter_limit(
            len(X), n_clusters, layer, self.config.iter_limit
        )
        logger.info(f"    - Adaptive iter_limit: {adaptive_iter_limit}")
        
        # 创建KMeans模型
        kmeans = KMeans(n_clusters=n_clusters, device=self.device, balanced=True)
        
        # 训练
        kmeans.fit_by_min_loss(
            X=X,
            target_nodes_num=target_nodes_num,
            distance='euclidean',
            iter_limit=adaptive_iter_limit,
            tqdm_flag=True,
            half=n_clusters >= 512,
            online=False
        )
        
        # 获取聚类中心
        cluster_centers = kmeans.cluster_centers.detach()
        
        # 预测聚类ID
        cluster_ids = kmeans.predict(X, balanced=False)
        if isinstance(cluster_ids, np.ndarray):
            cluster_ids = torch.from_numpy(cluster_ids).to(self.device)
        
        # 计算残差（使用相同的cluster_ids，避免重新预测）
        logger.info(f"    - Computing residuals for layer {layer + 1}")
        residual_data = self._compute_residuals_with_centers(
            X, cluster_ids, cluster_centers
        )
        
        # 及时释放KMeans模型
        del kmeans
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return cluster_centers, cluster_ids, residual_data
    
    def _train_middle_layer(self, X: torch.Tensor, layer: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        训练中间层：在前一层的每个簇内递归进行聚类，并在内部计算残差
        
        严格按照extreme版本的逻辑：
        1. 递归训练KMeans（每个父簇内训练）
        2. 重新分配聚类ID（使用未取余的ID）
        3. 使用未取余的ID计算残差
        4. 返回取余后的ID用于保存
        
        Args:
            X: 输入数据张量，shape为(N, embedding_dim)
            layer: 层索引
        
        Returns:
            (kmeans_centers, cluster_ids, residual_data): 聚类中心、聚类ID（已取余）和残差数据
        """
        cur_need_cluster = self.config.need_clusters[layer]
        pre_need_cluster = self.config.need_clusters[layer - 1]
        
        if layer - 1 >= len(self.result_cluster_ids):
            raise RuntimeError(
                f"Previous layer {layer - 1} cluster IDs not found. "
                f"Expected at least {layer} layers but only have {len(self.result_cluster_ids)} layers."
            )
        
        prev_cluster_ids = self.result_cluster_ids[layer - 1]
        
        # 计算目标节点数
        target_nodes_num = 1
        for idx, x in enumerate(self.config.need_clusters):
            if idx > layer:
                target_nodes_num *= x
        
        logger.info(f"    - Training {pre_need_cluster} sub-clusters recursively")
        logger.info(f"    - Each sub-cluster trains {cur_need_cluster} centers (total: {pre_need_cluster * cur_need_cluster})")
        
        # 递归聚类：在前一层的每个簇内进行聚类
        kmeans_centers_list = []
        for i in tqdm(range(pre_need_cluster), desc="    Sub-clusters", leave=False):
            idx = torch.where(prev_cluster_ids == i)[0]
            sub_data = X[idx]
            
            # 优化：传入 is_sub_cluster=True，使用更少的迭代次数
            adaptive_iter_limit = self._calculate_adaptive_iter_limit(
                len(idx), cur_need_cluster, layer, self.config.iter_limit,
                is_sub_cluster=True
            )
            
            sub_kmeans = KMeans(n_clusters=cur_need_cluster, device=self.device, balanced=True)
            sub_kmeans.fit_by_min_loss(
                X=sub_data,
                target_nodes_num=target_nodes_num,
                distance='euclidean',
                iter_limit=adaptive_iter_limit,
                tqdm_flag=True,
                half=cur_need_cluster >= 512,
                online=False
            )
            
            kmeans_centers_list.append(sub_kmeans.cluster_centers.detach())
            
            del sub_kmeans, sub_data, idx
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # 合并所有聚类中心
        kmeans_centers = torch.cat(kmeans_centers_list, dim=0)
        del kmeans_centers_list
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        logger.info(f"    - Reassigning clusters and computing residuals for layer {layer + 1}")
        # 重新分配聚类ID并计算残差（返回未取余的ID和残差）
        cluster_ids_raw, residual_data = self._reassign_clusters_middle_layer_with_residuals(
            X, kmeans_centers, prev_cluster_ids, layer
        )
        
        # 取余得到最终的聚类ID
        cluster_ids = cluster_ids_raw % cur_need_cluster
        
        return kmeans_centers, cluster_ids, residual_data
    
    def _train_last_layer(self, X: torch.Tensor, layer: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        训练最后一层：使用两个KMeans模型，通过match_matrix筛选最优聚类，并在内部计算残差
        
        严格按照extreme版本的逻辑：
        1. 训练2个KMeans模型并合并
        2. 生成match_matrix
        3. 重新分配聚类ID（使用未映射的ID）
        4. 使用未映射的ID计算残差
        5. 映射ID到最终范围
        
        Args:
            X: 输入数据张量，shape为(N, embedding_dim)
            layer: 层索引
        
        Returns:
            (cluster_centers, cluster_ids, residual_data): 聚类中心、聚类ID（已映射）和残差数据
        """
        n_clusters = self.config.layer_clusters[layer]
        need_clusters = self.config.need_clusters[layer]
        
        if len(self.result_cluster_ids) < 2:
            raise RuntimeError(
                f"Previous layers cluster IDs not found. "
                f"Expected at least 2 layers but only have {len(self.result_cluster_ids)} layers."
            )
        
        # 优化：最后一层不要过度降低iter_limit（从0.6改为0.85）
        # 第3层处理残差数据，分布最分散，收敛最困难，需要更多迭代
        # 第3层只需要把把数据打散即可，不需要充分迭代
        #adaptive_iter_limit = max(20, int(self._calculate_adaptive_iter_limit(
        #    len(X), n_clusters, layer, self.config.iter_limit
        #) * 0.85))
        adaptive_iter_limit = 20
        logger.info(f"    - Adaptive iter_limit: {adaptive_iter_limit}")
        
        # 训练两个KMeans模型
        kmeans_centers_list = []
        for part in tqdm(range(2), desc="    KMeans models", leave=False):
            kmeans = KMeans(n_clusters=n_clusters, device=self.device, balanced=True)
            kmeans.fit(
                X=X,
                distance='euclidean',
                iter_limit=adaptive_iter_limit,
                tqdm_flag=True,
                half=n_clusters >= 512,
                online=False
            )
            kmeans_centers_list.append(kmeans.cluster_centers.detach())
            del kmeans
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # 合并聚类中心
        cur_kmeans_centers = torch.cat(kmeans_centers_list, dim=0)
        
        # 构建match_matrix
        prev_prev_cluster_ids = self.result_cluster_ids[-2].cpu().numpy()
        prev_cluster_ids = self.result_cluster_ids[-1].cpu().numpy()
        prev_prev_need_cluster = self.config.need_clusters[layer - 2]
        
        match_matrix = self._assign_last_match_matrix(
            cur_kmeans_centers, 2 * n_clusters, X,
            prev_prev_need_cluster, self.config.need_clusters[layer - 1],
            prev_prev_cluster_ids, prev_cluster_ids,
            need_clusters, 2 * need_clusters, layer
        )
        self.match_matrices.append(match_matrix)
        
        # 计算组合ID
        before_cluster_ids = prev_prev_cluster_ids * prev_prev_need_cluster + prev_cluster_ids
        
        logger.info(f"    - Reassigning clusters and computing residuals for layer {layer + 1}")
        # 重新分配聚类ID并计算残差（使用未映射的ID）
        cluster_ids_raw, residual_data = self._reassign_clusters_last_layer_with_residuals(
            X, cur_kmeans_centers, before_cluster_ids, match_matrix, layer
        )
        
        # 映射到最终的聚类ID
        cluster_ids = self._merge_match_matrix_cluster_ids(
            match_matrix, cluster_ids_raw, before_cluster_ids
        )
        
        return cur_kmeans_centers, cluster_ids, residual_data
    
    def _reassign_clusters_middle_layer_with_residuals(self, X: torch.Tensor, kmeans_centers: torch.Tensor,
                                                       prev_cluster_ids: torch.Tensor, layer: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        重新分配中间层的聚类ID并计算残差，严格按照extreme版本的逻辑
        
        关键：
        1. 使用未取余的cluster_ids计算残差
        2. 返回未取余的cluster_ids和残差数据
        
        Args:
            X: 输入数据张量
            kmeans_centers: 合并的聚类中心 [pre_need_cluster * cur_need_cluster, dim]
            prev_cluster_ids: 前一层的聚类ID
            layer: 层索引
        
        Returns:
            (cluster_ids_raw, residual_data): 未取余的聚类ID和残差数据
        """
        pre_need_cluster = self.config.need_clusters[layer - 1]
        cur_need_cluster = self.config.need_clusters[layer]
        
        # 确保prev_cluster_ids为long类型
        if prev_cluster_ids.dtype != torch.long:
            prev_cluster_ids = prev_cluster_ids.long()
        
        # 中间层的聚类中心数量 = pre_need_cluster × cur_need_cluster
        num_centers = len(kmeans_centers)
        safe_batch = self._calculate_safe_batch_size(X, num_centers, self.device)
        logger.info(f"    - Safe batch_size: {safe_batch:,} (num_centers: {num_centers:,})")
        
        cluster_ids = []
        for i in tqdm(range(0, len(X), safe_batch), total=(len(X) + safe_batch - 1) // safe_batch, 
                      desc="    Reassigning", leave=False):
            batch = X[i:i+safe_batch].float().to(self.device)
            batch_cluster_ids = prev_cluster_ids[i:i+safe_batch]
            
            # 计算距离
            distance = pairwise_distance_full(batch, kmeans_centers, device=self.device)
            
            # 创建掩码
            mask = torch.zeros_like(distance)
            for cluster_idx in range(pre_need_cluster):
                cluster_mask = (batch_cluster_ids == cluster_idx)
                if cluster_mask.any():
                    start_idx = cluster_idx * cur_need_cluster
                    end_idx = start_idx + cur_need_cluster
                    mask[cluster_mask, start_idx:end_idx] = 1.0
            
            # 原地操作
            distance.add_(10000.0 * (1 - mask))
            
            cluster_assignments = torch.argmin(distance, dim=1)
            cluster_ids.append(cluster_assignments.cpu())
            
            del batch, distance, mask
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        cluster_ids_raw = torch.cat(cluster_ids)
        
        # 计算残差（使用未取余的cluster_ids_raw）
        logger.info(f"    - Computing residuals with raw cluster IDs (range: 0-{len(kmeans_centers)-1})")
        residual_data = self._compute_residuals_with_centers(X, cluster_ids_raw, kmeans_centers)
        
        # 返回未取余的cluster_ids和残差
        return cluster_ids_raw, residual_data
    
    def _reassign_clusters_last_layer_with_residuals(self, X: torch.Tensor, kmeans_centers: torch.Tensor,
                                                     before_cluster_ids: np.ndarray, match_matrix: List,
                                                     layer: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        重新分配最后一层的聚类ID并计算残差，严格按照extreme版本的逻辑
        
        关键：
        1. 使用未映射的cluster_ids计算残差
        2. 返回未映射的cluster_ids和残差数据
        
        Args:
            X: 输入数据张量
            kmeans_centers: 合并的聚类中心 [2 * n_clusters, dim]
            before_cluster_ids: 前两层的组合聚类ID
            match_matrix: 匹配矩阵，指定哪些聚类中心可用
            layer: 层索引
        
        Returns:
            (cluster_ids_raw, residual_data): 未映射的聚类ID和残差数据
        """
        # 最后一层的聚类中心数量 = 2 × n_clusters
        num_centers = len(kmeans_centers)
        safe_batch = self._calculate_safe_batch_size(X, num_centers, self.device)
        logger.info(f"    - Safe batch_size: {safe_batch:,} (num_centers: {num_centers:,})")
        
        # 保持match_matrix在CPU
        match_matrix_np = np.array(match_matrix, dtype=np.float32)
        
        cluster_ids = []
        
        for i in tqdm(range(0, len(X), safe_batch), total=(len(X) + safe_batch - 1) // safe_batch,
                      desc="    Reassigning", leave=False):
            batch = X[i:i+safe_batch].float().to(self.device)
            batch_before_ids = before_cluster_ids[i:i+safe_batch]
            
            # 计算距离
            distance = pairwise_distance_full(batch, kmeans_centers, device=self.device)
            
            # 从CPU match_matrix中索引
            match_matrix_batch = torch.from_numpy(
                match_matrix_np[batch_before_ids]
            ).float().to(self.device)
            
            # 原地操作
            distance.add_(10000.0 * (1 - match_matrix_batch))
            
            cluster_assignments = torch.argmin(distance, dim=1)
            cluster_ids.append(cluster_assignments.cpu())
            
            del batch, distance, match_matrix_batch
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        cluster_ids_raw = torch.cat(cluster_ids)
        
        # 计算残差（使用未映射的cluster_ids_raw）
        logger.info(f"    - Computing residuals with raw cluster IDs (range: 0-{len(kmeans_centers)-1})")
        residual_data = self._compute_residuals_with_centers(X, cluster_ids_raw, kmeans_centers)
        
        # 返回未映射的cluster_ids和残差
        return cluster_ids_raw, residual_data
    
    def _assign_last_match_matrix(self, cur_kmeans_centers: torch.Tensor, cur_n_cluster: int,
                                 X: torch.Tensor, prev_prev_need_cluster: int,
                                 prev_need_cluster: int, prev_prev_cluster_ids: np.ndarray,
                                 prev_cluster_ids: np.ndarray, cur_need_cluster: int,
                                 cur_trunct_cluster: int, layer: int) -> List:
        """
        为最后一层构建match_matrix，指定每个前两层的分组可以使用哪些聚类中心
        
        Args:
            cur_kmeans_centers: 当前层的聚类中心
            cur_n_cluster: 当前层的聚类中心总数
            X: 输入数据
            prev_prev_need_cluster: 前前层的实际聚类数
            prev_need_cluster: 前一层的实际聚类数
            prev_prev_cluster_ids: 前前层的聚类ID
            prev_cluster_ids: 前一层的聚类ID
            cur_need_cluster: 当前层的实际聚类数
            cur_trunct_cluster: 当前层的截断聚类数
            layer: 层索引
        
        Returns:
            match_matrix: 二维列表，shape为(prev_prev_need_cluster*prev_need_cluster, cur_n_cluster)
        """
        cur_match_matrix = []
        cur_kmeans_centers_np = cur_kmeans_centers.cpu().numpy()
        
        for i in tqdm(range(prev_prev_need_cluster), desc="    Match matrix", leave=False):
            for j in range(prev_need_cluster):
                # 获取该分组的数据
                idx = np.where((prev_prev_cluster_ids == i) & (prev_cluster_ids == j))[0]
                
                if len(idx) == 0:
                    cur_match_matrix.append([0] * cur_n_cluster)
                    continue
                
                sub_data = X[idx]
                
                if len(idx) <= cur_need_cluster:
                    sub_centers = sub_data.cpu().numpy()
                elif len(idx) < cur_trunct_cluster:
                    random_idx = np.random.choice(len(idx), cur_need_cluster, replace=False)
                    sub_data = sub_data[random_idx]
                    sub_centers = sub_data.cpu().numpy()
                else:
                    sub_kmeans = KMeans(n_clusters=cur_need_cluster, device=self.device, balanced=True)
                    adaptive_iter = self._calculate_adaptive_iter_limit(
                        len(sub_data), cur_need_cluster, layer, base_iter_limit=20
                    )
                    _ = sub_kmeans.fit(X=sub_data.to(self.device), distance='euclidean',
                                      iter_limit=adaptive_iter, tqdm_flag=True, half=False, online=False)
                    sub_centers = sub_kmeans.cluster_centers.cpu().numpy()
                
                # 优化：使用向量化操作计算距离，避免嵌套循环
                match_matrix_row = [0] * cur_n_cluster
                exist_idx_set = set()
                
                # 批量计算距离：(len(sub_centers), cur_n_cluster)
                sub_centers_tensor = torch.from_numpy(sub_centers).float().to(self.device)
                cur_centers_tensor = cur_kmeans_centers.float()
                distances = torch.cdist(sub_centers_tensor, cur_centers_tensor)
                
                for j_idx in range(min(len(sub_centers), cur_need_cluster)):
                    # 找到最近的未使用中心
                    dist_row = distances[j_idx].clone()
                    dist_row[list(exist_idx_set)] = float('inf')
                    min_idx = torch.argmin(dist_row).item()
                    
                    match_matrix_row[min_idx] = 1
                    exist_idx_set.add(min_idx)
                
                del sub_centers_tensor, cur_centers_tensor, distances
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                # 如果数量不够，随机选择
                if len(exist_idx_set) < cur_need_cluster:
                    for _ in range(cur_need_cluster - len(exist_idx_set)):
                        min_idx = np.random.randint(cur_n_cluster)
                        while min_idx in exist_idx_set:
                            min_idx = np.random.randint(cur_n_cluster)
                        match_matrix_row[min_idx] = 1
                        exist_idx_set.add(min_idx)
                
                cur_match_matrix.append(match_matrix_row)
        
        return cur_match_matrix
    
    def _merge_match_matrix_cluster_ids(self, match_matrix: List, cluster_ids: torch.Tensor,
                                       before_cluster_ids: np.ndarray) -> torch.Tensor:
        """
        通过match_matrix将聚类ID从全局空间映射到实际聚类空间
        
        Args:
            match_matrix: 匹配矩阵
            cluster_ids: 全局聚类ID
            before_cluster_ids: 前两层的组合聚类ID
        
        Returns:
            映射后的聚类ID
        """
        # 构建映射表
        mapping_result = {}
        for row in range(len(match_matrix)):
            res = {}
            cnt = 0
            for col in range(len(match_matrix[row])):
                if match_matrix[row][col] == 1:
                    res[col] = cnt
                    cnt += 1
            mapping_result[row] = res
        
        # 应用映射
        result_cluster_ids = []
        for i in range(len(cluster_ids)):
            prev_id = before_cluster_ids[i]
            cur_id = cluster_ids[i].item()
            result_cluster_ids.append(mapping_result[prev_id][cur_id])
        
        return torch.tensor(result_cluster_ids, device=self.device)
    
    def _compute_residuals_with_centers(self, X: torch.Tensor, cluster_ids: torch.Tensor,
                                        cluster_centers: torch.Tensor) -> torch.Tensor:
        """
        使用给定的聚类中心计算残差数据，残差 = 原始数据 - 聚类中心，然后按维度分组归一化
        
        这个函数用于在训练时计算残差，直接使用传入的聚类中心，避免从列表中索引
        
        Args:
            X: 输入数据张量
            cluster_ids: 当前层的聚类ID
            cluster_centers: 聚类中心张量
        
        Returns:
            归一化后的残差数据
        """
        # 确保cluster_ids为long类型
        if cluster_ids.dtype != torch.long:
            cluster_ids = cluster_ids.long()
        
        # 确保在同一设备上
        cluster_centers = cluster_centers.to(X.device)
        
        # 获取每个样本对应的聚类中心
        assigned_centers = cluster_centers[cluster_ids]
        
        # 计算残差
        residuals = X - assigned_centers
        
        # 按维度分组归一化（原地操作）
        cur_idx = 0
        for dim in self.config.group_dims:
            group_residuals = residuals[:, cur_idx:cur_idx+dim]
            norms = torch.norm(group_residuals, dim=1, keepdim=True) + 1e-8
            residuals[:, cur_idx:cur_idx+dim].div_(norms)
            cur_idx += dim
        
        del assigned_centers
        if X.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return residuals
    
    def _compute_residuals(self, X: torch.Tensor, cluster_ids: torch.Tensor,
                          layer: int) -> torch.Tensor:
        """
        计算残差数据用于下一层训练（用于预测阶段）
        
        Args:
            X: 输入数据张量
            cluster_ids: 当前层的聚类ID
            layer: 层索引
        
        Returns:
            归一化后的残差数据
        """
        kmeans_centers = self.cluster_centers_list[layer]
        return self._compute_residuals_with_centers(X, cluster_ids, kmeans_centers)
    
    def _predict_layer_0(self, X: torch.Tensor, layer: int) -> torch.Tensor:
        """
        预测第1层：直接计算到聚类中心的距离并分配
        
        Args:
            X: 输入数据张量
            layer: 层索引
        
        Returns:
            聚类ID
        """
        kmeans_centers = self.cluster_centers_list[layer]
        num_centers = len(kmeans_centers)
        safe_batch = self._calculate_safe_batch_size(X, num_centers, self.device)
        
        cluster_ids = []
        for i in tqdm(range(0, len(X), safe_batch), total=(len(X) + safe_batch - 1) // safe_batch,
                      desc="    Predicting", leave=False):
            batch = X[i:i+safe_batch]
            distance = pairwise_distance_full(batch, kmeans_centers, device=self.device)
            cluster_assignments = torch.argmin(distance, dim=1)
            cluster_ids.append(cluster_assignments.cpu())
            
            del batch, distance
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        return torch.cat(cluster_ids)
    
    def _predict_middle_layer(self, X: torch.Tensor, layer: int, all_cluster_ids: List[torch.Tensor]) -> torch.Tensor:
        """
        预测中间层：在前一层对应簇内分配聚类
        
        Args:
            X: 输入数据张量
            layer: 层索引
            all_cluster_ids: 之前层的预测聚类ID列表
        
        Returns:
            聚类ID
        """
        kmeans_centers = self.cluster_centers_list[layer]
        prev_cluster_ids = all_cluster_ids[layer - 1]
        pre_need_cluster = self.config.need_clusters[layer - 1]
        cur_need_cluster = self.config.need_clusters[layer]
        
        # 确保prev_cluster_ids为long类型
        if prev_cluster_ids.dtype != torch.long:
            prev_cluster_ids = prev_cluster_ids.long()
        
        # 动态计算安全batch_size
        num_centers = len(kmeans_centers)
        safe_batch = self._calculate_safe_batch_size(X, num_centers, self.device)
        
        cluster_ids = []
        for i in tqdm(range(0, len(X), safe_batch), total=(len(X) + safe_batch - 1) // safe_batch,
                      desc="    Predicting", leave=False):
            batch = X[i:i+safe_batch]
            batch_cluster_ids = prev_cluster_ids[i:i+safe_batch]
            
            # 计算距离
            distance = pairwise_distance_full(batch, kmeans_centers, device=self.device)
            
            # 优化：不创建完整的one-hot矩阵，直接创建掩码
            mask = torch.zeros_like(distance)
            for cluster_idx in range(pre_need_cluster):
                cluster_mask = (batch_cluster_ids == cluster_idx)
                if cluster_mask.any():
                    start_idx = cluster_idx * cur_need_cluster
                    end_idx = start_idx + cur_need_cluster
                    mask[cluster_mask, start_idx:end_idx] = 1.0
            
            # 原地操作
            distance.add_(10000.0 * (1 - mask))
            
            cluster_assignments = torch.argmin(distance, dim=1)
            cluster_ids.append(cluster_assignments.cpu())
            
            del batch, distance, mask
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        cluster_ids = torch.cat(cluster_ids)
        
        # 通过模运算将ID重新映射到[0, need_clusters[layer]-1]
        cluster_ids = cluster_ids % cur_need_cluster
        
        return cluster_ids
    
    def _predict_last_layer(self, X: torch.Tensor, layer: int, all_cluster_ids: List[torch.Tensor]) -> torch.Tensor:
        """
        预测最后一层：使用match_matrix约束进行聚类分配
        
        Args:
            X: 输入数据张量
            layer: 层索引
            all_cluster_ids: 之前层的预测聚类ID列表
        
        Returns:
            聚类ID
        """
        cur_kmeans_centers = self.cluster_centers_list[layer]
        match_matrix = self.match_matrices[layer - 1] if layer - 1 < len(self.match_matrices) else []
        
        # 使用当前预测数据的聚类ID，转换为numpy用于索引计算
        prev_prev_cluster_ids = all_cluster_ids[-2].cpu().numpy()
        prev_cluster_ids = all_cluster_ids[-1].cpu().numpy()
        prev_prev_need_cluster = self.config.need_clusters[layer - 2]
        
        # before_cluster_ids = prev_prev_cluster_ids * prev_prev_need_cluster + prev_cluster_ids
        before_cluster_ids = prev_prev_cluster_ids * prev_prev_need_cluster + prev_cluster_ids
        
        # 动态计算安全batch_size
        num_centers = len(cur_kmeans_centers)
        safe_batch = self._calculate_safe_batch_size(X, num_centers, self.device)
        
        # 优化：保持match_matrix在CPU，避免占用显存
        match_matrix_np = None
        if match_matrix:
            match_matrix_np = np.array(match_matrix, dtype=np.float32)
        
        cluster_ids = []
        for i in tqdm(range(0, len(X), safe_batch), total=(len(X) + safe_batch - 1) // safe_batch,
                      desc="    Predicting", leave=False):
            batch = X[i:i+safe_batch]
            batch_before_ids = before_cluster_ids[i:i+safe_batch]
            
            # 计算距离
            distance = pairwise_distance_full(batch, cur_kmeans_centers, device=self.device)
            
            # 基于before_cluster_ids，获取match_matrix
            if match_matrix_np is not None:
                # 优化：从CPU match_matrix中索引，然后转移到GPU
                match_matrix_batch = torch.from_numpy(
                    match_matrix_np[batch_before_ids]
                ).float().to(self.device)
                
                # 原地操作
                distance.add_(10000.0 * (1 - match_matrix_batch))
                
                del match_matrix_batch
            
            cluster_assignments = torch.argmin(distance, dim=1)
            cluster_ids.append(cluster_assignments.cpu())
            
            del batch, distance
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        cluster_ids = torch.cat(cluster_ids)
        
        # 通过match_matrix映射到最终的聚类ID
        if match_matrix:
            final_cluster_ids = self._merge_match_matrix_cluster_ids(
                match_matrix, cluster_ids, before_cluster_ids
            )
        else:
            final_cluster_ids = cluster_ids
        
        return final_cluster_ids
    
    def _load_previous_checkpoints(self, start_layer: int):
        """
        加载之前的检查点，用于恢复训练
        
        Args:
            start_layer: 起始层索引
        
        Raises:
            RuntimeError: 如果检查点数据不完整
        """
        for layer in range(start_layer):
            checkpoint = self.checkpoint_manager.load_layer_checkpoint(layer, self.device)
            if not checkpoint:
                raise RuntimeError(
                    f"Incomplete checkpoint data at layer {layer}. "
                    f"Use --clear-checkpoints flag to start training from scratch."
                )
            
            required_keys = ['cluster_ids', 'cluster_centers']
            missing_keys = [k for k in required_keys if k not in checkpoint or checkpoint[k] is None]
            if missing_keys:
                raise RuntimeError(
                    f"Incomplete checkpoint data at layer {layer}. Missing: {missing_keys}. "
                    f"Use --clear-checkpoints flag to start training from scratch."
                )
            
            self.cluster_centers_list.append(checkpoint['cluster_centers'])
            self.result_cluster_ids.append(checkpoint['cluster_ids'])
            
            if checkpoint.get('match_matrix'):
                self.match_matrices.append(checkpoint['match_matrix'])

    
    def save_model(self, model_dir: str):
        """保存训练好的模型"""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        config_file = model_dir / "config.json"
        with open(config_file, 'w') as f:
            config_dict = asdict(self.config)
            json.dump(config_dict, f, indent=2)
        
        # 保存聚类中心
        centers_file = model_dir / "cluster_centers.pkl"
        with open(centers_file, 'wb') as f:
            centers_data = [c.cpu().numpy() for c in self.cluster_centers_list]
            pickle.dump(centers_data, f)
        
        # 保存匹配矩阵
        if self.match_matrices:
            matrix_file = model_dir / "match_matrices.pkl"
            with open(matrix_file, 'wb') as f:
                pickle.dump(self.match_matrices, f)
    
    def load_model(self, model_dir: str):
        """加载训练好的模型"""
        model_dir = Path(model_dir)
        
        # 加载配置
        config_file = model_dir / "config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
            for key, value in config_dict.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        # 加载聚类中心
        centers_file = model_dir / "cluster_centers.pkl"
        if centers_file.exists():
            with open(centers_file, 'rb') as f:
                centers_list = pickle.load(f)
            self.cluster_centers_list = [
                torch.from_numpy(c).to(self.device) for c in centers_list
            ]
        
        # 加载匹配矩阵
        matrix_file = model_dir / "match_matrices.pkl"
        if matrix_file.exists():
            with open(matrix_file, 'rb') as f:
                self.match_matrices = pickle.load(f)
        
        self.is_trained = len(self.cluster_centers_list) > 0
    
    def get_training_status(self) -> Dict:
        """获取训练状态"""
        if self.checkpoint_manager:
            last_layer = self.checkpoint_manager.get_last_completed_layer()
            return {
                'is_trained': self.is_trained,
                'last_completed_layer': last_layer,
                'total_layers': len(self.config.layer_clusters),
                'can_resume': last_layer >= 0,
            }
        else:
            return {
                'is_trained': self.is_trained,
                'last_completed_layer': -1,
                'total_layers': len(self.config.layer_clusters),
                'can_resume': False,
            }


# 为了兼容性，保留原始的参数类
class hierarchicalRqClusterParams:
    """兼容原始代码的参数类"""
    
    def __init__(self, 
                 layer_clusters: List[int] = None,
                 need_clusters: List[int] = None,
                 embedding_dim: int = 1024,
                 group_dims: Union[int, List[int]] = None,
                 hierarchical_weights: Union[float, List[List[float]]] = None):
        
        if layer_clusters is None:
            layer_clusters = [128, 256, 256]
        if need_clusters is None:
            need_clusters = [128, 128, 128]
        if group_dims is None:
            group_dims = embedding_dim
        if hierarchical_weights is None:
            hierarchical_weights = 1.0
        
        self.config = HierarchicalRQKMeansConfig(
            layer_clusters=layer_clusters,
            need_clusters=need_clusters,
            embedding_dim=embedding_dim,
            group_dims=group_dims,
            hierarchical_weights=hierarchical_weights,
        )
        
        # 为了兼容性，暴露属性
        self.layer_clusters = self.config.layer_clusters
        self.need_clusters = self.config.need_clusters
        self.embedding_dim = self.config.embedding_dim
        self.group_dims = self.config.group_dims
        self.hierarchical_weights = self.config.hierarchical_weights