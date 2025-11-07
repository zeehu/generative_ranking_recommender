"""
优化后的配置文件 - 用于高性能训练
基于原config.py，针对2×L20 GPU + 20 CPU + 50GB内存优化
"""
import os
from dataclasses import dataclass, field
from typing import List, Dict

# 导入HierarchicalRQKMeansConfig
from src.semantic_id_generator.hierarchical_rq_kmeans import HierarchicalRQKMeansConfig

@dataclass
class DataConfig:
    """Configuration for data paths and processing."""
    # --- Paths to be filled by user ---
    playlist_songs_file: str = "data/gen_playlist_song.csv.sort"
    playlist_info_file: str = "data/gen_playlist_info.csv"
    song_info_file: str = "data/gen_song_info.csv"
    
    # --- Paths for generated files (outputs of steps) ---
    song_vectors_file: str = "outputs/sg1_vs512_w50_ep20_song_vectors.csv"
    # Updated path for the new semantic ID generator
    semantic_ids_file: str = "outputs/semantic_id/song_semantic_ids.jsonl"

    # Data split ratios for training/validation/test
    train_split_ratio: float = 0.98
    val_split_ratio: float = 0.01
    min_songs_per_playlist: int = 10 # Minimum number of songs required in a playlist after filtering by semantic ID availability

@dataclass
class Word2VecConfig:
    """Configuration for Word2Vec training."""
    corpus_file: str = "outputs/playlists_corpus.txt" # Path to the preprocessed corpus
    corpus_ids_file: str = "outputs/playlists_corpus.ids.txt" # Path to the corresponding playlist IDs
    vector_size: int = 512      # Dimensionality of the song vectors.
    window: int = 50           # Context window size.
    min_count: int = 10          # Ignores all songs with total frequency lower than this.
    workers: int = 20           # Use all available CPU cores, -1 means all.
    epochs: int = 20             # Increased epochs for better quality.
    sample: float = 1e-5        # More aggressive subsampling for frequent words.

# Pre-configured settings for different dataset sizes
H_RQ_KMEANS_PROD = HierarchicalRQKMeansConfig(
    layer_clusters=[128, 1280, 1280],
    need_clusters=[128, 128, 128],
    embedding_dim=512,
    group_dims=[512],  # 如果没有特殊分组，使用单个维度
    hierarchical_weights=[[1.0], [1.0], [1.0]],  # 如果没有特殊权重，使用均匀权重
    iter_limit=100
)

H_RQ_KMEANS_TEST = HierarchicalRQKMeansConfig(
    layer_clusters=[32, 128, 128],  # 第2层改为128，使其不等于need_clusters[1]=32，触发递归聚类策略
    need_clusters=[32, 32, 32],
    embedding_dim=512,
    group_dims=[512],  # 如果没有特殊分组，使用单个维度
    hierarchical_weights=[[1.0], [1.0], [1.0]],  # 如果没有特殊权重，使用均匀权重
    iter_limit=50
)

@dataclass
class PlaylistTIGERConfig:
    """Configuration for the T5 Generator model (OPTIMIZED)."""
    model_name: str = "/home/search/base-model/mengzi-t5-base" # Local path for offline env
    max_input_length: int = 128
    max_target_length: int = 384
    num_train_epochs: int = 5
    
    # ========== 优化后的Batch Size配置 ==========
    # 原配置: batch_size=160, grad_accum=2, 有效batch=640
    # 新配置: batch_size=32, grad_accum=10, 有效batch=640 (保持不变)
    # 优势: 更小的batch size提高GPU利用率，减少显存压力
    per_device_train_batch_size: int = 32   # 160 → 32 (降低5倍)
    per_device_eval_batch_size: int = 64    # 256 → 64 (评估时可以更大)
    gradient_accumulation_steps: int = 10   # 2 → 10 (增加5倍)
    # 有效batch size = 32 × 2 GPUs × 10 = 640 (与原来相同)
    
    learning_rate: float = 5e-4
    warmup_steps: int = 500
    weight_decay: float = 0.01
    fp16: bool = True
    gradient_checkpointing_kwargs: dict = field(default_factory=lambda: {"use_reentrant": False})
    
    # ========== 新增优化选项 ==========
    use_torch_compile: bool = False  # 设为True可提速5-15%，但首次编译会慢
    
    # Number of custom semantic ID tokens to add to the tokenizer
    num_semantic_id_tokens: int = 0 # This will be calculated dynamically

    def __post_init__(self):
        # Ensure num_semantic_id_tokens is set if h_rqkmeans is available
        # This requires access to the main Config object, which is not available here.
        # It will be set in train_t5.py based on h_rqkmeans.need_clusters.
        pass

# Note: Config for Ranker model will be added later.

@dataclass
class Config:
    """Main configuration for the project (OPTIMIZED)."""
    data: DataConfig = field(default_factory=DataConfig)
    word2vec: Word2VecConfig = field(default_factory=Word2VecConfig)
    # Use the new hierarchical config
    h_rqkmeans: HierarchicalRQKMeansConfig = field(default_factory=lambda: H_RQ_KMEANS_PROD)
    h_rqkmeans_test: HierarchicalRQKMeansConfig = field(default_factory=lambda: H_RQ_KMEANS_TEST)
    generator_t5: PlaylistTIGERConfig = field(default_factory=PlaylistTIGERConfig)
    
    # Common paths
    output_dir: str = "outputs"
    model_dir: str = "models"
    log_dir: str = "logs"
    
    # ========== 优化后的系统设置 ==========
    device: str = "cuda"
    seed: int = 42
    num_workers: int = 8  # 4 → 8 (充分利用20个CPU)
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "generator"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "ranker"), exist_ok=True)
        # Add new output directory for semantic_id module
        os.makedirs(os.path.join(self.output_dir, "semantic_id"), exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, "generator"), exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, "ranker"), exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
