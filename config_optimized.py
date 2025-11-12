"""
优化后的配置文件 - 用于高性能训练     
针对 3×L20 GPU + 20 CPU + 50GB内存 + 400万训练数据优化

硬件规格:
- GPU: 3 × NVIDIA L20 (48GB GDDR6, 864GB/s带宽, 11776 CUDA核心)
- CPU: 20核心
- 内存: 50GB
- 训练数据: 400万条

优化策略:
1. Batch Size优化: 针对3卡优化per_device_batch和梯度累积，最大化GPU利用率
2. 学习率调整: 根据有效batch size线性缩放
3. DataLoader优化: 充分利用CPU核心，启用预取和pin_memory
4. 训练轮数优化: 400万数据3轮足够，减少不必要的训练时间
5. 评估策略优化: 合理的评估频率，避免过度评估

预期效果:
- 训练时长: ~10-11小时 (3 epochs)
- GPU利用率: >85%
- 显存占用: 38-42GB/48GB (安全范围)
- 每步耗时: ~3.5秒
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
    need_clusters=[128, 128, 256],
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
class TrainingConfig:
    """训练过程控制配置"""
    # 评估和保存策略
    # 根据实际计算 (3卡配置):
    #   - 训练样本: 3,938,699
    #   - 有效batch: 128×3×5 = 1920
    #   - 每轮步数: 3,938,699 / 1920 ≈ 2,051步
    #   - 总步数: 2,051 × 3 = 6,153步
    # 
    # 优化策略:
    #   - eval_steps=500 → 每轮评估4次，总计约12次
    #   - save_steps=500 → 与评估同步
    #   - 评估占用时间: ~5分钟/次 × 12 = 60分钟 (可接受)
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # 日志配置
    logging_steps: int = 100
    logging_first_step: bool = True
    report_to: str = "none"
    
    # 学习率调度器
    lr_scheduler_type: str = "cosine"  # cosine | linear | polynomial
    
    # DataLoader配置
    dataloader_pin_memory: bool = True
    dataloader_prefetch_factor: int = 4
    dataloader_persistent_workers: bool = True
    
    # DDP配置
    ddp_find_unused_parameters: bool = False
    ddp_bucket_cap_mb: int = 50
    ddp_broadcast_buffers: bool = False
    
    # 其他
    remove_unused_columns: bool = False
    ignore_data_skip: bool = False
    disable_tqdm: bool = False

@dataclass
class PlaylistTIGERConfig:
    """Configuration for the T5 Generator model (OPTIMIZED)."""
    model_name: str = "/home/search/base-model/mengzi-t5-base" # Local path for offline env
    max_input_length: int = 128
    max_target_length: int = 384
    
    # ========== 训练轮数优化 ==========
    # 400万数据，3轮足够收敛，减少训练时间
    num_train_epochs: int = 3  # 5 → 3
    
    # ========== 3卡优化的Batch Size配置 ==========
    # L20规格: 48GB显存, 11776 CUDA核心, 864GB/s带宽
    # 
    # 3卡优化策略:
    #   目标: 保持有效batch size接近2048，同时最大化GPU利用率
    #   
    #   方案对比:
    #   1. 128×3×5=1920 (推荐) → 显存~38GB (79%), 更新频率适中
    #   2. 128×3×6=2304        → 显存~42GB (88%), 更新频率较慢
    #   3. 160×3×4=1920        → 显存~42GB (88%), batch过大可能影响收敛
    # 
    # 最优配置 (方案1):
    #   - per_device_batch=128 → 显存~38GB (79%利用率，安全)
    #   - gradient_accum=5 → 平衡更新频率和通信开销
    #   - 有效batch=128×3×5=1920 (接近目标2048)
    # 
    # 预期效果:
    #   1. 显存利用率: 79% (充分利用，留有余量)
    #   2. GPU利用率: 85-90% (接近最优)
    #   3. 总训练时长: ~10-11小时 (3 epochs)
    #   4. 梯度更新频率: 每5步同步，平衡稳定性和速度
    #   5. 通信开销: 适中 (3卡DDP通信效率高)
    per_device_train_batch_size: int = 128  # 针对3卡优化
    per_device_eval_batch_size: int = 160   # 评估时可以更大
    gradient_accumulation_steps: int = 5    # 4 → 5 (针对3卡调整)
    # 有效batch size = 128 × 3 GPUs × 5 = 1920
    
    # ========== 学习率配置 (根据3卡batch size缩放) ==========
    # 线性缩放规则: lr_new = lr_base × sqrt(batch_new / batch_base)
    # 基准: batch=640, lr=2e-4
    # 新配置: batch=1920, lr = 2e-4 × sqrt(1920/640) ≈ 3.46e-4
    # 使用 4.0e-4 以加快收敛 (略高于理论值，经验调优)
    learning_rate: float = 4.0e-4  # 根据batch=1920调整
    
    # Warmup优化: 适中的warmup步数，平衡稳定性和速度
    # 总步数约6153步，warmup 10%约615步
    warmup_steps: int = 615      # 约占总步数10%
    warmup_ratio: float = 0.10   # 占总步数的10%
    
    weight_decay: float = 0.01
    # ========== 混合精度训练 ==========
    # L20 Tensor Core性能: FP16是FP32的2倍 (181 vs 90.5 TFLOPS)
    fp16: bool = True
    fp16_opt_level: str = "O2"  # 更激进的混合精度优化
    fp16_backend: str = "auto"  # 自动选择最优后端
    
    # ========== 梯度优化 ==========
    max_grad_norm: float = 1.0  # 梯度裁剪，防止梯度爆炸
    gradient_checkpointing: bool = True  # 必须开启，节省30-40%显存
    gradient_checkpointing_kwargs: dict = field(default_factory=lambda: {"use_reentrant": False})
    
    # ========== 编译优化 (可选) ==========
    # torch.compile可提速5-15%，但首次编译需要额外时间
    # 建议: 长期训练(>4小时)开启，短期训练关闭
    use_torch_compile: bool = True  # 设为True启用编译优化
    torch_compile_backend: str = "inductor"  # 编译后端
    torch_compile_mode: str = "max-autotune"  # 编译模式: reduce-overhead | max-autotune
    
    # Number of custom semantic ID tokens to add to the tokenizer
    num_semantic_id_tokens: int = 0 # This will be calculated dynamically

    def __post_init__(self):
        # Ensure num_semantic_id_tokens is set if h_rqkmeans is available
        # This requires access to the main Config object, which is not available here.
        # It will be set in train_t5.py based on h_rqkmeans.need_clusters.
        pass

@dataclass
class SystemConfig:
    """系统和硬件配置"""
    # 硬件信息（用于日志显示）
    expected_num_gpus: int = 3  # 4 → 3
    gpu_model: str = "L20"
    gpu_memory_gb: int = 48
    cpu_cores: int = 20
    system_memory_gb: int = 50
    
    # 性能估算参数 (基于3卡×batch=128配置)
    estimated_time_per_step: float = 3.5  # seconds (3卡略慢于4卡)
    expected_gpu_utilization: str = "85-90%"
    expected_speedup: str = "3卡优化配置"

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
    training: TrainingConfig = field(default_factory=TrainingConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    # Common paths
    output_dir: str = "outputs"
    model_dir: str = "models"
    log_dir: str = "logs"
    
    # ========== 优化后的系统设置 ==========
    device: str = "cuda"
    seed: int = 42
    
    # DataLoader workers优化 (充分利用20核CPU)
    # 分配策略: 15核给DataLoader, 2核给系统, 3核给DDP通信
    # 每个GPU: 15/3 = 5个workers (充分利用CPU)
    num_workers: int = 15  # 针对3卡优化
    
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
