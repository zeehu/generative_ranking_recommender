"""
Configuration file for the Generative Ranking Recommender project.
"""
import os
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class DataConfig:
    """Configuration for data paths and processing."""
    # --- Paths to be filled by user ---
    playlist_songs_file: str = "data/gen_playlist_song.csv.sort"
    playlist_info_file: str = "data/gen_playlist_info.csv"
    song_info_file: str = "data/gen_song_info.csv"
    
    # --- Paths for generated files (outputs of steps) ---
    song_vectors_file: str = "outputs/song_vectors.csv"
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
    vector_size: int = 256      # Dimensionality of the song vectors.
    window: int = 100           # Context window size.
    min_count: int = 5          # Ignores all songs with total frequency lower than this.
    workers: int = -1           # Use all available CPU cores, -1 means all.
    epochs: int = 10             # Increased epochs for better quality.
    sample: float = 1e-5        # More aggressive subsampling for frequent words.

@dataclass
class HierarchicalRQKMeansConfig:
    """Configuration for the Hierarchical RQ-KMeans semantic ID generator."""
    layer_clusters: List[int] = field(default_factory=list)
    need_clusters: List[int] = field(default_factory=list)
    embedding_dim: int = 256
    iter_limit: int = 100

# Pre-configured settings for different dataset sizes
H_RQ_KMEANS_PROD = HierarchicalRQKMeansConfig(
    layer_clusters=[128, 1280, 2560],
    need_clusters=[128, 128, 256],
    embedding_dim=256,
    iter_limit=100
)

H_RQ_KMEANS_TEST = HierarchicalRQKMeansConfig(
    layer_clusters=[32, 32, 64],
    need_clusters=[32, 32, 8],
    embedding_dim=256,
    iter_limit=50
)

@dataclass
class PlaylistTIGERConfig:
    """Configuration for the T5 Generator model."""
    model_name: str = "/home/search/base_model/menzi-t5-base" # Local path for offline env
    max_input_length: int = 128
    max_target_length: int = 256
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 192
    per_device_eval_batch_size: int = 8
    learning_rate: float = 5e-4
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    gradient_checkpointing_kwargs: dict = field(default_factory=lambda: {"use_reentrant": False})
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
    """Main configuration for the project."""
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
    
    # System settings
    device: str = "cuda"
    seed: int = 42
    num_workers: int = 2
    
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
