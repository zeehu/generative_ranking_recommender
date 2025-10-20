"""
Configuration file for the Generative Ranking Recommender project.
"""
import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class DataConfig:
    """Configuration for data paths and processing."""
    # --- Paths to be filled by user ---
    playlist_songs_file: str = "path/to/your/gen_playlist_song.csv"
    playlist_info_file: str = "path/to/your/gen_playlist_info.csv"
    song_info_file: str = "path/to/your/gen_song_info.csv"
    
    # --- Paths for generated files (outputs of steps) ---
    song_vectors_file: str = "outputs/song_vectors.csv"
    semantic_ids_file: str = "outputs/generator/song_semantic_ids.jsonl"

@dataclass
class Word2VecConfig:
    """Configuration for Word2Vec training."""
    vector_size: int = 256      # Dimensionality of the song vectors.
    window: int = 100          # Increased context window size.
    min_count: int = 5          # Ignores all songs with total frequency lower than this.
    sample: float = 1e-4        # Subsampling threshold for frequent words.
    workers: int = -1           # Use all available CPU cores, -1 means all.
    epochs: int = 10            # Number of iterations over the corpus.

@dataclass
class SongRQKMeansConfig:
    """Configuration for Song RQ-KMeans training."""
    input_dim: int = 256      # Should match Word2Vec vector_size
    vocab_size: int = 2048  # This is 'k' in k-means
    levels: int = 2         # Number of residual levels
    seed: int = 42

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

# Note: Config for Ranker model will be added later.

@dataclass
class Config:
    """Main configuration for the project."""
    data: DataConfig = field(default_factory=DataConfig)
    word2vec: Word2VecConfig = field(default_factory=Word2VecConfig)
    rqkmeans: SongRQKMeansConfig = field(default_factory=SongRQKMeansConfig)
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
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, "generator"), exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, "ranker"), exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
