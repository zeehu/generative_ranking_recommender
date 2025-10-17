Step G1: Generate Song Semantic IDs using RQ-KMeans.

This script reads the song vectors created in Step G0 and uses Faiss for
efficient K-Means clustering to perform Residual Quantization.

import os
import sys
import numpy as np
import json
import logging
from tqdm import tqdm
import pandas as pd

try:
    import faiss
except ImportError:
    print("Faiss library not found. Please install it first.")
    print("CPU version: pip install faiss-cpu")
    print("GPU version: pip install faiss-gpu")
    sys.exit(1)

# Add project root to sys.path to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.sys.path.insert(0, project_root)

from config import Config
from src.common.utils import setup_logging

logger = logging.getLogger(__name__)

class KMeansTrainer:
    """Orchestrates the RQ-KMeans training and semantic ID generation."""

    def __init__(self, config: Config):
        self.config = config
        self.kmeans_config = config.rqkmeans
        self.data_config = config.data
        # Check for torch and CUDA for GPU usage
        try:
            import torch
            self.use_gpu = torch.cuda.is_available() and hasattr(faiss, 'StandardGpuResources')
        except ImportError:
            self.use_gpu = False

        if self.use_gpu:
            logger.info("Faiss GPU support detected. Using GPU for K-Means.")
        else:
            logger.info("Using CPU for K-Means. This might be slow for large datasets.")

    def run(self):
        logger.info("--- Starting Step G1: RQ-KMeans Training ---")

        song_ids, vectors = self._load_data()
        indices_per_level, codebooks = self._train_rq_kmeans(vectors)
        self._save_results(song_ids, indices_per_level, codebooks)

        logger.info("--- Step G1 Completed Successfully ---")

    def _load_data(self) -> tuple[list, np.ndarray]:
        vector_file = self.data_config.song_vectors_file
        logger.info(f"Loading song vectors from {vector_file}...")
        try:
            df = pd.read_csv(vector_file, dtype={'mixsongid': str})
            df.set_index('mixsongid', inplace=True)
            song_ids = df.index.tolist()
            vectors = df.to_numpy(dtype='float32')
            logger.info(f"Successfully loaded {len(song_ids)} song vectors.")
            return song_ids, vectors
        except FileNotFoundError:
            logger.error(f"FATAL: Song vector file not found at {vector_file}")
            logger.error("Please run Step G0 (train_word2vec.py) first.")
            sys.exit(1)

    def _train_rq_kmeans(self, vectors: np.ndarray) -> tuple[np.ndarray, list]:
        logger.info("Starting residual quantization training with K-Means...")
        
        residuals = vectors.copy()
        all_indices = []
        all_codebooks = []
        
        for level in range(self.kmeans_config.levels):
            logger.info(f"--- Training Level {level + 1}/{self.kmeans_config.levels} ---")
            
            d = self.kmeans_config.input_dim
            k = self.kmeans_config.vocab_size
            seed = self.kmeans_config.seed + level

            kmeans = faiss.Kmeans(d, k, niter=20, verbose=True, seed=seed, gpu=self.use_gpu)
            kmeans.train(residuals.astype('float32')) # Faiss requires float32
            all_codebooks.append(kmeans.centroids)

            _D, I = kmeans.index.search(residuals.astype('float32'), 1)
            all_indices.append(I.flatten())

            assigned_centroids = kmeans.centroids[I.flatten()]
            residuals = residuals - assigned_centroids

        final_indices = np.stack(all_indices, axis=1)
        return final_indices, all_codebooks

    def _save_results(self, song_ids: list, indices: np.ndarray, codebooks: list):
        logger.info("Saving semantic IDs and codebooks...")

        output_path = self.data_config.semantic_ids_file
        with open(output_path, 'w') as f:
            for i, song_id in enumerate(tqdm(song_ids, desc="Saving Semantic IDs")):
                item = {
                    'song_id': str(song_id),
                    'semantic_ids': indices[i].tolist()
                }
                f.write(json.dumps(item) + '\n')
        logger.info(f"Semantic IDs saved to {output_path}")

        codebooks_path = os.path.join(self.config.model_dir, "generator", "rq_kmeans_codebooks.npy")
        np.save(codebooks_path, np.array(codebooks))
        logger.info(f"Codebooks saved to {codebooks_path}")

if __name__ == "__main__":
    config = Config()
    log_file_path = os.path.join(config.log_dir, "g1_train_rq_kmeans.log")
    setup_logging(log_file=log_file_path)
    logger = logging.getLogger(__name__)

    if config.data.song_vectors_file == "outputs/song_vectors.csv" and not os.path.exists(config.data.song_vectors_file):
         logger.error("="*80)
         logger.error(f"FATAL: Default song vector file '{config.data.song_vectors_file}' not found.")
         logger.error("Please run Step G0 (src/common/train_word2vec.py) first.")
         logger.error("="*80)
         sys.exit(1)

    trainer = KMeansTrainer(config)
    trainer.run()