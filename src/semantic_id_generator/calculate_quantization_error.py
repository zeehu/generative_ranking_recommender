"""
This script calculates the global quantization error of the semantic IDs.

It measures the average distance between the original song vectors and their
reconstructed vectors from the semantic ID centroids. A lower error indicates
that the semantic IDs retain more information from the original vector space.
"""
import os
import sys
import json
import numpy as np
import torch
from tqdm import tqdm
import pickle

# Add project root for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config
from src.semantic_id_generator.hierarchical_rq_kmeans import HierarchicalRQKMeans
from src.common.utils import load_song_vectors

def calculate_quantization_error(config: Config):
    """
    Calculates and reports the quantization error (MSE).
    """
    print("--- Starting Quantization Error Calculation ---")

    # 1. Load the trained HierarchicalRQKMeans model
    model_path = os.path.join(config.model_dir, "semantic_id", "h_rq_kmeans.pkl")
    if not os.path.exists(model_path):
        print(f"FATAL: Trained model not found at {model_path}")
        print("Please run simplified_semantic_id_generator.py to train the model first.")
        return

    print(f"Loading trained HierarchicalRQKMeans model from {model_path}...")
    with open(model_path, 'rb') as f:
        h_rq_kmeans: HierarchicalRQKMeans = pickle.load(f)
    
    # 2. Load original song vectors
    print("Loading original song vectors...")
    song_vectors = load_song_vectors(config.data.song_vectors_file)
    song_ids_with_vectors = list(song_vectors.keys())
    
    if not song_vectors:
        print("FATAL: No song vectors loaded. Check the path in config.py.")
        return

    # 3. Load semantic ID mappings
    semantic_ids_file = config.data.semantic_ids_file
    if not os.path.exists(semantic_ids_file):
        print(f"FATAL: Semantic ID file not found at {semantic_ids_file}")
        return

    print(f"Loading semantic IDs from {semantic_ids_file}...")
    song_to_semantic_id = {}
    with open(semantic_ids_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            song_to_semantic_id[item['song_id']] = tuple(item['semantic_ids'])

    # 4. Calculate error
    squared_errors = []
    
    print("Calculating reconstruction errors for each song...")
    for song_id in tqdm(song_ids_with_vectors, desc="Processing songs"):
        if song_id not in song_to_semantic_id:
            continue

        original_vector = song_vectors[song_id]
        semantic_id = song_to_semantic_id[song_id]

        # Reconstruct the vector from the ID
        reconstructed_vector = h_rq_kmeans.reconstruct_vector_from_id(semantic_id)

        if reconstructed_vector is None:
            continue
            
        # Calculate squared Euclidean distance
        error = np.sum((original_vector - reconstructed_vector.numpy()) ** 2)
        squared_errors.append(error)

    if not squared_errors:
        print("\nERROR: Could not calculate errors. No overlapping songs found between vectors and semantic IDs.")
        return

    # 5. Report results
    mse = np.mean(squared_errors)
    rmse = np.sqrt(mse)

    print("\n" + "="*80)
    print("  Quantization Error Report")
    print("="*80)
    print(f"  Total songs evaluated: {len(squared_errors)}")
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    print("\n" + "="*80)
    print("\nInterpretation:")
    print("  - MSE/RMSE measures the average squared distance between a song's original vector and")
    print("    its representation by the semantic ID's cluster center.")
    print("  - A lower value is better, indicating less information loss.")
    print("  - If this value is high, consider increasing the number of clusters in `config.py`")
    print("    (H_RQ_KMEANS_PROD.need_clusters) and retraining the semantic ID model.")


if __name__ == "__main__":
    main_config = Config()
    # A utility function needs to be added to utils.py to load vectors properly.
    # For now, let's assume it exists and is correct.
    calculate_quantization_error(main_config)
