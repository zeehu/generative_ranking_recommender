"""
This script calculates the global quantization error of the semantic IDs.

It measures the average distance between the original song vectors and their
reconstructed vectors from the semantic ID centroids.

This version is adapted to use the saved components from `train_semantic_ids.py`
(specifically `cluster_centers.pkl`) instead of a full model object.

NOTE: It calculates the error based on the FIRST LAYER (L1) of centroids only.
This represents the "coarse" quantization error, which is the most significant
component of the total quantization error.
"""
import os
import sys
import json
import numpy as np
import pickle
from tqdm import tqdm

# Add project root for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config
from src.common.utils import load_song_vectors, setup_logging

logger = setup_logging()

def calculate_l1_quantization_error(config: Config):
    """
    Calculates and reports the L1 quantization error (MSE) using saved components.
    """
    print("--- Starting L1 Quantization Error Calculation ---")

    # 1. Load the list of cluster centers for all layers
    centers_path = os.path.join(config.model_dir, "semantic_id", "cluster_centers.pkl")
    if not os.path.exists(centers_path):
        logger.error(f"FATAL: Cluster centers file not found at {centers_path}")
        logger.error("Please run train_semantic_ids.py to generate the cluster centers first.")
        return

    logger.info(f"Loading cluster centers from {centers_path}...")
    try:
        with open(centers_path, 'rb') as f:
            all_centers = pickle.load(f)
        
        if not isinstance(all_centers, list) or len(all_centers) == 0:
            raise ValueError("Expected a non-empty list of cluster centers.")
        
        # We only need the first layer of centroids for this analysis
        l1_centers = all_centers[0]
        logger.info(f"Successfully loaded {len(all_centers)} layers of centroids. Using L1 with {len(l1_centers)} centers.")

    except Exception as e:
        logger.error(f"Failed to load or parse cluster centers file: {e}")
        return

    # 2. Load original song vectors
    logger.info("Loading original song vectors...")
    song_vectors = load_song_vectors(config.data.song_vectors_file)
    if not song_vectors:
        logger.error("FATAL: No song vectors loaded. Aborting.")
        return

    # 3. Load semantic ID mappings
    semantic_ids_file = config.data.semantic_ids_file
    if not os.path.exists(semantic_ids_file):
        logger.error(f"FATAL: Semantic ID file not found at {semantic_ids_file}. Aborting.")
        return

    logger.info(f"Loading semantic IDs from {semantic_ids_file}...")
    song_to_semantic_id = {}
    with open(semantic_ids_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            song_to_semantic_id[item['song_id']] = tuple(item['semantic_ids'])

    # 4. Calculate error
    squared_errors = []
    
    logger.info("Calculating L1 reconstruction errors for each song...")
    for song_id, original_vector in tqdm(song_vectors.items(), desc="Processing songs"):
        if song_id not in song_to_semantic_id:
            continue

        semantic_id = song_to_semantic_id[song_id]
        l1_id = semantic_id[0]

        # Reconstruct the vector from the L1 ID
        if l1_id >= len(l1_centers):
            logger.warning(f"Song ID {song_id} has an out-of-bounds L1 ID {l1_id}. Skipping.")
            continue
            
        reconstructed_vector = l1_centers[l1_id]
        
        # Calculate squared Euclidean distance
        error = np.sum((original_vector - reconstructed_vector) ** 2)
        squared_errors.append(error)

    if not squared_errors:
        logger.error("\nERROR: Could not calculate errors. No overlapping songs found between vectors and semantic IDs.")
        return

    # 5. Report results
    mse = np.mean(squared_errors)
    rmse = np.sqrt(mse)

    print("\n" + "="*80)
    print("  L1 Quantization Error Report")
    print("="*80)
    print(f"  Total songs evaluated: {len(squared_errors)}")
    print(f"  Mean Squared Error (MSE) on L1: {mse:.4f}")
    print(f"  Root Mean Squared Error (RMSE) on L1: {rmse:.4f}")
    print("\n" + "="*80)
    print("\nInterpretation:")
    print("  - This metric measures the average squared distance between a song's original vector")
    print("    and the center of the main cluster (L1) it belongs to.")
    print("  - This is a core measure of information loss during the first, most critical")
    print("    step of quantization.")
    print("  - A lower value is better. If this value is high, consider increasing the number of")
    print("    clusters in the first layer of `need_clusters` in `config.py` and retraining.")


if __name__ == "__main__":
    main_config = Config()
    calculate_l1_quantization_error(main_config)