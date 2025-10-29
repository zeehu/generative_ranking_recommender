import os
import sys
import json
import csv
import numpy as np
from tqdm import tqdm
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Add project root to sys.path to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config

def calculate_metrics_batch(config: Config, sample_size: int = 50000):
    print("\n" + "="*80)
    print("  Batch Calculation of Clustering Metrics (Silhouette, CH, DB)")
    print("="*80)

    # Load semantic IDs (this is smaller, so load fully)
    semantic_ids_map = {}
    try:
        with open(config.data.semantic_ids_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading semantic IDs"):
                item = json.loads(line)
                semantic_ids_map[item['song_id']] = tuple(item['semantic_ids'])
    except FileNotFoundError:
        print(f"ERROR: Semantic ID file not found at {config.data.semantic_ids_file}.")
        return

    # Get all song IDs that have semantic IDs
    all_song_ids_with_sem_ids = list(semantic_ids_map.keys())
    
    if not all_song_ids_with_sem_ids:
        print("No songs with semantic IDs found for metric calculation.")
        return

    # --- Unified Sampling for all metrics ---
    # Sample song IDs first, then load only their vectors.
    if len(all_song_ids_with_sem_ids) > sample_size:
        print(f"Sampling {sample_size} song IDs for metric calculations (full dataset is too large)...")
        indices = np.random.choice(len(all_song_ids_with_sem_ids), sample_size, replace=False)
        sampled_song_ids = [all_song_ids_with_sem_ids[i] for i in indices]
    else:
        sampled_song_ids = all_song_ids_with_sem_ids
    
    # Now, load only the vectors for the sampled song IDs
    X_sample_list = []
    labels_sample_list = []
    
    # Create a set for efficient lookup of sampled song IDs
    sampled_song_ids_set = set(sampled_song_ids)
    
    print(f"Loading vectors for {len(sampled_song_ids)} sampled songs from {config.data.song_vectors_file}...")
    try:
        with open(config.data.song_vectors_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in tqdm(reader, desc="Reading sampled vectors"):
                if len(row) < 2: continue
                song_id = row[0].strip() # Ensure song_id is stripped
                
                if song_id in sampled_song_ids_set:
                    try:
                        embed = np.array(row[1:], dtype=np.float32)
                        X_sample_list.append(embed)
                        labels_sample_list.append(str(semantic_ids_map[song_id]))
                    except ValueError:
                        print(f"Skipping vector for song_id {song_id} due to non-numeric data.")
                        continue
    except FileNotFoundError:
        print(f"ERROR: Song vectors file not found at {config.data.song_vectors_file}.")
        return

    X_sample = np.array(X_sample_list)
    labels_sample = np.array(labels_sample_list)
    
    unique_labels_sample = np.unique(labels_sample)

    # Ensure there's more than one cluster in the sample
    if len(unique_labels_sample) <= 1:
        print("Only one unique semantic ID found in the sample. Cannot calculate clustering metrics.")
        return
    
    # --- Calculate Calinski-Harabasz Index --- (Higher is better)
    if len(X_sample) > len(unique_labels_sample) and len(unique_labels_sample) > 1:
        ch_score = calinski_harabasz_score(X_sample, labels_sample)
        print(f"Calinski-Harabasz Index: {ch_score:.4f} (Higher is better)")
    else:
        print("Not enough samples relative to unique IDs in sample for Calinski-Harabasz Index.")

    # --- Calculate Davies-Bouldin Index --- (Lower is better)
    if len(unique_labels_sample) > 1:
        db_score = davies_bouldin_score(X_sample, labels_sample)
        print(f"Davies-Bouldin Index: {db_score:.4f} (Lower is better)")
    else:
        print("Not enough unique IDs in sample for Davies-Bouldin Index.")

    # --- Calculate Silhouette Score --- (Higher is better)
    if len(unique_labels_sample) > 1 and len(X_sample) > 1: # Silhouette requires n_samples > 1 and n_unique_labels > 1
        print(f"Calculating Silhouette Score on {len(X_sample)} samples...")
        silhouette_avg = silhouette_score(X_sample, labels_sample)
        print(f"Silhouette Score: {silhouette_avg:.4f} (Higher is better)")
    else:
        print("Not enough samples or unique IDs in sample for Silhouette Score.")
    
    print("="*80)

if __name__ == '__main__':
    try:
        main_config = Config()
        # You can adjust the sample_size here if needed
        calculate_metrics_batch(main_config, sample_size=50000) 
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure all required files exist at the specified paths in config.py.")
