"""
Interactive script to evaluate the quality of generated semantic IDs.

This tool allows you to pick a song and find its nearest neighbors within the
same semantic cluster (at the first level), ranked by cosine similarity of their
original vectors. This provides a qualitative measure of the clustering quality.
It also provides quantitative metrics (Silhouette, CH, DB scores).
"""
import os
import sys
import json
import csv
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Add project root to sys.path to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config

class Evaluator:
    def __init__(self, config: Config):
        print("--- Semantic ID Quality Evaluator ---")
        self.config = config
        self.song_info = self._load_song_info(config.data.song_info_file)
        self.song_vectors = self._load_song_vectors(config.data.song_vectors_file)
        self.semantic_ids, self.l1_clusters = self._load_semantic_ids(config.data.semantic_ids_file)

        if not self.song_info or not self.song_vectors or not self.semantic_ids:
            raise RuntimeError("Failed to load necessary files. Please check paths in config.py.")

        print("\nInitialization complete. Evaluator is ready.")

    def _load_song_info(self, path):
        print(f"Loading song info from: {path}")
        info = {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                for row in tqdm(reader, desc="Reading song info"):
                    if len(row) >= 3:
                        info[row[0]] = {"name": row[1], "singer": row[2]}
        except FileNotFoundError:
            print(f"Warning: Song info file not found at {path}. Output will not contain names.")
        return info

    def _load_song_vectors(self, path):
        print(f"Loading song vectors from: {path}")
        vectors = {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in tqdm(reader, desc="Reading song vectors"):
                    if len(row) > 1:
                        vectors[row[0]] = np.array(row[1:], dtype=np.float32)
        except FileNotFoundError:
            print(f"ERROR: Song vectors file not found at {path}.")
        return vectors

    def _load_semantic_ids(self, path):
        print(f"Loading semantic IDs from: {path}")
        s_ids = {}
        l1_clusters = defaultdict(list)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="Reading semantic IDs"):
                    item = json.loads(line)
                    song_id = item['song_id']
                    sem_id_tuple = tuple(item['semantic_ids'])
                    s_ids[song_id] = sem_id_tuple
                    if sem_id_tuple:
                        l1_clusters[sem_id_tuple[0]].append(song_id)
        except FileNotFoundError:
            print(f"ERROR: Semantic ID file not found at {path}.")
        return s_ids, l1_clusters

    def find_neighbors(self, target_song_id: str, top_n: int = 10):
        """Finds and prints the nearest neighbors for a given song ID."""
        if target_song_id not in self.semantic_ids:
            print(f"Error: Song ID '{target_song_id}' not found in the semantic ID mapping.")
            return
        
        target_sem_id = self.semantic_ids[target_song_id]
        target_vector = self.song_vectors.get(target_song_id)
        target_info = self.song_info.get(target_song_id, {"name": "N/A", "singer": "N/A"})

        if target_vector is None:
            print(f"Error: Vector for song ID '{target_song_id}' not found.")
            return

        print("\n" + "-"*80)
        print(f"Target Song: {target_info['name']} - {target_info['singer']} (ID: {target_song_id})")
        print(f"Semantic ID: {target_sem_id}")
        print("-"*80)

        l1_cluster_id = target_sem_id[0]
        neighbor_song_ids = self.l1_clusters.get(l1_cluster_id, [])

        if len(neighbor_song_ids) <= 1:
            print("No other songs found in the same L1 cluster.")
            return

        print(f"Found {len(neighbor_song_ids)} songs in L1 cluster '{l1_cluster_id}'. Calculating similarities...")

        # Normalize the target vector once
        target_vector_norm = target_vector / np.linalg.norm(target_vector)

        similarities = []
        for neighbor_id in neighbor_song_ids:
            if neighbor_id == target_song_id:
                continue
            
            neighbor_vector = self.song_vectors.get(neighbor_id)
            if neighbor_vector is not None:
                # Normalize neighbor vector and compute cosine similarity
                neighbor_vector_norm = neighbor_vector / np.linalg.norm(neighbor_vector)
                cosine_sim = np.dot(target_vector_norm, neighbor_vector_norm)
                similarities.append((neighbor_id, cosine_sim))

        # Sort by similarity in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)

        print(f"\nTop {top_n} most similar neighbors in the same L1 cluster:")
        for i, (song_id, sim) in enumerate(similarities[:top_n], 1):
            info = self.song_info.get(song_id, {"name": "N/A", "singer": "N/A"})
            sem_id = self.semantic_ids.get(song_id, "N/A")
            print(f"  {i}. {info['name']} - {info['singer']} (Similarity: {sim:.4f}) (ID: {sem_id})")

    def find_songs_by_semantic_id(self, id_prefix_str: str, max_display: int = 10):
        """Finds and displays songs that match a given semantic ID prefix."""
        try:
            # Convert string like "10,55" to a tuple of integers
            id_prefix = tuple(map(int, id_prefix_str.split(',')))
        except ValueError:
            print(f"Error: Invalid semantic ID format. Please use comma-separated integers, e.g., '10,55'.")
            return

        if not 1 <= len(id_prefix) <= 3:
            print("Error: Please provide 1, 2, or 3 levels for the semantic ID.")
            return

        print(f"\nSearching for songs with semantic ID prefix: {id_prefix}...")
        
        matching_songs = []
        # This is a linear scan, which is slow for large datasets but fine for this tool.
        for song_id, sem_id in self.semantic_ids.items():
            if sem_id[:len(id_prefix)] == id_prefix:
                matching_songs.append(song_id)
        
        if not matching_songs:
            print("No songs found matching this semantic ID prefix.")
            return

        print(f"Found {len(matching_songs)} songs. Showing up to {max_display}:")
        for i, song_id in enumerate(matching_songs[:max_display], 1):
            info = self.song_info.get(song_id, {"name": "N/A", "singer": "N/A"})
            sem_id = self.semantic_ids.get(song_id, "N/A")
            print(f"  {i}. {info['name']} - {info['singer']} (Full ID: {sem_id})")

    def calculate_clustering_metrics(self, sample_size: int = 50000):
        """Calculates and prints clustering evaluation metrics (Silhouette, CH, DB)."""
        print("\n" + "="*80)
        print("  Calculating Clustering Metrics (Silhouette, CH, DB)")
        print("="*80)

        # Filter for songs that have both vector and semantic ID
        valid_song_ids = [sid for sid in self.song_vectors.keys() if sid in self.semantic_ids]
        
        if not valid_song_ids:
            print("No valid songs with both vectors and semantic IDs found for metric calculation.")
            return

        # --- Unified Sampling for all metrics ---
        # If the dataset is larger than sample_size, we sample to reduce computation.
        if len(valid_song_ids) > sample_size:
            print(f"Sampling {sample_size} points for all metric calculations (full dataset is too large)...")
            # Randomly sample indices
            indices = np.random.choice(len(valid_song_ids), sample_size, replace=False)
            sampled_song_ids = [valid_song_ids[i] for i in indices]
        else:
            sampled_song_ids = valid_song_ids
        
        # Prepare data: X (vectors), labels (full semantic ID as string) from the sampled IDs
        X_sample_list = []
        labels_sample_list = []

        print(f"Collecting {len(sampled_song_ids)} data points for metrics...")
        for song_id in tqdm(sampled_song_ids, desc="Collecting data for metrics"):
            X_sample_list.append(self.song_vectors[song_id])
            labels_sample_list.append(str(self.semantic_ids[song_id])) # Convert tuple to string for label

        X_sample = np.array(X_sample_list)
        labels_sample = np.array(labels_sample_list)
        
        unique_labels_sample = np.unique(labels_sample)

        # Ensure there's more than one cluster in the sample
        if len(unique_labels_sample) <= 1:
            print("Only one unique semantic ID found in the sample. Cannot calculate clustering metrics.")
            return
        
        # --- Calculate Calinski-Harabasz Index --- (Higher is better)
        # Requires at least 2 clusters and n_samples > n_clusters
        if len(X_sample) > len(unique_labels_sample) and len(unique_labels_sample) > 1:
            ch_score = calinski_harabasz_score(X_sample, labels_sample)
            print(f"Calinski-Harabasz Index: {ch_score:.4f} (Higher is better)")
        else:
            print("Not enough samples relative to unique IDs in sample for Calinski-Harabasz Index.")

        # --- Calculate Davies-Bouldin Index --- (Lower is better)
        # Requires at least 2 clusters
        if len(unique_labels_sample) > 1:
            db_score = davies_bouldin_score(X_sample, labels_sample)
            print(f"Davies-Bouldin Index: {db_score:.4f} (Lower is better)")
        else:
            print("Not enough unique IDs in sample for Davies-Bouldin Index.")

        # --- Calculate Silhouette Score --- (Higher is better)
        # Requires at least 2 clusters and n_samples > 1
        if len(unique_labels_sample) > 1 and len(X_sample) > 1:
            print(f"Calculating Silhouette Score on {len(X_sample)} samples...")
            silhouette_avg = silhouette_score(X_sample, labels_sample)
            print(f"Silhouette Score: {silhouette_avg:.4f} (Higher is better)")
        else:
            print("Not enough samples or unique IDs in sample for Silhouette Score.")
        
        print("="*80)

    def run_interactive(self):
        """Starts the interactive command-line session."""
        print("\n" + "="*50)
        print("  🎶 语义ID质量交互式检验工具 🎶")
        print("="*50)
        print("  - 输入 歌曲ID (e.g., '12345') 来查找相似歌曲。")
        print("  - 输入 语义ID (e.g., '10' 或 '10,55') 来查看该簇下的歌曲。")
        print("  - 输入 'metrics' 来计算聚类评估指标。")
        print("  - 输入 'exit' 或 'quit' 即可退出。")
        print("-"*50)

        while True:
            try:
                prompt = input("\n请输入 song_id, semantic_id, 或 'metrics' > ").strip()
                if prompt.lower() in ['exit', 'quit']:
                    break
                if not prompt:
                    continue
                
                if prompt.lower() == 'metrics':
                    self.calculate_clustering_metrics()
                else:
                    # Simple dispatch logic
                    is_semantic_id_query = ',' in prompt or (prompt.isdigit() and len(prompt) < 4) # Heuristic

                    if is_semantic_id_query:
                        self.find_songs_by_semantic_id(prompt)
                    else:
                        self.find_neighbors(prompt)

            except KeyboardInterrupt:
                break
        print("\n感谢使用，再见！")

if __name__ == '__main__':
    try:
        # Instantiate the main Config object to get all configurations and file paths
        main_config = Config()

        evaluator = Evaluator(config=main_config)
        evaluator.run_interactive()
    except Exception as e:
        print(f"\nAn error occurred during initialization: {e}")
        print("Please ensure all required files exist at the specified paths in config.py.")
