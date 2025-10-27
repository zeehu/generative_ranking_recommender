
import torch
import numpy as np
import os
import os.path as osp
import pickle
import csv
import json
from tqdm.auto import tqdm
from dataclasses import dataclass, field
from typing import List, Dict

# Assuming balancekmeans is in the python path
# You might need to adjust sys.path if it's not installed
try:
    from balancekmeans import KMeans, pairwise_distance_full
except ImportError:
    # If balancekmeans is not in the standard path, add its location.
    # This is based on the original script's structure.
    import sys
    # Adjust this path to where the 'BalanceRqKMeans' directory is relative to your project root
    sys.path.append(osp.abspath(osp.join(__file__, '../../..')))
    from BalanceRqKMeans.balancekmeans import KMeans, pairwise_distance_full


@dataclass
class HierarchicalRQConfig:
    """Configuration for SimplifiedHierarchicalRQ."""
    # Number of clusters to train for each layer.
    layer_clusters: List[int] = field(default_factory=lambda: [128, 1280, 2560])
    # Number of clusters to actually use/select from each layer.
    need_clusters: List[int] = field(default_factory=lambda: [128, 128, 256])
    embedding_dim: int = 256
    # K-means training iterations.
    iter_limit: int = 100

    def __post_init__(self):
        assert len(self.need_clusters) == len(self.layer_clusters), "Length of need_clusters must match layer_clusters."
        assert self.layer_clusters[0] == self.need_clusters[0], "First layer's 'layer_clusters' must equal 'need_clusters'."

class SimplifiedHierarchicalRQ:
    """
    A simplified, object-oriented implementation of hierarchical residual quantization
    with balanced k-means and dynamic center selection.
    """
    def __init__(self, config: HierarchicalRQConfig):
        self.config = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # State variables that will be populated during training
        self.trained_kmeans_models: List[KMeans] = []
        self.dynamic_match_matrix = None # For the final layer
        self.final_layer_centers = None # For the final layer
        self.middle_layer_centers = None # For recursive layers
        print(f"Initialized SimplifiedHierarchicalRQ on device: {self.device}")

    def _load_data(self, data_path: str) -> (List[str], torch.Tensor):
        """
        Loads all song embeddings from a single CSV file with no header.
        Format: song_id,dim_1,dim_2,...,dim_N
        """
        song_ids, embeddings = [], []
        print(f"Loading data from CSV file (no header): {data_path}...")

        if not osp.isfile(data_path):
            raise FileNotFoundError(f"The specified data file was not found: {data_path}")

        with open(data_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in tqdm(reader, desc="Reading rows from CSV"):
                if len(row) < 2: continue
                
                song_id = row[0]
                # Convert all subsequent columns to a numpy array of floats
                try:
                    embed = np.array(row[1:], dtype=np.float32)
                except ValueError:
                    # Skip row if vector part contains non-numeric data
                    print(f"Skipping row for song_id {song_id} due to non-numeric vector data.")
                    continue

                if embed.shape[0] == self.config.embedding_dim:
                    song_ids.append(song_id)
                    embeddings.append(embed)

        if not song_ids:
            raise ValueError("No valid data with the correct embedding dimension found in the CSV file.")
        
        # Use half precision if any layer has a large number of clusters
        use_half = any(n > 512 for n in self.config.layer_clusters)
        tensor_embeddings = torch.from_numpy(np.vstack(embeddings))
        return song_ids, tensor_embeddings.half() if use_half else tensor_embeddings.float()

    def _get_residuals(self, data: torch.Tensor, kmeans: KMeans) -> torch.Tensor:
        """Calculates residuals after a k-means step."""
        residuals = []
        batch_size = 100000 # Adjust based on GPU memory
        
        for i in tqdm(range(0, len(data), batch_size), desc="Calculating Residuals"):
            batch = data[i:i+batch_size].to(self.device)
            with torch.no_grad():
                cluster_ids = kmeans.predict(batch)
                centers = kmeans.cluster_centers[cluster_ids]
                residual = batch - centers
            residuals.append(residual.cpu())
            del batch, centers, residual
            if self.device.type == 'cuda': torch.cuda.empty_cache()

        return torch.cat(residuals)

    def train(self, data_path: str):
        """
        Trains the full hierarchical RQ model.
        """
        song_ids, embeddings = self._load_data(data_path)
        current_data = embeddings
        
        all_layer_ids: Dict[str, List[int]] = {sid: [] for sid in song_ids}
        previous_level_ids = None

        for layer_idx, (n_clusters, n_need) in enumerate(zip(self.config.layer_clusters, self.config.need_clusters)):
            print(f"--- Training Layer {layer_idx + 1}/{len(self.config.layer_clusters)} ---")
            
            use_half = n_clusters > 512
            
            kmeans = KMeans(n_clusters=n_clusters, device=self.device, balanced=True)

            # Layer-specific training logic
            if layer_idx == 0:
                # First layer uses fit_by_min_loss to control cluster size
                target_nodes = np.prod(self.config.need_clusters[1:])
                kmeans.fit_by_min_loss(
                    X=current_data, target_nodes_num=target_nodes, iter_limit=self.config.iter_limit, half=use_half
                )
                self.trained_kmeans_models.append(kmeans)
                cluster_ids = kmeans.predict(current_data.to(self.device))

            elif n_clusters == n_need: # Middle layers (if they exist and are simple)
                # This is a recursive-style training within each previous cluster
                # Note: The original code had complex recursive logic. This is a simplified placeholder.
                # For a true recursive implementation, one would train a separate KMeans for each parent cluster.
                # Here, we train one large KMeans, which is a simplification.
                print("Training a single KMeans for the whole layer (simplified approach).")
                kmeans.fit(X=current_data, iter_limit=self.config.iter_limit, half=use_half)
                self.trained_kmeans_models.append(kmeans)
                cluster_ids = kmeans.predict(current_data.to(self.device))

            else: # Final layer with dynamic center selection
                print("Training final layer with dynamic center selection...")
                # This part is complex, retaining the original logic.
                
                # 1. Train multiple KMeans to create a large pool of candidate centers
                kmeans_part1 = KMeans(n_clusters=n_clusters, device=self.device, balanced=True)
                kmeans_part1.fit(X=current_data, iter_limit=20, half=use_half)
                
                kmeans_part2 = KMeans(n_clusters=n_clusters, device=self.device, balanced=True)
                kmeans_part2.fit(X=current_data, iter_limit=20, half=use_half)

                candidate_centers = torch.cat([kmeans_part1.cluster_centers, kmeans_part2.cluster_centers], dim=0)
                self.final_layer_centers = candidate_centers
                self.trained_kmeans_models.append(None) # Placeholder for final layer model

                # 2. Assign a unique subset of centers for each subgroup from previous layers
                # This requires IDs from the two previous layers
                prev_ids_l1 = torch.tensor([all_layer_ids[sid][layer_idx-2] for sid in song_ids])
                prev_ids_l2 = torch.tensor([all_layer_ids[sid][layer_idx-1] for sid in song_ids])

                self.dynamic_match_matrix = self._get_dynamic_match_matrix(
                    current_data, prev_ids_l1, prev_ids_l2, candidate_centers
                )

                # 3. Assign clusters based on the dynamic matrix
                cluster_ids = self._predict_with_dynamic_matrix(
                    current_data, prev_ids_l1, prev_ids_l2, candidate_centers, self.dynamic_match_matrix
                )

            # --- Post-layer processing ---
            current_ids_np = cluster_ids.cpu().numpy()
            for i, song_id in enumerate(song_ids):
                all_layer_ids[song_id].append(current_ids_np[i])
            
            previous_level_ids = cluster_ids

            # Calculate residuals for the next layer
            if layer_idx < len(self.config.layer_clusters) - 1:
                print("Calculating residuals for next layer...")
                # The 'kmeans' object needs to be correctly set for residual calculation
                # For the dynamic layer, we need a different way to get the center for each point
                if self.dynamic_match_matrix is not None and layer_idx == len(self.config.layer_clusters) - 2:
                    # This case is complex as we don't have one single kmeans model.
                    # We'd need to calculate residuals based on the dynamically assigned centers.
                    # For simplification, we'll stop residual calculation before the final dynamic layer in this version.
                    print("Skipping residual calculation after dynamic layer for simplicity.")
                    current_data = None 
                else:
                    current_data = self._get_residuals(current_data, kmeans)
            
        self.semantic_ids = all_layer_ids
        print("Training complete.")

    def _get_dynamic_match_matrix(self, data, prev_ids_l1, prev_ids_l2, candidate_centers):
        """
        Implements the complex logic from the original script to create a dynamic
        match matrix for the final layer.
        """
        print("Generating dynamic match matrix for the final layer...")
        n_prev1 = self.config.need_clusters[-3]
        n_prev2 = self.config.need_clusters[-2]
        n_need = self.config.need_clusters[-1]
        n_candidates = candidate_centers.shape[0]
        
        match_matrix = []
        
        # This double loop is computationally intensive
        for i in tqdm(range(n_prev1), desc="Matrix Gen L1"):
            for j in range(n_prev2):
                subgroup_mask = (prev_ids_l1 == i) & (prev_ids_l2 == j)
                subgroup_indices = torch.where(subgroup_mask)[0]
                
                sub_centers = None
                if len(subgroup_indices) == 0:
                    sub_centers = candidate_centers[np.random.choice(n_candidates, n_need, replace=False)].cpu().numpy()
                elif len(subgroup_indices) <= n_need:
                    sub_centers = data[subgroup_indices].cpu().numpy()
                else:
                    # Run a temporary KMeans on the subgroup to find its ideal centers
                    sub_data = data[subgroup_indices].to(self.device)
                    temp_kmeans = KMeans(n_clusters=n_need, device=self.device, balanced=True)
                    temp_kmeans.fit(X=sub_data, iter_limit=20, tqdm_flag=False)
                    sub_centers = temp_kmeans.cluster_centers.cpu().numpy()

                # Find the closest candidate centers for these ideal sub-centers
                dist_matrix = np.linalg.norm(sub_centers[:, np.newaxis, :] - candidate_centers.cpu().numpy()[np.newaxis, :, :], axis=2)
                
                selected_indices = set()
                row_match = [0] * n_candidates
                
                # Greedily assign unique candidates
                for k in range(len(sub_centers)):
                    sorted_candidate_indices = np.argsort(dist_matrix[k])
                    for candidate_idx in sorted_candidate_indices:
                        if candidate_idx not in selected_indices:
                            selected_indices.add(candidate_idx)
                            break
                
                # Fill up if not enough unique centers were found
                while len(selected_indices) < n_need:
                    rand_idx = np.random.randint(n_candidates)
                    if rand_idx not in selected_indices:
                        selected_indices.add(rand_idx)

                for idx in selected_indices:
                    row_match[idx] = 1
                
                match_matrix.append(row_match)
        
        return torch.tensor(match_matrix, dtype=torch.float32)

    def _predict_with_dynamic_matrix(self, data, prev_ids_l1, prev_ids_l2, candidate_centers, match_matrix):
        """Assigns cluster IDs using the pre-computed dynamic match matrix."""
        print("Predicting with dynamic matrix...")
        n_prev2 = self.config.need_clusters[-2]
        
        # Combine previous layer IDs to get the subgroup index
        subgroup_indices = prev_ids_l1 * n_prev2 + prev_ids_l2
        
        final_ids = []
        batch_size = 10000
        for i in tqdm(range(0, len(data), batch_size), desc="Dynamic Prediction"):
            batch_data = data[i:i+batch_size].to(self.device)
            batch_subgroup_indices = subgroup_indices[i:i+batch_size]
            
            dist = pairwise_distance_full(batch_data, candidate_centers.to(self.device))
            
            # Get the corresponding match rows for the batch
            batch_match_matrix = match_matrix[batch_subgroup_indices].to(self.device)
            
            # Mask out non-allowed centers
            dist[batch_match_matrix == 0] = float('inf')
            
            # Find the closest allowed center
            batch_final_ids = torch.argmin(dist, dim=1)
            final_ids.append(batch_final_ids)
        
        return torch.cat(final_ids)

    def save_model(self, path: str):
        """Saves the trained model components to a file."""
        print(f"Saving model to {path}...")
        with open(path, 'wb') as f:
            pickle.dump({
                'config': self.config,
                'trained_kmeans_models': [km.cluster_centers if km else None for km in self.trained_kmeans_models],
                'dynamic_match_matrix': self.dynamic_match_matrix,
                'final_layer_centers': self.final_layer_centers,
            }, f)
        print("Model saved.")

    @classmethod
    def load_model(cls, path: str):
        """Loads a trained model from a file."""
        print(f"Loading model from {path}...")
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        model = cls(checkpoint['config'])
        
        # Re-create KMeans objects from saved centers
        model.trained_kmeans_models = []
        for centers in checkpoint['trained_kmeans_models']:
            if centers is not None:
                km = KMeans(n_clusters=centers.shape[0], cluster_centers=centers, device=model.device)
                model.trained_kmeans_models.append(km)
            else:
                model.trained_kmeans_models.append(None)

        model.dynamic_match_matrix = checkpoint['dynamic_match_matrix']
        model.final_layer_centers = checkpoint['final_layer_centers']
        print("Model loaded.")
        return model

    def save_semantic_ids(self, output_file: str):
        """Saves the generated song_id -> semantic_id mapping in JSONL format."""
        if not hasattr(self, 'semantic_ids'):
            print("No semantic IDs generated yet. Run train() or predict() first.")
            return
            
        print(f"Saving semantic IDs to {output_file}...")
        unique_ids = set()
        with open(output_file, 'w', encoding='utf-8') as f:
            for song_id, ids in self.semantic_ids.items():
                # Create a dictionary for the JSON object
                data = {"song_id": song_id, "semantic_ids": ids}
                # Write the JSON string followed by a newline
                f.write(json.dumps(data) + '\n')
                unique_ids.add(tuple(ids))
        
        print(f"Saved {len(self.semantic_ids)} total IDs.")
        print(f"Found {len(unique_ids)} unique semantic IDs.")


if __name__ == '__main__':
    # This script is now an executable for a specific task.
    
    # 1. Define input and output paths
    input_csv_path = "outputs/song_vectors.csv"
    output_dir = "outputs/semantic_id"
    
    print(f"--- Starting Semantic ID Generation ---")
    print(f"Input file: {input_csv_path}")
    print(f"Output directory: {output_dir}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # 2. Initialize model with configuration optimized for large-scale data
    # The config defaults are already set for this purpose.
    try:
        config = HierarchicalRQConfig()
        model = SimplifiedHierarchicalRQ(config)

        # 3. Train the model using data from the CSV file
        model.train(data_path=input_csv_path)

        # 4. Define save paths and save the model and results
        model_save_path = osp.join(output_dir, "semantic_rq_model.pkl")
        # Use the project-standard filename
        results_save_path = osp.join(output_dir, "song_semantic_ids.jsonl")
        
        model.save_model(model_save_path)
        model.save_semantic_ids(results_save_path)

        print(f"--- Semantic ID Generation Complete ---")
        print(f"Model saved to: {model_save_path}")
        print(f"Semantic IDs saved to: {results_save_path}")

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Please ensure the input file exists and the path is correct.")
    except ValueError as e:
        print(f"\nERROR: {e}")
        print("An error occurred during data processing. Please check the input file format and content.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
