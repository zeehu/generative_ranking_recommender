"""
Generate predictions using vLLM for high-throughput offline inference.
OPTIMIZED VERSION: Uses vLLM engine, Tensor Parallelism, and Continuous Batching.

Output Format:
Query 	 SemID||SongID||SongName||Singer 	 SemID||SongID||SongName||Singer ...
"""
import os
import sys
import argparse
import logging
import torch
import json
import re
from typing import List, Dict, Tuple
from collections import defaultdict
from tqdm import tqdm

# Add project root to sys.path for config and utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config
from src.common.utils import setup_logging

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("Error: vLLM is not installed. Please run `pip install vllm`.")
    sys.exit(1)

logger = logging.getLogger(__name__)

class PredictionFormatter:
    """Helper class to format vLLM outputs back to song information."""
    
    def __init__(self, config: Config):
        self.config = config
        self.semantic_to_song_cluster = self._create_reverse_map()
        self.song_info_map = self._load_song_info()
        
    def _create_reverse_map(self) -> Dict[Tuple[int, ...], List[str]]:
        """Load semantic ID -> song IDs mapping."""
        mapping = defaultdict(list)
        semantic_ids_file = self.config.data.semantic_ids_file
        
        logger.info(f"Loading semantic ID map from {semantic_ids_file}...")
        if not os.path.exists(semantic_ids_file):
            logger.error(f"Semantic ID file not found: {semantic_ids_file}")
            return mapping

        with open(semantic_ids_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading ID Map"):
                try:
                    item = json.loads(line)
                    # Assuming item['semantic_ids'] is a list like [1, 2, 3]
                    mapping[tuple(item['semantic_ids'])].append(item['song_id'])
                except json.JSONDecodeError:
                    continue
        return mapping

    def _load_song_info(self) -> Dict[str, Dict[str, str]]:
        """Load song metadata."""
        import csv
        mapping = {}
        path = getattr(self.config.data, 'song_info_file', 'data/gen_song_info.csv')
        
        logger.info(f"Loading song info from {path}...")
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    if len(row) >= 3:
                        mapping[row[0]] = {"name": row[1], "singer": row[2]}
        except FileNotFoundError:
            logger.warning(f"Song info file not found: {path}")
        return mapping

    def clean_text(self, text: str) -> str:
        if not text: return "Unknown"
        return str(text).replace("\n", " ").replace("\r", " ").replace("\t", " ").replace("||", " ").strip()

    def parse_output(self, output_text: str) -> List[str]:
        """
        Parse raw T5 output string into formatted song entries.
        Expected format from T5: <id_l1_X> <id_l2_Y> <id_l3_Z> ...
        """
        # Simple regex to find the numeric ID in <id_...> tokens
        # Matches patterns like <id_l1_123>, <id_l2_45>, etc.
        # Note: vLLM output might contain spaces between tokens.
        
        # Strategy: Extract all numbers that are part of an <id_...> token
        # Pattern: <id_l[1-3]_(\d+)> 
        
        # Pattern: <id_l[1-3]_(\d+)> 
        matches = re.findall(r"<id_l[1-3]_(\d+)>", output_text)
        
        if not matches:
            return []
            
        ids = [int(m) for m in matches]
        
        # Group into triplets (L1, L2, L3)
        # Assuming 3 levels hierarchy
        levels = 3
        formatted_entries = []
        seen_songs = set()
        
        for i in range(0, len(ids), levels):
            chunk = ids[i : i + levels]
            if len(chunk) != levels:
                continue
                
            sem_id_tuple = tuple(chunk)
            
            # Expand to songs
            if sem_id_tuple in self.semantic_to_song_cluster:
                song_ids = self.semantic_to_song_cluster[sem_id_tuple]
                
                for song_id in song_ids:
                    if song_id in seen_songs:
                        continue
                    seen_songs.add(song_id)
                    
                    info = self.song_info_map.get(song_id, {"name": "Unknown", "singer": "Unknown"})
                    s_name = self.clean_text(info.get("name", "Unknown"))
                    s_singer = self.clean_text(info.get("singer", "Unknown"))
                    
                    # Format: SemID||SongID||SongName||Singer
                    entry = f"{sem_id_tuple}||{song_id}||{s_name}||{s_singer}"
                    formatted_entries.append(entry)
                    
        return formatted_entries

def main():
    parser = argparse.ArgumentParser(description="vLLM Inference for Playlist Generation")
    parser.add_argument("--input_file", type=str, default="data/query_tag_map_top_mixsongid_rain_2025-11-17.txt")
    parser.add_argument("--output_file", type=str, default="outputs/predictions_vllm.txt")
    parser.add_argument("--model_path", type=str, default="models/generator/final_model/")
    parser.add_argument("--tensor_parallel_size", type=int, default=2, help="Number of GPUs to use")
    parser.add_argument("--sample_size", type=int, default=None, help="Debug: sample N queries")
    parser.add_argument("--seed", type=int, default=42)
    
    # Decoding args
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--num_return_sequences", type=int, default=5, help="Number of sequences per query")

    args = parser.parse_args()
    
    setup_logging(log_file="logs/generate_predictions_vllm.log")
    logger.info("--- Starting vLLM Inference ---")

    # 1. Load Queries
    logger.info(f"Loading queries from {args.input_file}...")
    queries = []
    if os.path.exists(args.input_file):
        with open(args.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split('\t')
                if len(parts) >= 1:
                    queries.append(parts[0])
    else:
        logger.error("Input file not found.")
        sys.exit(1)
        
    # Deduplicate
    queries = list(dict.fromkeys(queries))
    logger.info(f"Loaded {len(queries)} unique queries.")
    
    if args.sample_size:
        import random
        random.seed(args.seed)
        queries = random.sample(queries, args.sample_size)
        logger.info(f"Sampled {len(queries)} queries for debugging.")

    # 2. Initialize Helper (Formatter)
    # Do this before loading vLLM to fail fast if data missing
    config = Config()
    formatter = PredictionFormatter(config)

    # 3. Initialize vLLM Engine
    logger.info(f"Initializing vLLM Engine on {args.tensor_parallel_size} GPUs...")
    try:
        # vLLM handles distributed initialization internally
        llm = LLM(
            model=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype="bfloat16", # Optimization for L20
            trust_remote_code=True
        )
    except Exception as e:
        logger.error(f"Failed to initialize vLLM: {e}")
        sys.exit(1)

    # 4. Define Sampling Params
    # Setting seed here ensures reproducibility for EACH request
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=args.num_return_sequences, # Generate N candidates per query
        seed=args.seed 
    )

    # 5. Run Inference
    logger.info("Running inference (vLLM handles batching)...")
    # Pass the list of strings directly. vLLM manages the queue.
    outputs = llm.generate(queries, sampling_params)

    # 6. Process and Save Results
    logger.info("Processing results...")
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for output in tqdm(outputs, desc="Formatting"):
            query = formatter.clean_text(output.prompt)
            
            # Aggregate all generated sequences for this query
            all_entries = []
            for generated_seq in output.outputs:
                # parse_output returns a list of formatted song strings
                entries = formatter.parse_output(generated_seq.text)
                all_entries.extend(entries)
            
            # Deduplicate entries (same song might be generated in different beams/samples)
            # Order is preserved by dict keys
            unique_entries = list(dict.fromkeys(all_entries))
            
            if unique_entries:
                line = f"{query}\t" + "\t".join(unique_entries) + "\n"
                f.write(line)

    logger.info(f"Done. Results saved to {args.output_file}")

if __name__ == "__main__":
    main()