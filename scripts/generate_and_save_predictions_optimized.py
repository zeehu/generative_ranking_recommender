"""
Generate predictions using T5 model and save detailed song information.
OPTIMIZED VERSION: Supports Multi-GPU Data Parallelism for faster inference.

Output Format:
Query 	 SemID||SongID||SongName||Singer 	 SemID||SongID||SongName||Singer ...
"""
import os
import sys
import argparse
import logging
import random
import torch
import math
import time
from tqdm import tqdm
from torch import multiprocessing as mp

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config
from src.generator.inference_t5 import PlaylistGenerator
from src.common.utils import setup_logging

# Set start method to spawn to ensure CUDA compatibility in multiprocessing
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """
    Clean text to ensure file integrity.
    1. Removes newlines/tabs to maintain one-line-per-query structure.
    2. Removes the internal separator (||) to prevent parsing errors.
    """
    if not text:
        return "Unknown"
    text = str(text)
    # Replace structural characters with spaces
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    # Remove our custom delimiter if it appears in text
    text = text.replace("||", " ")
    return text.strip()

def run_inference_worker(rank, gpu_id, queries, args, output_file):
    """
    Worker function for multi-GPU inference.
    Each worker runs on a dedicated GPU and processes a subset of queries.
    """
    # 1. Setup environment for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 2. Setup independent logging
    log_file = f"logs/gen_worker_{rank}.log"
    os.makedirs("logs", exist_ok=True)
    worker_logger = logging.getLogger(f"worker_{rank}")
    worker_logger.setLevel(logging.INFO)
    # Clear existing handlers to avoid duplicate logs if re-initialized
    if worker_logger.hasHandlers():
        worker_logger.handlers.clear()
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    worker_logger.addHandler(handler)
    
    worker_logger.info(f"Worker {rank} started on GPU {gpu_id} (Physical ID) with {len(queries)} queries.")
    
    # 3. Set seeds for reproducibility
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    
    # 4. Initialize Config and Generator
    config = Config()
    config.device = "cuda" 
    
    try:
        worker_logger.info("Initializing Generator model...")
        # Ensure model is loaded in eval mode
        generator = PlaylistGenerator(config, model_path=args.model_path, use_trie_constraint=True)
    except Exception as e:
        worker_logger.error(f"Failed to initialize generator: {e}")
        return

    # 5. Inference Loop
    batch_size = args.batch_size
    worker_logger.info(f"Starting inference with batch_size={batch_size}")
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for i in tqdm(range(0, len(queries), batch_size), desc=f"GPU {gpu_id}", position=rank):
            batch_queries = queries[i : i + batch_size]
            try:
                do_sample = (args.strategy == "sample")
                
                # Batch Generation
                batch_results = generator.generate_batch(
                    batch_queries, 
                    max_songs=50, 
                    do_sample=do_sample, 
                    num_beams=args.num_beams,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p
                )
                
                # Process and Write Results
                for query, results in zip(batch_queries, batch_results):
                    output_parts = []
                    # Using a set to avoid duplicate songs for the same query if expanded from different sem_ids
                    seen_songs = set()

                    for item in results:
                        semantic_id_tuple = item['semantic_id']
                        
                        # Expand semantic ID cluster to all songs
                        if semantic_id_tuple in generator.semantic_to_song_cluster:
                            song_ids = generator.semantic_to_song_cluster[semantic_id_tuple]
                            
                            for song_id in song_ids:
                                if song_id in seen_songs:
                                    continue
                                seen_songs.add(song_id)

                                song_info = generator.song_info_map.get(song_id, {"name": "Unknown", "singer": "Unknown"})
                                song_name = clean_text(song_info.get("name", "Unknown"))
                                singer = clean_text(song_info.get("singer", "Unknown"))
                                
                                # Format: SemID||SongID||SongName||Singer
                                sem_id_str = str(semantic_id_tuple) # e.g., "(1, 2, 3)"
                                entry_str = f"{sem_id_str}||{song_id}||{song_name}||{singer}"
                                output_parts.append(entry_str)
                    
                    # Write line: CleanQuery <TAB> Result1 <TAB> Result2 ...
                    if output_parts:
                        clean_query = clean_text(query)
                        output_line = f"{clean_query}\t" + "\t".join(output_parts) + "\n"
                        out_f.write(output_line)
                    
            except Exception as e:
                worker_logger.error(f"Error processing batch starting at index {i}: {e}")
                continue
                
    worker_logger.info(f"Worker {rank} finished successfully.")


def main():
    parser = argparse.ArgumentParser(description="Generate predictions (Multi-GPU Optimized)")
    parser.add_argument("--input_file", type=str, default="data/query_tag_map_top_mixsongid_rain_2025-11-17.txt")
    parser.add_argument("--output_file", type=str, default="outputs/offline_dict_optimized.txt")
    parser.add_argument("--model_path", type=str, default="models/generator/final_model/")
    
    # Data args
    parser.add_argument("--sample_size", type=int, default=None, help="Sample N queries (None=all)")
    
    # Inference args
    parser.add_argument("--strategy", type=str, default="sample", choices=["beam", "sample"])
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    
    # Parallelism args
    parser.add_argument("--num_gpus", type=int, default=2, help="Number of GPUs to use")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per GPU")
    
    args = parser.parse_args()
    
    setup_logging(log_file="logs/generate_predictions_main.log")
    logger.info("--- Starting Optimized Multi-GPU Inference ---")
    
    # 1. Load and Preprocess Queries
    logger.info(f"Loading queries from {args.input_file}...")
    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)
        
    queries = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split('\t')
            if len(parts) >= 1:
                queries.append(parts[0])
                
    # Deduplicate
    original_len = len(queries)
    queries = list(dict.fromkeys(queries))
    logger.info(f"Loaded {len(queries)} unique queries (deduplicated from {original_len}).")
    
    if args.sample_size and args.sample_size < len(queries):
        random.seed(args.seed)
        queries = random.sample(queries, args.sample_size)
        logger.info(f"Sampled {len(queries)} queries.")

    # 2. Distribute Workload
    available_gpus = torch.cuda.device_count()
    num_workers = min(args.num_gpus, available_gpus)
    
    if num_workers < 1:
        logger.warning("No GPU detected. Running in single-process CPU mode (very slow).")
        run_inference_worker(0, -1, queries, args, args.output_file)
        return

    logger.info(f"Launching {num_workers} worker processes on {num_workers} GPUs.")
    
    chunk_size = math.ceil(len(queries) / num_workers)
    processes = []
    temp_files = []
    
    for i in range(num_workers):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(queries))
        worker_queries = queries[start_idx:end_idx]
        
        if not worker_queries:
            continue
            
        part_file = f"{args.output_file}.part{i}"
        temp_files.append(part_file)
        
        p = mp.Process(
            target=run_inference_worker,
            args=(i, i, worker_queries, args, part_file)
        )
        p.start()
        processes.append(p)
    
    # 3. Wait for Completion
    for p in processes:
        p.join()
        if p.exitcode != 0:
            logger.error(f"Process {p.pid} failed with exit code {p.exitcode}.")
            sys.exit(1)
            
    # 4. Merge Results
    logger.info("All workers finished. Merging results...")
    with open(args.output_file, 'w', encoding='utf-8') as final_out:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                with open(temp_file, 'r', encoding='utf-8') as f:
                    import shutil
                    shutil.copyfileobj(f, final_out)
                os.remove(temp_file)
            else:
                logger.warning(f"Temporary file {temp_file} missing.")

    logger.info(f"Successfully generated predictions for {len(queries)} queries.")
    logger.info(f"Output saved to: {args.output_file}")

if __name__ == "__main__":
    main()
