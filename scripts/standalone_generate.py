"""
Standalone High-Performance Generator for T5-based Recommender.
Removes dependencies on inference_t5.py.
Supports Multi-GPU, BF16, torch.compile, and Regex-based parsing.

Output Format:
Query 	 SemID||SongID||SongName||Singer 	 ...
"""
import os
import sys
import argparse
import logging
import random
import torch
import math
import json
import re
import csv
import shutil
from tqdm import tqdm
from typing import List, Dict, Tuple
from collections import defaultdict
from torch import multiprocessing as mp

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config
from src.generator.tiger_model import TIGERModel
from src.common.utils import setup_logging

# Ensure CUDA compatibility
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Clean text for TSV output."""
    if not text: return "Unknown"
    return str(text).replace("\n", " ").replace("\r", " ").replace("\t", " ").replace("||", " ").strip()

class StandaloneGenerator:
    """
    A self-contained generator class optimized for batch inference.
    """
    def __init__(self, config: Config, model_path: str, device_id: int):
        self.config = config
        self.device = torch.device(f'cuda:{device_id}')
        
        # 1. Load Model (Optimized)
        self._load_model(model_path)
        
        # 2. Load Data Mappings
        self.sem_id_to_songs = self._load_semantic_map()
        self.song_info = self._load_song_info()
        
        # 3. Regex for fast parsing
        # Matches <id_l1_123>, <id_l2_45>, etc.
        self.token_pattern = re.compile(r"<id_l[1-3]_(\d+)>")

    def _load_model(self, model_path: str):
        logger.info(f"Loading model from {model_path}...")
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        logger.info(f"Using dtype: {dtype}")
        
        try:
            self.model = TIGERModel.from_pretrained(model_path, torch_dtype=dtype).to(self.device)
            self.model.eval()
            
            # Compile for speedup (PyTorch 2.0+)
            if hasattr(torch, "compile"):
                try:
                    logger.info("Compiling model with mode='reduce-overhead'...")
                    self.model.model = torch.compile(self.model.model, mode="reduce-overhead")
                except Exception as e:
                    logger.warning(f"Compilation failed, proceeding without it: {e}")
                    
            self.tokenizer = self.model.tokenizer.base_tokenizer
        except Exception as e:
            logger.error(f"Critical error loading model: {e}")
            sys.exit(1)

    def _load_semantic_map(self) -> Dict[Tuple[int, ...], List[str]]:
        """Load semantic_ids -> song_ids."""
        path = self.config.data.semantic_ids_file
        mapping = defaultdict(list)
        if not os.path.exists(path):
            logger.error(f"Semantic ID file missing: {path}")
            return mapping
            
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    mapping[tuple(item['semantic_ids'])].append(item['song_id'])
                except: continue
        return mapping

    def _load_song_info(self) -> Dict[str, Dict[str, str]]:
        """Load song metadata."""
        path = getattr(self.config.data, 'song_info_file', 'data/gen_song_info.csv')
        info_map = {}
        if not os.path.exists(path):
            return info_map
            
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) >= 3:
                    info_map[row[0]] = {"name": row[1], "singer": row[2]}
        return info_map

    def parse_ids(self, text: str) -> List[Tuple[int, ...]]:
        """Extract semantic IDs from text using Regex."""
        # Extract all numbers from tokens like <id_l1_123>
        raw_ids = [int(x) for x in self.token_pattern.findall(text)]
        
        # Group into triplets (l1, l2, l3)
        # Assuming 3 levels. If levels differ, change '3' below.
        tuples = []
        for i in range(0, len(raw_ids), 3):
            chunk = raw_ids[i:i+3]
            if len(chunk) == 3:
                tuples.append(tuple(chunk))
        return tuples

    def generate_batch(self, queries: List[str], **gen_kwargs) -> List[List[str]]:
        """
        Run batch generation and return formatted result strings.
        """
        # Tokenize
        inputs = self.tokenizer(
            queries, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=self.config.generator_t5.max_input_length
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.model.generate(
                **inputs,
                **gen_kwargs
            )
        
        # Decode & Format
        # outputs shape: (batch_size * num_return_sequences, seq_len)
        # We need to regroup them by query
        
        num_return = gen_kwargs.get("num_return_sequences", 1)
        formatted_results = [] # List[List[str]]
        
        decoded_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
        
        for i in range(len(queries)):
            query_results = []
            seen_songs = set()
            
            # Get the slice of generations for this query
            start = i * num_return
            end = start + num_return
            texts = decoded_texts[start:end]
            
            for text in texts:
                sem_tuples = self.parse_ids(text)
                for sem_tuple in sem_tuples:
                    # Expand to songs
                    song_ids = self.sem_id_to_songs.get(sem_tuple, [])
                    for song_id in song_ids:
                        if song_id in seen_songs: continue
                        seen_songs.add(song_id)
                        
                        # Get Metadata
                        meta = self.song_info.get(song_id, {"name": "Unknown", "singer": "Unknown"})
                        s_name = clean_text(meta["name"])
                        s_singer = clean_text(meta["singer"])
                        
                        # Format: SemID||SongID||SongName||Singer
                        entry = f"{sem_tuple}||{song_id}||{s_name}||{s_singer}"
                        query_results.append(entry)
            
            formatted_results.append(query_results)
            
        return formatted_results

def worker_process(rank, gpu_id, queries, args, output_file):
    """Worker process logic."""
    # 1. Setup Env & Log
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log_file = f"logs/gen_worker_{rank}.log"
    os.makedirs("logs", exist_ok=True)
    
    logging.basicConfig(
        filename=log_file, level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(f"worker_{rank}")
    logger.info(f"Worker {rank} on GPU {gpu_id} started with {len(queries)} queries.")

    # 2. Seeds
    seed = args.seed + rank
    random.seed(seed)
    torch.manual_seed(seed)

    # 3. Init Generator (Model is loaded here, inside the process)
    config = Config()
    # Since CUDA_VISIBLE_DEVICES is set, we use cuda:0 inside this view
    generator = StandaloneGenerator(config, args.model_path, device_id=0)

    # 4. Inference
    batch_size = args.batch_size
    gen_kwargs = {
        "max_new_tokens": config.generator_t5.max_target_length,
        "do_sample": (args.strategy == "sample"),
        "num_beams": args.num_beams,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "num_return_sequences": args.num_return_sequences,
        "pad_token_id": generator.tokenizer.pad_token_id
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        for i in tqdm(range(0, len(queries), batch_size), desc=f"GPU {gpu_id}", position=rank):
            batch_q = queries[i : i + batch_size]
            try:
                results = generator.generate_batch(batch_q, **gen_kwargs)
                
                for q, res_list in zip(batch_q, results):
                    if res_list:
                        clean_q = clean_text(q)
                        line = f"{clean_q}\t" + "\t".join(res_list) + "\n"
                        f.write(line)
            except Exception as e:
                logger.error(f"Batch error: {e}")
                continue

    logger.info("Worker finished.")

def main():
    parser = argparse.ArgumentParser(description="Standalone Multi-GPU T5 Inference")
    parser.add_argument("--input_file", type=str, default="data/query_tag_map_top_mixsongid_rain_2025-11-17.txt")
    parser.add_argument("--output_file", type=str, default="outputs/predictions_standalone.txt")
    parser.add_argument("--model_path", type=str, default="models/generator/final_model/")
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=128)
    
    # Decoding
    parser.add_argument("--strategy", type=str, default="sample", choices=["sample", "beam"])
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num_return_sequences", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_size", type=int, default=None)

    args = parser.parse_args()
    
    setup_logging(log_file="logs/generate_main.log")
    logger.info("--- Starting Standalone Multi-GPU Inference ---")

    # Load Data
    logger.info("Loading queries...")
    if not os.path.exists(args.input_file):
        logger.error("Input file not found.")
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
    queries = list(dict.fromkeys(queries))
    logger.info(f"Loaded {len(queries)} unique queries.")
    
    if args.sample_size:
        random.seed(args.seed)
        queries = random.sample(queries, args.sample_size)

    # Prepare Workers
    num_workers = min(args.num_gpus, torch.cuda.device_count())
    chunk_size = math.ceil(len(queries) / num_workers)
    processes = []
    temp_files = []

    logger.info(f"Launching {num_workers} workers (Batch size: {args.batch_size})...")

    for i in range(num_workers):
        start = i * chunk_size
        end = min((i+1) * chunk_size, len(queries))
        worker_q = queries[start:end]
        if not worker_q: continue
        
        tmp = f"{args.output_file}.part{i}"
        temp_files.append(tmp)
        
        p = mp.Process(target=worker_process, args=(i, i, worker_q, args, tmp))
        p.start()
        processes.append(p)

    # Wait
    for p in processes:
        p.join()
        if p.exitcode != 0:
            logger.error("A worker failed.")
            sys.exit(1)

    # Merge
    logger.info("Merging results...")
    with open(args.output_file, 'w', encoding='utf-8') as out:
        for tmp in temp_files:
            if os.path.exists(tmp):
                with open(tmp, 'r', encoding='utf-8') as f:
                    shutil.copyfileobj(f, out)
                os.remove(tmp)

    logger.info(f"Done! Results at {args.output_file}")

if __name__ == "__main__":
    main()
