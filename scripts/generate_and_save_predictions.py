"""
Generate predictions using T5 model and save detailed song information.
"""
import os
import sys
import argparse
import logging
import random
import torch
from tqdm import tqdm

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config
from src.generator.inference_t5 import PlaylistGenerator
from src.common.utils import setup_logging

logger = logging.getLogger(__name__)

def escape_text(text: str) -> str:
    """Escape special characters for output format."""
    if not text:
        return ""
    return text.replace("\\", "\\\\").replace("\t", "\\t").replace(":", "\\:").replace(",", "\\,")

def main():
    parser = argparse.ArgumentParser(description="Generate predictions and save details")
    parser.add_argument(
        "--input_file", 
        type=str, 
        default="data/query_tag_map_top_mixsongid_rain_2025-11-17.txt",
        help="Input file containing queries"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="outputs/predictions_with_details_beam.txt",
        help="Output file path"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=“models/generator/final_model/”,
        help="Model path"
    )
    parser.add_argument(
        "--sample_size", 
        type=int, 
        default=1000,
        help="Number of queries to sample (default: all)"
    )
    parser.add_argument(
        "--strategy", 
        type=str, 
        default="beam",
        choices=["beam", "sample"],
        help="Decoding strategy: 'beam' or 'sample'"
    )
    parser.add_argument("--num_beams", type=int, default=4, help="Number of beams for beam search")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k for sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p for sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_file="logs/generate_predictions.log")
    
    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load config
    config = Config()
    
    # Initialize generator
    logger.info("Initializing generator...")
    generator = PlaylistGenerator(config, model_path=args.model_path, use_trie_constraint=True)
    
    # Load queries
    logger.info(f"Loading queries from {args.input_file}...")
    queries = []
    if os.path.exists(args.input_file):
        with open(args.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 1:
                    queries.append(parts[0])
    else:
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)
        
    logger.info(f"Loaded {len(queries)} queries.")
    
    # Sample if needed
    if args.sample_size and args.sample_size < len(queries):
        logger.info(f"Sampling {args.sample_size} queries...")
        queries = random.sample(queries, args.sample_size)
    
    # Open output file
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Batch processing
    batch_size = 32
    logger.info(f"Starting batch inference with batch_size={batch_size}...")
    logger.info(f"Strategy: {args.strategy}")
    if args.strategy == "beam":
        logger.info(f"  - num_beams: {args.num_beams}")
    else:
        logger.info(f"  - temperature: {args.temperature}")
        logger.info(f"  - top_k: {args.top_k}")
        logger.info(f"  - top_p: {args.top_p}")
    
    with open(args.output_file, 'w', encoding='utf-8') as out_f:
        for i in tqdm(range(0, len(queries), batch_size), desc="Processing batches"):
            batch_queries = queries[i:i+batch_size]
            try:
                # Generate batch
                do_sample = (args.strategy == "sample")
                batch_results = generator.generate_batch(
                    batch_queries, 
                    max_songs=50, 
                    do_sample=do_sample, 
                    num_beams=args.num_beams,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p
                )
                
                for query, results in zip(batch_queries, batch_results):
                    output_parts = []
                    for item in results:
                        semantic_id_tuple = item['semantic_id']
                        
                        # Get all songs for this semantic ID
                        if semantic_id_tuple in generator.semantic_to_song_cluster:
                            song_ids = generator.semantic_to_song_cluster[semantic_id_tuple]
                            
                            # Format each song
                            for song_id in song_ids:
                                song_info = generator.song_info_map.get(song_id, {"name": "Unknown", "singer": "Unknown"})
                                song_name = escape_text(song_info.get("name", "Unknown"))
                                singer = escape_text(song_info.get("singer", "Unknown"))
                                
                                # Format: semantic_id:song_id:song_name:singer
                                sem_id_str = str(semantic_id_tuple)
                                output_parts.append(f"{sem_id_str}:{song_id}:{song_name}:{singer}")
                    
                    # Join with comma
                    output_line = f"{escape_text(query)}\t{','.join(output_parts)}\n"
                    out_f.write(output_line)
                    
            except Exception as e:
                logger.error(f"Error processing batch starting at index {i}: {e}")
                continue
                
    logger.info(f"Done. Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
