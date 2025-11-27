"""
Script to compare Decoding Strategies (Sampling vs Beam Search) for TIGER Model.
Usage:
    python scripts/compare_decoding_strategies.py --model_path models/generator/final_model/
"""
import os
import sys
import argparse
import logging
import torch
from tabulate import tabulate

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config
from scripts.standalone_generate import StandaloneGenerator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/generator/final_model/")
    parser.add_argument("--input_queries", nargs="+", default=["周杰伦", "悲伤的歌", "摇滚", "Coding Music"])
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        logger.error(f"Model path not found: {args.model_path}")
        return

    config = Config()
    
    # Force 1 GPU for comparison to avoid multi-process complexity in this script
    # We can use StandaloneGenerator but we need to be careful with its __init__
    # StandaloneGenerator loads model on init.
    
    try:
        logger.info(f"Loading model from {args.model_path}...")
        # device_id=0 assumes we have a GPU. 
        generator = StandaloneGenerator(config, args.model_path, device_id=0)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    queries = args.input_queries
    results = {}

    # 1. Run Sampling
    logger.info("Running Sampling Strategy...")
    sample_kwargs = {
        "max_new_tokens": 128, # shorter for test
        "do_sample": True,
        "num_beams": 1,
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 0.9,
        "num_return_sequences": 1,
        "pad_token_id": generator.tokenizer.pad_token_id
    }
    res_sample = generator.generate_batch(queries, **sample_kwargs)
    # StandaloneGenerator returns List[List[str]], where inner list is num_return_sequences
    # We take the first one
    results["Sampling"] = [r[0] if r else "N/A" for r in res_sample]

    # 2. Run Beam Search
    logger.info("Running Beam Search Strategy...")
    beam_kwargs = {
        "max_new_tokens": 128,
        "do_sample": False,
        "num_beams": 5,
        "num_return_sequences": 1,
        "pad_token_id": generator.tokenizer.pad_token_id
    }
    res_beam = generator.generate_batch(queries, **beam_kwargs)
    results["Beam Search"] = [r[0] if r else "N/A" for r in res_beam]

    # 3. Print Comparison
    table_data = []
    for i, q in enumerate(queries):
        # Simplify output for display: just show SongName
        # Format is SemID||SongID||SongName||Singer
        def parse_name(raw):
            if "||" in raw:
                parts = raw.split("||")
                if len(parts) >= 3:
                    return f"{parts[2]} ({parts[3]})"
            return raw

        s_out = parse_name(results["Sampling"][i])
        b_out = parse_name(results["Beam Search"][i])
        
        table_data.append([q, s_out, b_out])

    print("\n" + "="*60)
    print("Decoding Strategy Comparison")
    print("="*60)
    print(tabulate(table_data, headers=["Query", "Sampling (T=0.8, k=50)", "Beam Search (B=5)"], tablefmt="grid"))

if __name__ == "__main__":
    main()
