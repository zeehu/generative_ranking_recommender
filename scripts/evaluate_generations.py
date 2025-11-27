"""
Script to evaluate generation results against voting (ground truth) results.
Performs hierarchical semantic ID matching and detects singer hallucination/bias.

Usage:
    python scripts/evaluate_generations.py \
        --pred_file outputs/predictions_standalone.txt \
        --vote_file data/voting_results.txt \
        --output_dir outputs/evaluation_results
"""
import os
import sys
import argparse
import logging
import ast
from collections import defaultdict
from typing import List, Dict, Tuple, Set

import json

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_semantic_map(file_path: str) -> Dict[str, Tuple[int, ...]]:
    """
    Loads song_id -> semantic_ids mapping from JSONL file.
    """
    logger.info(f"Loading semantic map from {file_path}...")
    sem_map = {}
    if not os.path.exists(file_path):
        logger.error(f"Semantic map file not found: {file_path}")
        sys.exit(1)
        
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                song_id = str(data['song_id'])
                sem_ids = tuple(data['semantic_ids'])
                sem_map[song_id] = sem_ids
            except:
                continue
    logger.info(f"Loaded {len(sem_map)} song semantic mappings.")
    return sem_map

def parse_prediction_line(line: str) -> Tuple[str, List[dict]]:
    """
    Parses a line from standalone_generate.py output.
    Format: Query 	 SemID||SongID||SongName||Singer||Freq 	 ...
    """
    parts = line.strip().split('\t')
    if len(parts) < 2:
        return None, []
    
    query = parts[0]
    results = []
    
    for item_str in parts[1:]:
        # Skip raw text logs if present
        if item_str.startswith("EMPTY_RAW:") or item_str == "EMPTY":
            continue
            
        # Format: SemID||SongID||SongName||Singer||Freq
        # SemID is like "(1, 2, 3)"
        fields = item_str.split('||')
        
        # Handle phantom entries (empty song info)
        # Phantom: (1,2,3)||||||||5 or similar
        if len(fields) >= 1:
            try:
                sem_id = ast.literal_eval(fields[0])
            except:
                continue # Invalid SemID format
            
            song_id = fields[1] if len(fields) > 1 else ""
            song_name = fields[2] if len(fields) > 2 else ""
            singer = fields[3] if len(fields) > 3 else ""
            freq = int(fields[4]) if len(fields) > 4 and fields[4].isdigit() else 1
            
            results.append({
                "sem_id": sem_id,
                "song_id": song_id,
                "song_name": song_name,
                "singer": singer,
                "freq": freq,
                "raw_str": item_str
            })
            
    return query, results

def load_voting_results(file_path: str, sem_map: Dict[str, Tuple[int, ...]]) -> Dict[str, Tuple[int, ...]]:
    """
    Loads voting results with format: query \t song_id:vote,song_id:vote...
    Maps the highest-voted song to its semantic ID.
    """
    logger.info(f"Loading voting results from {file_path}...")
    voting_data = {}
    
    if not os.path.exists(file_path):
        logger.warning(f"Voting file not found: {file_path}")
        return voting_data

    valid_queries = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            parts = line.split('\t')
            if len(parts) < 2: continue
            
            query = parts[0]
            vote_str = parts[1] # "song1:5,song2:3"
            
            best_sem_id = None
            max_vote = -1
            
            # Parse song:vote pairs
            pairs = vote_str.split(',')
            for pair in pairs:
                if ':' not in pair: continue
                try:
                    song_id, vote_count = pair.rsplit(':', 1)
                    vote_count = int(vote_count)
                    song_id = song_id.strip()
                    
                    # Check if we have a semantic ID for this song
                    if song_id in sem_map:
                        if vote_count > max_vote:
                            max_vote = vote_count
                            best_sem_id = sem_map[song_id]
                except:
                    continue
            
            if best_sem_id:
                voting_data[query] = best_sem_id
                valid_queries += 1
                
    logger.info(f"Loaded {valid_queries} valid voting entries (mapped to Semantic IDs).")
    return voting_data

def save_lines(filepath, lines):
    with open(filepath, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')

def main():
    parser = argparse.ArgumentParser(description="Evaluate Generation Results")
    parser.add_argument("--pred_file", type=str, required=True, help="Path to prediction file")
    parser.add_argument("--vote_file", type=str, required=True, help="Path to voting results file")
    parser.add_argument("--sem_file", type=str, default="outputs/semantic_id/song_semantic_ids.jsonl", help="Path to song semantic IDs JSONL")
    parser.add_argument("--output_dir", type=str, default="outputs/eval_results", help="Directory to save split files")
    
    args = parser.parse_args()
    ensure_dir(args.output_dir)

    # 1. Load Semantic Map
    sem_map = load_semantic_map(args.sem_file)

    # 2. Load Voting Data
    voting_map = load_voting_results(args.vote_file, sem_map)

    # 2. Process Predictions
    logger.info(f"Processing predictions from {args.pred_file}...")
    
    # Get total lines for tqdm
    total_lines = sum(1 for _ in open(args.pred_file, 'r', encoding='utf-8'))
    
    # Buckets
    singer_hallucinations = [] # (Query, Line)
    
    match_l3_gen = []
    match_l3_vote = []
    
    match_l2_gen = []
    match_l2_vote = []
    
    match_l1_gen = []
    match_l1_vote = []
    
    mismatch_gen = []
    mismatch_vote = []
    
    remainder_gen = []
    
    processed_count = 0
    total_gen_songs = 0 # For stats
    
    from tqdm import tqdm
    
    with open(args.pred_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Evaluating", unit="q"):
            processed_count += 1
            clean_line = line.strip()
            query, parsed_results = parse_prediction_line(clean_line)
            
            if not query:
                continue
            
            # Stats: Count generated songs
            total_gen_songs += len(parsed_results)

            # --- Check 1: Singer Hallucination / Bias ---
            # Logic: All generated songs have the same singer, AND singer name not in query.
            # Only consider cases where we have at least some valid song info (not empty phantom entries)
            valid_singers = set()
            has_valid_song = False
            
            for res in parsed_results:
                s = res['singer']
                if s and s != "Unknown":
                    valid_singers.add(s)
                    has_valid_song = True
            
            is_hallucination = False
            if has_valid_song and len(valid_singers) == 1:
                unique_singer = list(valid_singers)[0]
                # Case-insensitive check
                if unique_singer.lower() not in query.lower():
                    singer_hallucinations.append(clean_line)
                    is_hallucination = True
            
            # Note: We continue to categorize even if it is a hallucination, 
            # or you can choose to `continue` here if you want them exclusive.
            # Assuming we want to categorize everything by ID match primarily.

            # --- Check 2: Semantic ID Matching ---
            if query in voting_map:
                vote_id = voting_map[query]
                vote_str_formatted = f"{query}\t{vote_id}"
                
                best_match_level = 0 # 0: Mismatch, 1: L1, 2: L2, 3: L3
                
                if parsed_results:
                    # Iterate through ALL generated results, not just Top-1
                    for res in parsed_results:
                        gen_id = res['sem_id']
                        
                        current_level = 0
                        if gen_id == vote_id:
                            current_level = 3
                        elif gen_id[:2] == vote_id[:2]:
                            current_level = 2
                        elif gen_id[:1] == vote_id[:1]:
                            current_level = 1
                        
                        # Keep the best match found so far
                        if current_level > best_match_level:
                            best_match_level = current_level
                            # Optimization: If L3 match found, we can stop looking
                            if best_match_level == 3:
                                break
                
                # Assign to bucket based on BEST match found across all candidates
                if best_match_level == 3:
                    match_l3_gen.append(clean_line)
                    match_l3_vote.append(vote_str_formatted)
                elif best_match_level == 2:
                    match_l2_gen.append(clean_line)
                    match_l2_vote.append(vote_str_formatted)
                elif best_match_level == 1:
                    match_l1_gen.append(clean_line)
                    match_l1_vote.append(vote_str_formatted)
                else:
                    mismatch_gen.append(clean_line)
                    mismatch_vote.append(vote_str_formatted)
            else:
                # No voting data for this query
                remainder_gen.append(clean_line)

    # 3. Print Statistics
    avg_songs = total_gen_songs / processed_count if processed_count > 0 else 0
    total_with_votes = len(match_l3_gen) + len(match_l2_gen) + len(match_l1_gen) + len(mismatch_gen)
    
    logger.info("="*60)
    logger.info("EVALUATION STATISTICS")
    logger.info("="*60)
    logger.info(f"Total Queries Processed:   {processed_count}")
    logger.info(f"Avg Generated Songs/Query: {avg_songs:.2f}")
    logger.info(f"Singer Hallucination Bias: {len(singer_hallucinations)} ({len(singer_hallucinations)/processed_count*100:.1f}%)")
    logger.info("-" * 60)
    logger.info(f"Queries with Voting Data:  {total_with_votes}")
    if total_with_votes > 0:
        logger.info(f"  L3 Match (Perfect):      {len(match_l3_gen):<5} ({len(match_l3_gen)/total_with_votes*100:.1f}%)")
        logger.info(f"  L2 Match (2 layers):     {len(match_l2_gen):<5} ({len(match_l2_gen)/total_with_votes*100:.1f}%)")
        logger.info(f"  L1 Match (1 layer):      {len(match_l1_gen):<5} ({len(match_l1_gen)/total_with_votes*100:.1f}%)")
        logger.info(f"  Mismatch:                {len(mismatch_gen):<5} ({len(mismatch_gen)/total_with_votes*100:.1f}%)")
    logger.info("-" * 60)
    logger.info(f"Remainder (No Voting Data): {len(remainder_gen)}")
    logger.info("="*60)

    # 4. Save Results
    logger.info(f"Processed {processed_count} queries.")
    logger.info("Saving results...")

    def save_pair(name, gen_data, vote_data=None):
        path_gen = os.path.join(args.output_dir, f"gen_{name}.txt")
        save_lines(path_gen, gen_data)
        logger.info(f"  {name}: {len(gen_data)} entries")
        
        if vote_data is not None:
            path_vote = os.path.join(args.output_dir, f"vote_{name}.txt")
            save_lines(path_vote, vote_data)

    save_lines(os.path.join(args.output_dir, "gen_singer_bias.txt"), singer_hallucinations)
    logger.info(f"  Singer Bias/Hallucination: {len(singer_hallucinations)} entries")

    save_pair("match_l3", match_l3_gen, match_l3_vote)
    save_pair("match_l2", match_l2_gen, match_l2_vote)
    save_pair("match_l1", match_l1_gen, match_l1_vote)
    save_pair("mismatch", mismatch_gen, mismatch_vote)
    save_pair("remainder", remainder_gen) # No voting data for these

    logger.info(f"Done! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
