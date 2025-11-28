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
                # Fast parsing of tuple string "(1, 2, 3)" -> tuple(1, 2, 3)
                # Remove parentheses and split
                sem_str = fields[0].strip().strip("()")
                if not sem_str:
                    continue
                sem_id = tuple(int(x.strip()) for x in sem_str.split(','))
            except:
                continue # Invalid SemID format
            
            song_id = fields[1] if len(fields) > 1 else ""
            song_name = fields[2] if len(fields) > 2 else ""
            singer = fields[3] if len(fields) > 3 else ""
            freq = int(fields[4]) if len(fields) > 4 and fields[4].isdigit() else 1
            
            # Filter out invalid entries (missing song info)
            if not song_id or not song_name or not singer:
                continue

            results.append({
                "sem_id": sem_id,
                "song_id": song_id,
                "song_name": song_name,
                "singer": singer,
                "freq": freq,
                "raw_str": item_str
            })
            
    return query, results

def load_voting_results(file_path: str, sem_map: Dict[str, Tuple[int, ...]]) -> Tuple[Dict[str, Tuple[int, ...]], Set[Tuple[str, str]]]:
    """
    Loads voting results with format: query \t song_id:vote,song_id:vote...
    Maps the highest-voted song to its semantic ID.
    Returns:
        voting_data: Dict[query, best_semantic_id]
        known_pairs: Set[(query, song_id)] - all pairs that already have votes
    """
    logger.info(f"Loading voting results from {file_path}...")
    voting_data = {}
    known_pairs = set()
    
    if not os.path.exists(file_path):
        logger.warning(f"Voting file not found: {file_path}")
        return voting_data, known_pairs

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
                    # Use rsplit to safely handle song_ids that might rarely contain ':' (though unlikely in this schema)
                    song_id, vote_count = pair.rsplit(':', 1)
                    vote_count = int(vote_count)
                    song_id = song_id.strip()
                    
                    known_pairs.add((query, song_id))
                    
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
    return voting_data, known_pairs

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
    parser.add_argument("--auto_vote_file", type=str, default=None, help="Optional: Path to save auto-generated vote file")
    
    args = parser.parse_args()
    ensure_dir(args.output_dir)
    if args.auto_vote_file:
        ensure_dir(os.path.dirname(args.auto_vote_file))

    # 1. Load Semantic Map
    sem_map = load_semantic_map(args.sem_file)

    # 2. Load Voting Data
    voting_map, known_voted_pairs = load_voting_results(args.vote_file, sem_map)

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
    
    # Auto-Vote Buffer
    auto_vote_lines = []
    
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

            # Prepare Voting Target Data (once per query)
            target_sem_id = voting_map.get(query)
            target_sem_id_l2 = None
            target_sem_id_l1 = None
            
            if target_sem_id:
                if len(target_sem_id) >= 2:
                    target_sem_id_l2 = target_sem_id[:2]
                if len(target_sem_id) >= 1:
                    target_sem_id_l1 = target_sem_id[:1]

            # State for this query
            best_match_level = 0  # 0: Mismatch, 1: L1, 2: L2, 3: L3
            valid_singers = set()
            has_valid_song = False

            # --- Single Pass Loop over Results ---
            for res in parsed_results:
                # Unpack
                gen_id = res['sem_id']
                song_id = res['song_id']
                song_name = res['song_name']
                singer = res['singer']
                freq = res['freq']

                # 1. Collect Singer Info (for hallucination check)
                if singer and singer != "Unknown":
                    valid_singers.add(singer)
                    has_valid_song = True

                # 2. Calculate Match Level (Reused for AutoVote & Evaluation)
                curr_match_level = 0
                if target_sem_id and gen_id:
                    if gen_id == target_sem_id:
                        curr_match_level = 3
                    elif target_sem_id_l2 and len(gen_id) >= 2 and gen_id[:2] == target_sem_id_l2:
                        curr_match_level = 2
                    elif target_sem_id_l1 and len(gen_id) >= 1 and gen_id[:1] == target_sem_id_l1:
                        curr_match_level = 1
                
                # Update global best match for this query
                if curr_match_level > best_match_level:
                    best_match_level = curr_match_level

                # 3. Auto Vote Logic
                if args.auto_vote_file:
                    if (query, song_id) not in known_voted_pairs:
                        vote_val = None
                        # Logic: best_match_level >= 2
                        if curr_match_level >= 2:
                            if freq >= 4: vote_val = 0
                            elif freq >= 2: vote_val = -1
                            else: vote_val = -2
                        else:
                            # Other data (Match < 2 or No GT)
                            if freq >= 4: vote_val = 0
                            elif freq == 3: vote_val = -1
                            elif freq == 2: vote_val = -2
                            # freq < 2 -> discard
                        
                        if vote_val is not None:
                            av_line = f"{query}\t{song_id}||{vote_val}||{song_name}||{singer}||{freq}"
                            auto_vote_lines.append(av_line)

            # --- Post-Loop Analysis ---

            # A. Check Singer Hallucination / Bias
            if has_valid_song and len(valid_singers) == 1:
                unique_singer = list(valid_singers)[0]
                if unique_singer.lower() not in query.lower():
                    singer_hallucinations.append(clean_line)
                    # Note: We don't stop here, we continue to categorize by match level
            
            # B. Semantic ID Matching Categorization
            if query in voting_map:
                vote_str_formatted = f"{query}\t{target_sem_id}"
                
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

    if args.auto_vote_file and auto_vote_lines:
        logger.info(f"Saving {len(auto_vote_lines)} auto-generated votes to {args.auto_vote_file}...")
        save_lines(args.auto_vote_file, auto_vote_lines)

    logger.info(f"Done! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
