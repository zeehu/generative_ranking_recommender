
"""
Debug script to detect and report semantic ID collisions.

This script reads the project-standard `song_semantic_ids.jsonl` file.
"""
import os
import json
from collections import defaultdict
from tqdm import tqdm

def find_collisions():
    """Finds and reports on semantic ID collisions from a JSONL file."""
    print("--- Starting Semantic ID Collision Analysis ---")
    
    # Path to the project-standard semantic ID file
    semantic_ids_file = os.path.join("outputs", "semantic_id", "song_semantic_ids.jsonl")
    
    if not os.path.exists(semantic_ids_file):
        print(f"\nFATAL: Semantic ID file not found at '{semantic_ids_file}'")
        print("Please run the simplified_semantic_id_generator.py script first to generate the IDs.")
        return

    print(f"Reading semantic IDs from {semantic_ids_file}...")
    
    # A dictionary mapping a semantic ID tuple to a list of song IDs
    reverse_map = defaultdict(list)

    with open(semantic_ids_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Building reverse map"):
            try:
                item = json.loads(line)
                # Use a tuple of the semantic IDs as the dictionary key
                key = tuple(item['semantic_ids'])
                reverse_map[key].append(item['song_id'])
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not parse line. Error: {e}. Line: '{line.strip()}'")
                continue

    print("Analyzing for collisions...")
    
    collisions = []
    for semantic_id_tuple, song_id_list in reverse_map.items():
        if len(song_id_list) > 1:
            collisions.append((semantic_id_tuple, song_id_list))

    # --- Print Report ---
    print("\n" + "="*80)
    print("  Semantic ID Collision Report")
    print("="*80)

    if not collisions:
        print("\n🎉 Congratulations! No collisions found.")
        print("This means every song has a unique semantic ID.")
    else:
        # Sort collisions by severity (most collided songs first)
        collisions.sort(key=lambda item: len(item[1]), reverse=True)

        total_songs_affected = sum(len(songs) for _, songs in collisions)
        worst_offender = collisions[0]  # The first one is the worst after sorting
        max_collision_size = len(worst_offender[1])

        print(f"\nFound {len(collisions)} unique semantic IDs that have collisions.")
        print(f"A total of {total_songs_affected} songs are involved in these collisions.")
        print("\n" + "-"*40)
        print("Worst Collision Case:")
        print(f"  A single Semantic ID {worst_offender[0]} was assigned to {max_collision_size} different songs.")
        print("-"*40)

        print("\n--- Example Collisions (showing top 5 most severe) ---")
        
        for i, (semantic_id, songs) in enumerate(collisions[:5]):
            print(f"\n{i+1}. Semantic ID {semantic_id} is shared by {len(songs)} songs:")
            # Show a subset of songs if the list is too long
            if len(songs) > 10:
                print(f"   {songs[:10]}... and {len(songs) - 10} more.")
            else:
                print(f"   {songs}")
            
    print("\n" + "="*80)

if __name__ == "__main__":
    find_collisions()
