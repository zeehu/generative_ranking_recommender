"""
Step G-1 (New): Preprocess raw playlist data.

This script streams a large, sorted playlist file, filters playlists by length,
and writes the output to a corpus file and a corresponding ID file.
"""
import os
import sys
import logging
import csv
from itertools import groupby
from tqdm import tqdm

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config
from src.common.utils import setup_logging

logger = logging.getLogger(__name__)

def preprocess_playlist_data(config: Config):
    logger.info("--- Starting Step G-1: Preprocessing Playlist Data (Streaming) ---")
    data_config = config.data
    w2v_config = config.word2vec

    input_file = data_config.playlist_songs_file
    output_corpus_file = w2v_config.corpus_file
    output_ids_file = w2v_config.corpus_ids_file

    min_len, max_len = 10, 300
    logger.info(f"Streaming from {input_file} and filtering playlists with length between {min_len} and {max_len}...")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_corpus_file), exist_ok=True)

    try:
        with open(input_file, 'r', encoding='utf-8') as fin, \
             open(output_corpus_file, 'w', encoding='utf-8') as f_corpus, \
             open(output_ids_file, 'w', encoding='utf-8') as f_ids:
            
            reader = csv.reader(fin, delimiter='\t')
            # Group by the first column (playlist_id) since the file is sorted
            groups = groupby(reader, key=lambda x: x[0])
            
            processed_count = 0
            for playlist_id, song_group in tqdm(groups, desc="Processing playlists"):
                # song_group is an iterator, convert to list
                song_list = [song[1] for song in song_group]
                
                # Apply length filter
                if min_len <= len(song_list) <= max_len:
                    f_corpus.write(' '.join(song_list) + '\n')
                    f_ids.write(str(playlist_id) + '\n')
                    processed_count += 1
        
        logger.info(f"Processing complete. Wrote {processed_count} playlists to output files.")

    except FileNotFoundError:
        logger.error(f"FATAL: Sorted playlist file not found at {input_file}")
        logger.error("Please ensure the file is sorted and the path in config.py is correct.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        sys.exit(1)

    logger.info(f"--- Step G-1 Completed Successfully. ---")

if __name__ == "__main__":
    config = Config()
    log_file_path = os.path.join(config.log_dir, "g-1_preprocess_playlists.log")
    setup_logging(log_file=log_file_path)
    logger = logging.getLogger(__name__)

    if "path/to/your" in config.data.playlist_songs_file:
        logger.error("FATAL: Please edit 'config.py' and set the path for 'playlist_songs_file'.")
        sys.exit(1)

    preprocess_playlist_data(config)