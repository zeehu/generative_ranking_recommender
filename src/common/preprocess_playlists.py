
"""
Step G-1 (New): Preprocess raw playlist data.

This script streams a large, sorted playlist file, filters playlists by length,
and converts the data into a clean corpus file and a corresponding ID file.
This version uses a manual for-loop for more explicit control and logging.
"""
import os
import sys
import logging
import csv
from tqdm import tqdm

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config
from src.common.utils import setup_logging

logger = logging.getLogger(__name__)

def preprocess_playlist_data(config: Config):
    logger.info("--- Starting Step G-1: Preprocessing Playlist Data (Manual Loop) ---")
    data_config = config.data
    w2v_config = config.word2vec

    input_file = data_config.playlist_songs_file
    output_corpus_file = w2v_config.corpus_file
    output_ids_file = w2v_config.corpus_ids_file

    min_len, max_len = 10, 300
    logger.info(f"Streaming from {input_file} and filtering playlists with length between {min_len} and {max_len}...")

    try:
        # First, get total line count for a precise progress bar
        logger.info("Counting total lines in source file for progress bar...")
        with open(input_file, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for line in f)
        logger.info(f"Found {total_lines} total lines.")

        os.makedirs(os.path.dirname(output_corpus_file), exist_ok=True)

        with open(input_file, 'r', encoding='utf-8') as fin, \
             open(output_corpus_file, 'w', encoding='utf-8') as f_corpus, \
             open(output_ids_file, 'w', encoding='utf-8') as f_ids:
            
            reader = csv.reader(fin, delimiter='\t')
            
            current_playlist_id = None
            current_songs = []
            processed_count = 0

            for i, row in enumerate(tqdm(reader, total=total_lines, desc="Processing lines")):
                if len(row) < 2:
                    logger.warning(f"Skipping malformed row {i+1}: {row}")
                    continue
                
                playlist_id, song_id = row[0], row[1]

                if playlist_id != current_playlist_id:
                    # A new playlist has started, process the previous one
                    if current_playlist_id is not None and min_len <= len(current_songs) <= max_len:
                        f_corpus.write(' '.join(current_songs) + '\n')
                        f_ids.write(str(current_playlist_id) + '\n')
                        processed_count += 1
                    
                    # Start the new playlist
                    current_playlist_id = playlist_id
                    current_songs = [song_id]
                else:
                    # Continue adding to the current playlist
                    current_songs.append(song_id)
            
            # Process the very last playlist in the file
            if current_playlist_id is not None and min_len <= len(current_songs) <= max_len:
                f_corpus.write(' '.join(current_songs) + '\n')
                f_ids.write(str(current_playlist_id) + '\n')
                processed_count += 1

        logger.info(f"Processing complete. Wrote {processed_count} playlists to output files.")

    except FileNotFoundError:
        logger.error(f"FATAL: Sorted playlist file not found at {input_file}")
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
