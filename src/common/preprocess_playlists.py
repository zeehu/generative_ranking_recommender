"""
Step G-1 (New): Preprocess raw playlist data.

This script filters playlists based on length (10 <= length <= 300)
and converts the data into a clean corpus file where each line represents
a full playlist (space-separated song IDs). It also saves the corresponding
playlist IDs to a separate file.
"""
import os
import sys
import pandas as pd
import logging
from tqdm import tqdm

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config
from src.common.utils import setup_logging

logger = logging.getLogger(__name__)

def preprocess_playlist_data(config: Config):
    logger.info("--- Starting Step G-1: Preprocessing Playlist Data ---")
    data_config = config.data
    w2v_config = config.word2vec

    input_file = data_config.playlist_songs_file
    output_corpus_file = w2v_config.corpus_file
    output_ids_file = w2v_config.corpus_ids_file

    try:
        logger.info(f"Loading raw playlist data from {input_file}...")
        df = pd.read_csv(input_file, sep='\t', header=None, names=['playlist_id', 'song_id'], dtype=str)
    except FileNotFoundError:
        logger.error(f"FATAL: Raw playlist file not found at {input_file}")
        sys.exit(1)

    logger.info("Grouping songs by playlist...")
    playlists = df.groupby('playlist_id')['song_id'].apply(list)
    logger.info(f"Found {len(playlists)} unique playlists.")

    min_len, max_len = 10, 300
    logger.info(f"Filtering playlists to have between {min_len} and {max_len} songs...")
    filtered_playlists = playlists[playlists.str.len().between(min_len, max_len)]
    logger.info(f"{len(filtered_playlists)} playlists remain after filtering.")

    logger.info(f"Saving formatted corpus to {output_corpus_file} and IDs to {output_ids_file}...")
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_corpus_file), exist_ok=True)
    
    with open(output_corpus_file, 'w', encoding='utf-8') as f_corpus, \
         open(output_ids_file, 'w', encoding='utf-8') as f_ids:
        
        for playlist_id, song_list in tqdm(filtered_playlists.items(), desc="Writing corpus and IDs"):
            f_corpus.write(' '.join(song_list) + '\n')
            f_ids.write(str(playlist_id) + '\n')

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
