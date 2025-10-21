
Step G-1 (New): Preprocess raw playlist data.

This script filters playlists based on length (10 <= length <= 300)
and converts the data into a clean corpus file where each line represents
a full playlist (space-separated song IDs). This format is optimal for
gensim's `corpus_file` training method.

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

    # Define input and output paths
    input_file = data_config.playlist_songs_file
    output_file = config.word2vec.corpus_file # New path in config

    # 1. Load the entire dataset into memory
    # This is necessary for accurate filtering by playlist length.
    try:
        logger.info(f"Loading raw playlist data from {input_file}...")
        df = pd.read_csv(input_file, sep='\t', header=None, names=['playlist_id', 'song_id'], dtype=str)
    except FileNotFoundError:
        logger.error(f"FATAL: Raw playlist file not found at {input_file}")
        sys.exit(1)

    # 2. Group songs by playlist
    logger.info("Grouping songs by playlist...")
    playlists = df.groupby('playlist_id')['song_id'].apply(list)
    logger.info(f"Found {len(playlists)} unique playlists.")

    # 3. Filter playlists by length
    min_len, max_len = 10, 300
    logger.info(f"Filtering playlists to have between {min_len} and {max_len} songs...")
    filtered_playlists = playlists[playlists.str.len().between(min_len, max_len)]
    logger.info(f"{len(filtered_playlists)} playlists remain after filtering.")

    # 4. Write to the output corpus file
    logger.info(f"Saving formatted corpus to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for playlist in tqdm(filtered_playlists, desc="Writing corpus"):
            f.write(' '.join(playlist) + '\n')

    logger.info(f"--- Step G-1 Completed Successfully. Corpus is ready at {output_file} ---")

if __name__ == "__main__":
    config = Config()
    log_file_path = os.path.join(config.log_dir, "g-1_preprocess_playlists.log")
    setup_logging(log_file=log_file_path)
    logger = logging.getLogger(__name__)

    if "path/to/your" in config.data.playlist_songs_file:
        logger.error("FATAL: Please edit 'config.py' and set the path for 'playlist_songs_file'.")
        sys.exit(1)

    preprocess_playlist_data(config)
