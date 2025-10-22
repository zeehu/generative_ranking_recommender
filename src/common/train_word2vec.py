"""
Step G0: Train Song Vectors using Word2Vec.

This script uses a memory-efficient streaming iterator to process a large,
sorted playlist file and trains a gensim Word2Vec model.
"""
import os
import sys
import pandas as pd
import logging
import time
import csv
from gensim.models import Word2Vec
from itertools import groupby

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config
from src.common.utils import setup_logging

logger = logging.getLogger(__name__)

class PlaylistCorpus:
    """An iterator that yields playlists (sentences) from a large, sorted CSV file."""
    def __init__(self, filepath):
        self.filepath = filepath

    def __iter__(self):
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                for playlist_id, song_group in groupby(reader, key=lambda x: x[0] if x else None):
                    if playlist_id is None:
                        continue
                    yield [song[1] for song in song_group if len(song) > 1]
        except FileNotFoundError:
            logger.error(f"FATAL: Sorted playlist file not found at {self.filepath}")
            raise

def train_song_vectors(config: Config):
    logger.info("--- Starting Step G0: Train Song Vectors with Word2Vec ---")
    data_config = config.data
    w2v_config = config.word2vec

    # 1. Initialize a corpus iterator from the sorted playlist file
    sentences = PlaylistCorpus(data_config.playlist_songs_file)

    # 2. Train Word2Vec model
    workers = w2v_config.workers if w2v_config.workers != -1 else os.cpu_count()
    logger.info(f"Initializing and training Word2Vec model with {workers} workers...")
    start_time = time.time()

    # By passing the iterator to the constructor, gensim handles both building
    # the vocabulary and training the model in a memory-efficient way.
    model = Word2Vec(
        sentences=sentences, # Correct parameter name
        vector_size=w2v_config.vector_size,
        window=w2v_config.window,
        min_count=w2v_config.min_count,
        workers=workers,
        sg=1,  # Use Skip-gram for quality
        sample=w2v_config.sample,
        epochs=w2v_config.epochs
        # compute_loss and callbacks removed for performance
    )
    
    end_time = time.time()
    logger.info(f"Word2Vec model training complete in {end_time - start_time:.2f} seconds.")

    # 3. Save the vectors to a CSV file
    output_file = data_config.song_vectors_file
    logger.info(f"Saving {len(model.wv.index_to_key)} song vectors to {output_file}...")
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        for song_id in model.wv.index_to_key:
            vector = model.wv[song_id]
            writer.writerow([song_id] + vector.tolist())

    # 4. Save the full model
    model_output_path = os.path.join(config.model_dir, "word2vec.model")
    model.save(model_output_path)
    logger.info(f"Full model saved to {model_output_path}")
    
    logger.info(f"--- Step G0 Completed Successfully. Song vectors are ready at {output_file} ---")

if __name__ == "__main__":
    config = Config()
    log_file_path = os.path.join(config.log_dir, "g0_train_word2vec.log")
    setup_logging(log_file=log_file_path)
    logger = logging.getLogger(__name__)

    if "path/to/your" in config.data.playlist_songs_file:
        logger.error("FATAL: Please edit 'config.py' and set the path for 'playlist_songs_file'.")
        sys.exit(1)

    train_song_vectors(config)