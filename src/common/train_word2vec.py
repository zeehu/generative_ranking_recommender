
"""
Step G0: Train Song Vectors using Word2Vec/FastText.

This script uses a memory-efficient streaming approach to process a large,
sorted playlist file, trains a gensim model, and saves the vectors.
"""
import os
import sys
import pandas as pd
import logging
import time
import csv
from gensim.models import FastText
from gensim.models.callbacks import CallbackAny2Vec
from itertools import groupby
from tqdm import tqdm

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config
from src.common.utils import setup_logging

logger = logging.getLogger(__name__)

class TqdmCallback(CallbackAny2Vec):
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.epoch = 0
        self.loss_before = 0

    def on_epoch_begin(self, model):
        self.epoch += 1
        print(f'Epoch {self.epoch}/{self.total_epochs}')
        self.loss_before = model.get_latest_training_loss()

    def on_epoch_end(self, model):
        current_loss = model.get_latest_training_loss()
        epoch_loss = current_loss - self.loss_before
        print(f"Epoch finished. Loss: {epoch_loss}")

class PlaylistCorpus:
    """An iterator that yields playlists (sentences) from a large, sorted CSV file."""
    def __init__(self, filepath):
        self.filepath = filepath

    def __iter__(self):
        with open(self.filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            # Group by the first column (playlist_id)
            for _, group in groupby(reader, key=lambda x: x[0]):
                yield [row[1] for row in group]

def train_song_vectors(config: Config):
    logger.info("--- Starting Step G0: Train Song Vectors ---")
    data_config = config.data
    w2v_config = config.word2vec

    # 1. Prepare a corpus iterator
    try:
        logger.info(f"Initializing corpus iterator for {data_config.playlist_songs_file}...")
        sentences = PlaylistCorpus(data_config.playlist_songs_file)
    except Exception as e:
        logger.error(f"FATAL: Failed to read {data_config.playlist_songs_file}. Error: {e}")
        sys.exit(1)

    # 2. Train Word2Vec model
    workers = w2v_config.workers if w2v_config.workers != -1 else os.cpu_count()
    logger.info(f"Initializing FastText model with {workers} workers...")
    
    model = FastText(
        corpus_iterable=sentences,
        vector_size=w2v_config.vector_size,
        window=w2v_config.window,
        min_count=w2v_config.min_count,
        workers=workers,
        sg=0,  # Use CBOW
        sample=1e-4,
        epochs=w2v_config.epochs,
        callbacks=[TqdmCallback(w2v_config.epochs)],
        compute_loss=True
    )
    logger.info("Word2Vec model training complete.")

    # 3. Save the vectors to a CSV file
    output_file = data_config.song_vectors_file
    logger.info(f"Saving {len(model.wv.index_to_key)} song vectors to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        # No header
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
