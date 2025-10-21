"""
Step G0: Train Song Vectors using Word2Vec/FastText.

This script reads playlist-song data, groups songs into playlists in a memory-safe
way, trains a gensim model, and saves the resulting vectors to CSV.
"""
import os
import sys
import pandas as pd
import logging
import time
from gensim.models import FastText
from gensim.models.callbacks import CallbackAny2Vec
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

def train_song_vectors(config: Config):
    logger.info("--- Starting Step G0: Train Song Vectors ---")
    data_config = config.data
    w2v_config = config.word2vec

    # 1. Load data and group into sentences (playlists)
    try:
        logger.info(f"Loading playlist songs from {data_config.playlist_songs_file}...")
        df = pd.read_csv(data_config.playlist_songs_file, dtype=str)
        df.columns = ['playlist_id', 'song_id']
        logger.info("Grouping songs into playlists (sentences)...")
        sentences = df.groupby('playlist_id')['song_id'].apply(list).tolist()
        logger.info(f"Created {len(sentences)} sentences for training.")
    except FileNotFoundError:
        logger.error(f"FATAL: Playlist songs file not found at {data_config.playlist_songs_file}")
        sys.exit(1)

    # 2. Train Word2Vec model
    workers = w2v_config.workers if w2v_config.workers != -1 else os.cpu_count()
    logger.info(f"Initializing FastText model with {workers} workers...")
    
    model = FastText(
        sentences=sentences,
        vector_size=w2v_config.vector_size,
        window=w2v_config.window,
        min_count=w2v_config.min_count,
        workers=workers,
        sg=0,  # Use CBOW as requested
        sample=1e-4, # Subsample frequent words
        epochs=w2v_config.epochs,
        callbacks=[TqdmCallback(w2v_config.epochs)],
        compute_loss=True
    )
    logger.info("Word2Vec model training complete.")

    # 3. Save the vectors to a CSV file
    output_file = data_config.song_vectors_file
    logger.info(f"Saving {len(model.wv.index_to_key)} song vectors to {output_file}...")
    vectors_df = pd.DataFrame(model.wv.vectors, index=model.wv.index_to_key)
    vectors_df.columns = [f'v_{i}' for i in range(w2v_config.vector_size)]
    vectors_df.index.name = 'mixsongid'
    vectors_df.to_csv(output_file)

    # 4. Save the full model for later use
    model_output_path = os.path.join(config.model_dir, "word2vec.model")
    model.save(model_output_path)
    logger.info(f"Full model saved to {model_output_path}")
    
    logger.info(f"--- Step G0 Completed Successfully. Song vectors are ready at {output_file} ---")

if __name__ == "__main__":
    config = Config()
    log_file_path = os.path.join(config.log_dir, "g0_train_word2vec.log")
    setup_logging(log_file=log_file_path)
    logger = logging.getLogger(__name__)

    if config.data.playlist_songs_file == "path/to/your/gen_playlist_song.csv":
        logger.error("FATAL: Please edit 'config.py' and set the path for 'playlist_songs_file'.")
        sys.exit(1)

    train_song_vectors(config)