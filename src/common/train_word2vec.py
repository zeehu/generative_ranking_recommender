

"""
Step G0: Train Song Vectors using Word2Vec/FastText.

This script adapts the user's reference implementation.
It reads playlist-song data, prepares a temporary corpus file for memory
efficiency, trains a gensim model, and saves the resulting vectors to CSV.
"""
import os
import sys
import pandas as pd
import logging
import time
from gensim.models import FastText # Using FastText as per user's reference code
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
    """Callback to show progress bar and training loss."""
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.epoch = 0
        self.loss_before = 0
        self.start_time = 0

    def on_epoch_begin(self, model):
        self.epoch += 1
        self.start_time = time.time()
        print(f'Epoch {self.epoch}/{self.total_epochs}')
        self.loss_before = model.get_latest_training_loss()

    def on_epoch_end(self, model):
        duration = time.time() - self.start_time
        current_loss = model.get_latest_training_loss()
        epoch_loss = current_loss - self.loss_before
        print(f"Epoch finished in {duration:.2f}s. Loss: {epoch_loss}")

def prepare_corpus_file(input_csv: str, output_txt: str):
    logger.info(f"Preparing corpus file at {output_txt} from {input_csv}...")
    with open(input_csv, 'r', encoding='utf-8') as fin, \
         open(output_txt, 'w', encoding='utf-8') as fout:
        
        reader = pd.read_csv(fin, dtype=str, chunksize=1000000)
        for chunk in tqdm(reader, desc="Converting playlists to corpus file"):
            chunk.columns = ['playlist_id', 'song_id']
            for _, playlist in chunk.groupby('playlist_id'):
                fout.write(' '.join(playlist['song_id']) + '\n')
    logger.info("Corpus file prepared.")

def train_song_vectors(config: Config):
    logger.info("--- Starting Step G0: Train Song Vectors ---")
    data_config = config.data
    w2v_config = config.word2vec

    temp_corpus_file = os.path.join(config.output_dir, "temp_corpus.txt")

    prepare_corpus_file(data_config.playlist_songs_file, temp_corpus_file)

    workers = w2v_config.workers if w2v_config.workers != -1 else os.cpu_count()
    logger.info(f"Initializing FastText model with {workers} workers...")
    
    model = FastText(
        vector_size=w2v_config.vector_size,
        window=w2v_config.window,
        min_count=w2v_config.min_count,
        workers=workers,
        sg=0,  # Use CBOW as requested
        sample=1e-4, # Subsample frequent words
        epochs=w2v_config.epochs
    )

    logger.info("Building vocabulary from corpus file...")
    model.build_vocab(corpus_file=temp_corpus_file, progress_per=100000)

    logger.info("Starting model training...")
    model.train(
        corpus_file=temp_corpus_file, 
        total_examples=model.corpus_count, 
        total_words=model.corpus_total_words, 
        epochs=model.epochs,
        compute_loss=True, 
        callbacks=[TqdmCallback(model.epochs)]
    )
    logger.info("Training complete.")

    output_file = data_config.song_vectors_file
    logger.info(f"Saving {len(model.wv.index_to_key)} song vectors to {output_file}...")
    vectors_df = pd.DataFrame(model.wv.vectors, index=model.wv.index_to_key)
    vectors_df.columns = [f'v_{i}' for i in range(w2v_config.vector_size)]
    vectors_df.index.name = 'mixsongid'
    vectors_df.to_csv(output_file)

    model_output_path = os.path.join(config.model_dir, "word2vec.model")
    model.save(model_output_path)
    logger.info(f"Full model saved to {model_output_path}")
    
    os.remove(temp_corpus_file)
    logger.info(f"Removed temporary corpus file: {temp_corpus_file}")
    logger.info(f"--- Step G0 Completed Successfully ---")

if __name__ == "__main__":
    config = Config()
    log_file_path = os.path.join(config.log_dir, "g0_train_word2vec.log")
    setup_logging(log_file=log_file_path)
    logger = logging.getLogger(__name__)

    if config.data.playlist_songs_file == "path/to/your/gen_playlist_song.csv":
        logger.error("FATAL: Please edit 'config.py' and set the path for 'playlist_songs_file'.")
        sys.exit(1)

    train_song_vectors(config)
