"""
Step G0: Train Song Vectors using Word2Vec/FastText.

This script now uses a pre-processed corpus file for maximum memory
efficiency and speed during training.
"""
import os
import sys
import pandas as pd
import logging
import time
from gensim.models import FastText
from gensim.models.callbacks import CallbackAny2Vec

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
        self.loss_before = 0.0

    def on_epoch_end(self, model):
        current_loss = model.get_latest_training_loss()
        epoch_loss = current_loss - self.loss_before
        self.loss_berfore = epoch_loss
        print(f"Epoch {self.epoch}/{self.total_epochs} finished. Loss: {epoch_loss}")
        self.epoch += 1

def train_song_vectors(config: Config):
    logger.info("--- Starting Step G0: Train Song Vectors ---")
    data_config = config.data
    w2v_config = config.word2vec

    corpus_file = w2v_config.corpus_file
    if not os.path.exists(corpus_file):
        logger.error(f"FATAL: Corpus file not found at {corpus_file}")
        logger.error("Please run Step G-1 (preprocess_playlists.py) first.")
        sys.exit(1)

    workers = w2v_config.workers if w2v_config.workers != -1 else os.cpu_count()
    logger.info(f"Initializing Word2Vec model with {workers} workers...")
    
    model = Word2Vec(
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

    logger.info(f"Building vocabulary from {corpus_file}...")
    model.build_vocab(corpus_file=corpus_file, progress_per=100000)

    logger.info("Starting model training...")
    model.train(
        corpus_file=corpus_file, 
        total_examples=model.corpus_count, 
        total_words=model.corpus_total_words, 
        epochs=w2v_config.epochs,
        compute_loss=True, 
        callbacks=[TqdmCallback(w2v_config.epochs)]
    )
    logger.info("Training complete.")

    output_file = data_config.song_vectors_file
    logger.info(f"Saving {len(model.wv.index_to_key)} song vectors to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        writer = pd.DataFrame(model.wv.vectors, index=model.wv.index_to_key)
        writer.columns = [f'v_{i}' for i in range(w2v_config.vector_size)]
        writer.index.name = 'mixsongid'
        writer.to_csv(f, sep='\t', header=False)

    model_output_path = os.path.join(config.model_dir, "word2vec.model")
    model.save(model_output_path)
    logger.info(f"Full model saved to {model_output_path}")
    
    logger.info(f"--- Step G0 Completed Successfully. Song vectors are ready at {output_file} ---")

if __name__ == "__main__":
    config = Config()
    log_file_path = os.path.join(config.log_dir, "g0_train_word2vec.log")
    setup_logging(log_file=log_file_path)
    logger = logging.getLogger(__name__)

    train_song_vectors(config)
