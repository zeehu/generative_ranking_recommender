
"""
Step G0b: Train Song Vectors using Word2Vec.

This script uses a pre-processed corpus file for maximum memory
efficiency and speed during training.
"""
import os
import sys
import pandas as pd
import logging
import time
import csv
from gensim.models import Word2Vec

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config
from src.common.utils import setup_logging

logger = logging.getLogger(__name__)

def train_song_vectors(config: Config):
    logger.info("--- Starting Step G0b: Train Song Vectors ---")
    data_config = config.data
    w2v_config = config.word2vec

    corpus_file = w2v_config.corpus_file
    if not os.path.exists(corpus_file):
        logger.error(f"FATAL: Corpus file not found at {corpus_file}")
        logger.error("Please run Step G0a (preprocess_playlists.py) first.")
        sys.exit(1)

    workers = w2v_config.workers if w2v_config.workers != -1 else os.cpu_count()
    logger.info(f"Initializing and training Word2Vec model with {workers} workers...")
    start_time = time.time()

    # Use the corpus_file argument for memory-efficient training
    model = Word2Vec(
        corpus_file=corpus_file,
        vector_size=w2v_config.vector_size,
        window=w2v_config.window,
        min_count=w2v_config.min_count,
        workers=workers,
        sg=1,  # Skip-gram for quality
        sample=w2v_config.sample,
        epochs=w2v_config.epochs
    )
    
    end_time = time.time()
    logger.info(f"Word2Vec model training complete in {end_time - start_time:.2f} seconds.")

    # Save the vectors to a CSV file
    output_file = data_config.song_vectors_file
    logger.info(f"Saving {len(model.wv.index_to_key)} song vectors to {output_file}...")
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        for song_id in model.wv.index_to_key:
            vector = model.wv[song_id]
            writer.writerow([song_id] + vector.tolist())

    # Save the full model
    model_output_path = os.path.join(config.model_dir, "word2vec.model")
    model.save(model_output_path)
    logger.info(f"Full model saved to {model_output_path}")
    
    logger.info(f"--- Step G0b Completed Successfully. Song vectors are ready at {output_file} ---")

if __name__ == "__main__":
    config = Config()
    log_file_path = os.path.join(config.log_dir, "g0b_train_word2vec.log")
    setup_logging(log_file=log_file_path)
    logger = logging.getLogger(__name__)

    train_song_vectors(config)
