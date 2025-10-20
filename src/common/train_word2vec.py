"""
Step G0: Train Song Vectors using Word2Vec.

This script reads playlist-song data and uses a Word2Vec model to learn
a vector representation for each unique song based on its co-occurrence
in playlists.
"""
import os
import sys
import pandas as pd
import logging
from gensim.models import Word2Vec

# Adjust path to import from the project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.common.utils import setup_logging
from config import Config

logger = logging.getLogger(__name__)

def train_song_vectors(config: Config):
    """Main function to train and save song vectors."""
    logger.info("--- Starting Step G0: Train Song Vectors with Word2Vec ---")

    # 1. Load data
    data_config = config.data
    w2v_config = config.word2vec

    try:
        logger.info(f"Loading playlist songs from {data_config.playlist_songs_file}...")
        df = pd.read_csv(data_config.playlist_songs_file, dtype=str)
        # Ensure column names are consistent
        df.columns = ['playlist_id', 'song_id']
    except FileNotFoundError:
        logger.error(f"FATAL: Playlist songs file not found at {data_config.playlist_songs_file}")
        logger.error("Please update the path in 'config.py'.")
        return

    # 2. Prepare sentences (playlists)
    logger.info("Grouping songs into playlists (sentences)...")
    # Group by playlist_id and collect song_ids into a list
    sentences = df.groupby('playlist_id')['song_id'].apply(list).tolist()
    logger.info(f"Created {len(sentences)} sentences for training.")

    # 3. Train Word2Vec model
    logger.info(f"Training Word2Vec model with vector_size={w2v_config.vector_size}, window={w2v_config.window}...")
    
    # Determine number of workers
    workers = w2v_config.workers
    if workers == -1:
        workers = os.cpu_count()
        logger.info(f"Using all available CPU cores: {workers}")

    model = Word2Vec(
        sentences=sentences,
        vector_size=w2v_config.vector_size,
        window=w2v_config.window,
        min_count=w2v_config.min_count,
        sample=w2v_config.sample, # Add subsampling
        workers=workers,
        epochs=w2v_config.epochs,
        sg=0  # Use CBOW model as requested
    )
    logger.info("Word2Vec model training complete.")

    # 4. Save the vectors to a CSV file
    output_file = data_config.song_vectors_file
    logger.info(f"Saving {len(model.wv.index_to_key)} song vectors to {output_file}...")
    
    # Create a DataFrame from the learned vectors
    vectors_df = pd.DataFrame(model.wv.vectors, index=model.wv.index_to_key)
    vectors_df.columns = [f'v_{i}' for i in range(w2v_config.vector_size)]
    vectors_df.index.name = 'mixsongid'
    
    # Save to CSV
    vectors_df.to_csv(output_file)

    # 5. Save the full model for later use (optional)
    model_output_path = os.path.join(config.model_dir, "word2vec.model")
    model.save(model_output_path)
    logger.info(f"Full Word2Vec model saved to {model_output_path}")

    logger.info(f"--- Step G0 Completed Successfully. Song vectors are ready at {output_file} ---")

if __name__ == "__main__":
    config = Config()
    # Setup logging
    log_file_path = os.path.join(config.log_dir, "g0_train_word2vec.log")
    setup_logging(log_file=log_file_path)
    logger = logging.getLogger(__name__)

    # Check for placeholder path
    if config.data.playlist_songs_file == "path/to/your/gen_playlist_song.csv":
        logger.error("="*80)
        logger.error("FATAL: Please edit 'config.py' and set the path for 'playlist_songs_file'.")
        logger.error("="*80)
        sys.exit(1)

    train_song_vectors(config)
