
Step G0b: Evaluate the quality of trained song vectors.

This script loads the trained song vectors, builds a Faiss index for efficient
similarity search, and provides an interactive prompt to find similar songs
for a given song ID.

import os
import sys
import numpy as np
import pandas as pd
import logging

try:
    import faiss
except ImportError:
    print("Faiss library not found. Please install it: pip install faiss-cpu")
    sys.exit(1)

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config
from src.common.utils import setup_logging

logger = logging.getLogger(__name__)

def load_song_info(song_info_path: str) -> pd.DataFrame:
    logger.info(f"Loading song info from {song_info_path}...")
    try:
        df = pd.read_csv(song_info_path, sep='	', header=None, names=['mixsongid', 'song_name', 'singer_name'])
        df.set_index('mixsongid', inplace=True)
        return df
    except FileNotFoundError:
        logger.error(f"Song info file not found at {song_info_path}. Cannot display song names.")
        return None

def get_song_details(song_info_df: pd.DataFrame, song_id: str) -> str:
    if song_info_df is None:
        return f"(ID: {song_id})"
    try:
        song = song_info_df.loc[song_id]
        return f"'{song['song_name']}' by {song['singer_name']} (ID: {song_id})"
    except KeyError:
        return f"Song with ID '{song_id}' (Details not found)"

def find_and_print_similar_songs(song_id, index, vectors, song_ids, id_to_index, song_info_df):
    if song_id not in id_to_index:
        print(f"\nError: Song ID '{song_id}' not found in the vocabulary.")
        return
        
    target_vector_index = id_to_index[song_id]
    target_vector = np.copy(vectors[target_vector_index:target_vector_index+1])
    faiss.normalize_L2(target_vector)

    print("\n" + "="*50)
    print("Query Song:")
    print(f"  {get_song_details(song_info_df, song_id)}")
    print("="*50)

    k = 11 
    distances, indices = index.search(target_vector, k)

    print("\nTop 10 Most Similar Songs:")
    for i in range(1, k):
        neighbor_index = indices[0][i]
        neighbor_id = song_ids[neighbor_index]
        similarity = distances[0][i]
        
        print(f"{i}. {get_song_details(song_info_df, neighbor_id)}")
        print(f"   (Similarity: {similarity:.4f})")

def main(config: Config):
    logger.info("--- Starting Step G0b: Evaluate Song Vectors ---")
    data_config = config.data

    # 1. Load Vectors
    logger.info(f"Loading song vectors from {data_config.song_vectors_file}...")
    try:
        vectors_df = pd.read_csv(data_config.song_vectors_file, dtype={'mixsongid': str})
        vectors_df.set_index('mixsongid', inplace=True)
    except FileNotFoundError:
        logger.error(f"FATAL: {data_config.song_vectors_file} not found. Run train_word2vec.py first.")
        return

    vectors = vectors_df.to_numpy().astype('float32')
    song_ids = vectors_df.index.tolist()
    id_to_index = {sid: i for i, sid in enumerate(song_ids)}

    # 2. Load Song Metadata
    song_info_df = load_song_info(data_config.song_info_file)

    # 3. Build Faiss Index
    logger.info("Building Faiss index...")
    dimension = vectors.shape[1]
    faiss.normalize_L2(vectors) # Normalize vectors for cosine similarity search
    index = faiss.IndexFlatIP(dimension) # IndexFlatIP is for dot product, which is cosine sim on normalized vectors
    index.add(vectors)
    logger.info(f"Index built successfully with {index.ntotal} vectors.")
    
    # 4. Interactive Search Loop
    print("\n" + "-"*20 + " Interactive Song Similarity Search " + "-"*20)
    print("Enter a song ID to find similar songs. Type 'exit' or 'quit' to end.")
    
    while True:
        try:
            query_id = input("\nEnter song ID: ").strip()
            if not query_id or query_id.lower() in ['exit', 'quit']:
                print("Exiting.")
                break
            
            find_and_print_similar_songs(query_id, index, vectors, song_ids, id_to_index, song_info_df)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    config = Config()
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)
    main(config)
