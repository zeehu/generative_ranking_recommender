import os
import argparse
import numpy as np
import pandas as pd
import faiss
from gensim.models import Word2Vec

def load_song_info(song_info_path):
    """Loads song information from a CSV file into a pandas DataFrame."""
    print(f"Loading song info from {song_info_path}...")
    try:
        df = pd.read_csv(song_info_path, dtype={'mix_song_id': str})
        df.set_index('mix_song_id', inplace=True)
        return df
    except FileNotFoundError:
        print(f"Error: Song info file not found at {song_info_path}.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the song info file: {e}")
        return None

def get_song_details(song_info_df, song_id):
    """Retrieves and formats song details from the DataFrame."""
    try:
        song = song_info_df.loc[song_id]
        return f"'{song['song_name']}' by {song['singer_name']} (ID: {song_id})"
    except KeyError:
        return f"Song with ID '{song_id}' (Details not found)"
    except Exception:
        return f"Song with ID '{song_id}' (Error retrieving details)"

def find_and_print_similar_songs(song_id, index, vectors, song_ids, id_to_index, song_info_df):
    """
    Performs the Faiss search for a given song_id and prints the results.
    """
    if song_id not in id_to_index:
        print(f"\nError: Song ID '{song_id}' not found in the model's vocabulary.")
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

def main(args):
    """
    Loads a model, builds/loads a Faiss index, and enters an interactive loop.
    """
    # --- 1. Load Model and Vectors ---
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return

    print(f"Loading Word2Vec model from {args.model_path}...")
    model = Word2Vec.load(args.model_path)
    
    vectors = model.wv.vectors.astype('float32')
    song_ids = model.wv.index_to_key
    id_to_index = {sid: i for i, sid in enumerate(song_ids)}

    # --- 2. Load Song Metadata ---
    song_info_df = load_song_info(args.song_info_path)
    if song_info_df is None:
        print("Cannot proceed without song info. Exiting.")
        return

    # --- 3. Load or Build Faiss Index ---
    rebuild_index = True
    if os.path.exists(args.index_path):
        try:
            model_mtime = os.path.getmtime(args.model_path)
            index_mtime = os.path.getmtime(args.index_path)
            if model_mtime < index_mtime:
                print(f"Loading existing Faiss index from {args.index_path}...")
                index = faiss.read_index(args.index_path)
                rebuild_index = False
                print("Index loaded successfully.")
            else:
                print("Model has been updated. Rebuilding index...")
        except Exception as e:
            print(f"Failed to load index or check timestamps: {e}. Rebuilding...")
    
    if rebuild_index:
        print("Building new Faiss index...")
        dimension = vectors.shape[1]
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(dimension)
        index.add(vectors)
        print(f"Index built successfully with {index.ntotal} vectors.")
        
        print(f"Saving index to {args.index_path}...")
        faiss.write_index(index, args.index_path)
    
    # The faiss.normalize_L2 function normalizes the array in-place.
    
    # --- 4. Interactive Search Loop ---
    print("\n" + "-"*20 + " Interactive Song Similarity Search " + "-"*20)
    print("Enter a song ID to find similar songs. Type 'exit' or 'quit' to end.")
    
    while True:
        try:
            query_id = input("\nEnter song ID: ").strip()
            if not query_id or query_id.lower() in ['exit', 'quit']:
                print("Exiting.")
                break
            
            find_and_print_similar_songs(
                query_id, index, vectors, song_ids, id_to_index, song_info_df
            )
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find similar songs interactively using a trained Word2Vec model and Faiss.")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_model_path = os.path.join(current_dir, "models", "song2vec.model")
    default_song_info_path = os.path.join(current_dir, "data", "gen_song_info.csv")
    default_index_path = os.path.join(current_dir, "models", "faiss_index.bin")

    parser.add_argument(
        "--model_path",
        type=str,
        default=default_model_path,
        help="Path to the trained song2vec.model file."
    )
    parser.add_argument(
        "--song_info_path",
        type=str,
        default=default_song_info_path,
        help="Path to the gen_song_info.csv file."
    )
    parser.add_argument(
        "--index_path",
        type=str,
        default=default_index_path,
        help="Path to save/load the Faiss index file."
    )

    args = parser.parse_args()
    main(args)