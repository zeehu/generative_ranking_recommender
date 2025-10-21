import os
import csv
import time
import multiprocessing
from gensim.models import FastText
from gensim.models.callbacks import CallbackAny2Vec
from itertools import groupby
from tqdm import tqdm


def prepare_corpus_file(input_csv, output_txt, min_len=5, max_len=500):
    """
    Converts a playlist CSV file into a line-by-line text corpus file
    that gensim's `corpus_file` argument can use efficiently.

    This function streams the data and is memory-efficient.
    """
    print(f"Preparing corpus file at {output_txt} from {input_csv}...")
    
    # First, count playlists for the progress bar in a memory-efficient way
    print("Counting playlists...")
    total_playlists = 0
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip header
        for _ in groupby(reader, key=lambda x: x[0]):
            total_playlists += 1
    
    # Now, process the file and write the output
    print(f"Found {total_playlists} total playlists. Converting to corpus file...")
    with open(input_csv, 'r', encoding='utf-8') as fin, \
         open(output_txt, 'w', encoding='utf-8') as fout:
        
        reader = csv.reader(fin)
        next(reader, None)  # Skip header
        
        groups = groupby(reader, key=lambda x: x[0])
        
        processed_count = 0
        for _, song_group in tqdm(groups, total=total_playlists, desc="Converting playlists"):
            playlist = [song[1] for song in song_group]
            
            if min_len <= len(playlist) < max_len:
                fout.write(' '.join(playlist) + '\n')
                processed_count += 1

    print(f"Corpus file prepared. Kept {processed_count} playlists out of {total_playlists}.")
    return output_txt


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
        # TQDM progress bar is handled by gensim's logging now
        self.loss_before = model.get_latest_training_loss()

    def on_batch_end(self, model, job_params, **kwargs):
        # This is no longer needed as gensim will log progress
        pass

    def on_epoch_end(self, model):
        duration = time.time() - self.start_time
        current_loss = model.get_latest_training_loss()
        epoch_loss = current_loss - self.loss_before
        print(f"Epoch finished in {duration:.2f}s. Loss: {epoch_loss}")

def train_song_vectors(data_path, model_path):
    """
    Trains song vectors using word2vec.

    Args:
        data_path (str): Path to the directory containing playlist data.
        model_path (str): Path to save the trained model.
    """
    input_csv_file = os.path.join(data_path, "gen_playlist_song.csv")
    corpus_txt_file = os.path.join(data_path, "playlists.txt")

    if not os.path.exists(input_csv_file):
        print(f"Data file not found: {input_csv_file}")
        return

    # Step 1: Prepare the corpus file in a memory-efficient way
    prepare_corpus_file(input_csv_file, corpus_txt_file, min_len=10, max_len=300)

    # Step 2: Initialize the model
    epochs = 30
    workers = 18 # Fixed number of workers
    print(f"Initializing FastText model with {workers} workers...")
    model = FastText(vector_size=100, window=200, min_count=5, workers=workers, 
                     sample=1e-3, sg=1, max_n=0)

    # Step 3: Build vocabulary
    print("Building vocabulary from corpus file...")
    model.build_vocab(corpus_file=corpus_txt_file)

    # Step 4: Train the model
    print("Starting model training...")
    tqdm_callback = TqdmCallback(epochs)
    model.train(corpus_file=corpus_txt_file, 
                total_examples=model.corpus_count, 
                total_words=model.corpus_total_words, 
                epochs=epochs, 
                compute_loss=True, 
                callbacks=[tqdm_callback])
    
    print("\nTraining complete.")
    
    model.save(model_path)
    print(f"Model saved to {model_path}")


