
Step R1: Prepare Training Data for the Ranker Model.

This script creates positive and negative pairs for training the Cross-Encoder.
Positive pair: (playlist_title, song_in_that_playlist)
Negative pair: (playlist_title, random_song_not_in_that_playlist)

import os
import sys
import pandas as pd
import random
import logging
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config
from src.common.utils import setup_logging

logger = logging.getLogger(__name__)

class RankerDataBuilder:
    def __init__(self, config: Config):
        self.config = config
        self.data_config = config.data

    def run(self, num_neg_samples: int = 1):
        logger.info("--- Starting Step R1: Ranker Data Generation ---")
        playlist_info = self._load_playlist_info()
        playlist_songs = self._load_playlist_songs()
        all_song_ids = pd.read_csv(self.data_config.song_vectors_file, dtype={'mixsongid': str})['mixsongid'].unique().tolist()

        training_data = self._create_training_pairs(playlist_info, playlist_songs, all_song_ids, num_neg_samples)
        
        self._save_data(training_data)
        logger.info("--- Step R1 Completed Successfully ---")

    def _load_playlist_info(self) -> dict:
        logger.info(f"Loading playlist info from {self.data_config.playlist_info_file}...")
        df = pd.read_csv(self.data_config.playlist_info_file, dtype=str).set_index('glid')
        return df.to_dict('index')

    def _load_playlist_songs(self) -> dict:
        logger.info(f"Loading playlist songs from {self.data_config.playlist_songs_file}...")
        df = pd.read_csv(self.data_config.playlist_songs_file, dtype=str)
        df.columns = ['playlist_id', 'song_id']
        return df.groupby('playlist_id')['song_id'].apply(set).to_dict()

    def _create_training_pairs(self, playlist_info: dict, playlist_songs: dict, all_song_ids: list, num_neg_samples: int) -> list:
        logger.info("Creating positive and negative training pairs...")
        pairs = []
        for glid, songs_in_playlist in tqdm(playlist_songs.items(), desc="Creating pairs"):
            if glid not in playlist_info:
                continue
            title = playlist_info[glid].get('listname', '')
            if not title or not songs_in_playlist:
                continue

            # Add positive samples
            for song_id in songs_in_playlist:
                pairs.append((title, song_id, 1)) # (text, item, label)

            # Add negative samples
            num_positives = len(songs_in_playlist)
            num_negatives_to_generate = num_positives * num_neg_samples
            
            neg_samples_count = 0
            while neg_samples_count < num_negatives_to_generate:
                random_song = random.choice(all_song_ids)
                if random_song not in songs_in_playlist:
                    pairs.append((title, random_song, 0))
                    neg_samples_count += 1
        return pairs

    def _save_data(self, data: list):
        output_dir = os.path.join(self.config.output_dir, "ranker")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "ranking_train_data.tsv")
        logger.info(f"Saving {len(data)} training pairs to {output_path}...")
        df = pd.DataFrame(data, columns=['text', 'song_id', 'label'])
        df.to_csv(output_path, sep='\t', index=False)

if __name__ == "__main__":
    config = Config()
    log_file_path = os.path.join(config.log_dir, "r1_prepare_ranker_data.log")
    setup_logging(log_file=log_file_path)
    logger = logging.getLogger(__name__)
    builder = RankerDataBuilder(config)
    builder.run(num_neg_samples=1) # For each positive, create 1 negative sample
