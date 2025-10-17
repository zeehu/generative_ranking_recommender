
"""
Step R1: Prepare Training Data for the Ranker Model (with Hard Negatives).

This script creates positive and negative pairs for training the Cross-Encoder.
Positive pair: (playlist_title, song_in_that_playlist)
Hard Negative pair: (playlist_title, song_in_same_cluster_but_not_in_playlist)
"""
import os
import sys
import pandas as pd
import random
import logging
from tqdm import tqdm
from collections import defaultdict
import json

# Add project root to sys.path
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
        logger.info("--- Starting Step R1: Ranker Data Generation (with Hard Negatives) ---")
        playlist_info = self._load_playlist_info()
        playlist_songs = self._load_playlist_songs()
        song_to_sem_id, sem_id_to_songs = self._load_semantic_maps()
        all_song_ids = list(song_to_sem_id.keys())

        training_data = self._create_training_pairs(
            playlist_info, playlist_songs, all_song_ids, 
            song_to_sem_id, sem_id_to_songs, num_neg_samples
        )
        
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

    def _load_semantic_maps(self) -> tuple[dict, dict]:
        logger.info(f"Loading semantic ID maps from {self.data_config.semantic_ids_file}...")
        song_to_sem_id = {}
        sem_id_to_songs = defaultdict(list)
        try:
            with open(self.data_config.semantic_ids_file, 'r') as f:
                for line in f: 
                    item = json.loads(line)
                    song_id, sem_id = item['song_id'], tuple(item['semantic_ids'])
                    song_to_sem_id[song_id] = sem_id
                    sem_id_to_songs[sem_id].append(song_id)
        except FileNotFoundError: 
            logger.error(f"FATAL: {self.data_config.semantic_ids_file} not found. Run G1 first.")
            sys.exit(1)
        return dict(song_to_sem_id), dict(sem_id_to_songs)

    def _create_training_pairs(self, playlist_info, playlist_songs, all_song_ids, 
                               song_to_sem_id, sem_id_to_songs, num_neg_samples) -> list:
        logger.info("Creating positive and hard negative training pairs...")
        pairs = []
        for glid, songs_in_playlist in tqdm(playlist_songs.items(), desc="Creating pairs"):
            if glid not in playlist_info:
                continue
            title = playlist_info[glid].get('listname', '')
            if not title or not songs_in_playlist:
                continue

            for positive_song_id in songs_in_playlist:
                # Add positive sample
                pairs.append((title, positive_song_id, 1))

                # Generate hard negative samples
                positive_sem_id = song_to_sem_id.get(positive_song_id)
                if not positive_sem_id:
                    continue

                hard_candidates = sem_id_to_songs.get(positive_sem_id, [])
                potential_negatives = set(hard_candidates) - songs_in_playlist

                neg_samples_count = 0
                while neg_samples_count < num_neg_samples:
                    if potential_negatives:
                        # Prioritize hard negatives from the same cluster
                        hard_negative_song_id = random.choice(list(potential_negatives))
                        pairs.append((title, hard_negative_song_id, 0))
                        potential_negatives.remove(hard_negative_song_id) # Avoid re-sampling the same negative
                    else:
                        # Fallback to random negative if no hard negatives are available
                        random_song_id = random.choice(all_song_ids)
                        if random_song_id not in songs_in_playlist:
                            pairs.append((title, random_song_id, 0))
                        else:
                            continue # Retry if we randomly picked a positive sample
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
    builder.run(num_neg_samples=1)
