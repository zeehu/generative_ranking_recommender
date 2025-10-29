"""
Step G2: Generate Training Corpus for the T5 Generator Model.

This script reads the raw playlist data, combines it with the generated
semantic IDs (song-to-cluster map), and produces train/val/test splits
in a `playlist_id	input_text	output_sequence` format.
"""
import os
import sys
import pandas as pd
import json
from tqdm import tqdm
import logging
import random

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
sys.path.insert(0, project_root)

from config import Config
from src.common.utils import setup_logging

logger = logging.getLogger(__name__)

class CorpusBuilder:
    def __init__(self, config: Config):
        self.config = config
        self.data_config = config.data
        self.t5_config = config.generator_t5

    def run(self):
        logger.info("--- Starting Step G2: Generator Corpus Generation ---")
        semantic_id_map = self._load_semantic_ids()
        playlist_info = self._load_playlist_info()
        playlist_songs = self._load_playlist_songs()
        corpus = self._build_corpus(playlist_info, playlist_songs, semantic_id_map)
        self._split_and_save(corpus)
        logger.info("--- Step G2 Completed Successfully ---")

    def _load_semantic_ids(self) -> dict:
        logger.info(f"Loading semantic IDs from {self.data_config.semantic_ids_file}...")
        if not os.path.exists(self.data_config.semantic_ids_file):
            logger.error(f"FATAL: Semantic ID file not found. Please run Step G1 first.")
            sys.exit(1)
        mapping = {}
        with open(self.data_config.semantic_ids_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                mapping[item['song_id']] = item['semantic_ids']
        logger.info(f"Loaded {len(mapping)} song-to-semantic-ID mappings.")
        return mapping

    def _load_playlist_info(self) -> dict:
        logger.info(f"Loading playlist info from {self.data_config.playlist_info_file}...")
        try:
            df = pd.read_csv(self.data_config.playlist_info_file, sep='\t', header=None, names=['glid', 'listname', 'tag_list'])
            df.set_index('glid', inplace=True)
            return df.to_dict('index')
        except FileNotFoundError:
            logger.error(f"FATAL: Playlist info file not found at {self.data_config.playlist_info_file}")
            sys.exit(1)

    def _load_playlist_songs(self) -> dict:
        logger.info(f"Loading playlist songs from {self.data_config.playlist_songs_file}...")
        try:
            df = pd.read_csv(self.data_config.playlist_songs_file, dtype=str)
            df.columns = ['playlist_id', 'song_id']
            grouped = df.groupby('playlist_id')['song_id'].apply(list)
            return grouped.to_dict()
        except FileNotFoundError:
            logger.error(f"FATAL: Playlist songs file not found at {self.data_config.playlist_songs_file}")
            sys.exit(1)

    def _build_corpus(self, playlist_info: dict, playlist_songs: dict, semantic_id_map: dict) -> list:
        logger.info("Building text-to-text corpus...")
        corpus = []
        for glid, songs in tqdm(playlist_songs.items(), desc="Processing playlists"):
            if glid not in playlist_info or not songs:
                continue

            title = playlist_info[glid].get('listname', '')
            if not title:
                continue

            # Per our "embrace collision" strategy, we do not de-duplicate songs or tokens here.
            sorted_songs = sorted(songs)
            semantic_tokens = []
            songs_with_semantic_ids = 0 # Count songs that actually have semantic IDs
            for song_id in sorted_songs:
                if song_id in semantic_id_map:
                    tokens = [f"<id_{sid}>" for sid in semantic_id_map[song_id]]
                    semantic_tokens.extend(tokens)
                    songs_with_semantic_ids += 1
            
            # Filter out playlists with too few songs after semantic ID filtering
            if songs_with_semantic_ids < self.data_config.min_songs_per_playlist:
                continue

            # If no semantic tokens were found (e.g., all songs filtered out), skip
            if not semantic_tokens:
                continue

            max_len = self.t5_config.max_target_length - 1
            truncated_tokens = semantic_tokens[:max_len]
            output_sequence = " ".join(truncated_tokens) + " <eos>"
            corpus.append((glid, title, output_sequence))
        
        logger.info(f"Successfully built corpus with {len(corpus)} entries.")
        return corpus

    def _split_and_save(self, corpus: list):
        logger.info("Splitting data and saving to files...")
        random.shuffle(corpus)
        train_ratio = self.data_config.train_split_ratio
        val_ratio = self.data_config.val_split_ratio
        
        # Calculate split indices
        total_len = len(corpus)
        train_end_idx = int(total_len * train_ratio)
        val_end_idx = train_end_idx + int(total_len * val_ratio)
        
        train_data = corpus[:train_end_idx]
        val_data = corpus[train_end_idx:val_end_idx]
        test_data = corpus[val_end_idx:] # Remaining data for test

        logger.info(f"Data split: {len(train_data)} train, {len(val_data)} validation, {len(test_data)} test.")
        output_dir = os.path.join(self.config.output_dir, "generator")
        os.makedirs(output_dir, exist_ok=True)
        self._save_to_tsv(train_data, os.path.join(output_dir, "train.tsv"))
        self._save_to_tsv(val_data, os.path.join(output_dir, "val.tsv"))
        self._save_to_tsv(test_data, os.path.join(output_dir, "test.tsv"))

    def _save_to_tsv(self, data: list, file_path: str):
        logger.info(f"Saving {len(data)} records to {file_path}...")
        with open(file_path, 'w', encoding='utf-8') as f:
            for glid, input_text, output_sequence in data:
                f.write(f"{glid}\t{input_text}\t{output_sequence}\n")

if __name__ == "__main__":
    config = Config()
    log_file_path = os.path.join(config.log_dir, "g2_prepare_corpus.log")
    setup_logging(log_file=log_file_path)
    logger = logging.getLogger(__name__)
    builder = CorpusBuilder(config)
    builder.run()