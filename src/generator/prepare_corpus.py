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
        line_count = 0
        error_count = 0
        
        with open(self.data_config.semantic_ids_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line_count += 1
                try:
                    item = json.loads(line.strip())
                    if 'song_id' not in item or 'semantic_ids' not in item:
                        logger.warning(f"Line {line_num}: Missing required fields")
                        error_count += 1
                        continue
                    
                    semantic_ids = item['semantic_ids']
                    if not isinstance(semantic_ids, list) or len(semantic_ids) != 3:
                        logger.warning(f"Line {line_num}: Invalid semantic_ids format (expected list of 3 integers)")
                        error_count += 1
                        continue
                    
                    mapping[item['song_id']] = semantic_ids
                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_num}: JSON decode error - {e}")
                    error_count += 1
                    continue
        
        logger.info(f"Loaded {len(mapping)} song-to-semantic-ID mappings from {line_count} lines.")
        if error_count > 0:
            logger.warning(f"Encountered {error_count} errors while loading semantic IDs")
        
        if len(mapping) == 0:
            logger.error("FATAL: No valid semantic IDs loaded!")
            sys.exit(1)
        
        return mapping

    def _load_playlist_info(self) -> dict:
        logger.info(f"Loading playlist info from {self.data_config.playlist_info_file}...")
        try:
            df = pd.read_csv(self.data_config.playlist_info_file, sep='\t')
            df.set_index('glid', inplace=True)
            return df.to_dict('index')
        except FileNotFoundError:
            logger.error(f"FATAL: Playlist info file not found at {self.data_config.playlist_info_file}")
            sys.exit(1)

    def _load_playlist_songs(self) -> dict:
        logger.info(f"Loading playlist songs from {self.data_config.playlist_songs_file}...")
        try:
            # Use chunked reading for better memory efficiency with large files
            chunk_size = 1000000
            playlist_songs = {}
            
            for chunk in pd.read_csv(
                self.data_config.playlist_songs_file, 
                sep='\t', 
                header=None, 
                names=['playlist_id', 'song_id'], 
                dtype=str,
                chunksize=chunk_size
            ):
                grouped = chunk.groupby('playlist_id')['song_id'].apply(list)
                for playlist_id, songs in grouped.items():
                    if playlist_id in playlist_songs:
                        playlist_songs[playlist_id].extend(songs)
                    else:
                        playlist_songs[playlist_id] = songs
            
            logger.info(f"Loaded {len(playlist_songs)} playlists")
            return playlist_songs
        except FileNotFoundError:
            logger.error(f"FATAL: Playlist songs file not found at {self.data_config.playlist_songs_file}")
            sys.exit(1)

    def _build_corpus(self, playlist_info: dict, playlist_songs: dict, semantic_id_map: dict) -> list:
        """
        Build text-to-text corpus with layer-specific semantic ID tokens.
        
        Each semantic ID is represented as <id_l{layer}_{id}> to distinguish
        which layer the ID belongs to. This allows the T5 model to learn
        multi-granular semantic information from the three-layer hierarchical IDs.
        """
        logger.info("Building text-to-text corpus...")
        corpus = []
        stats = {
            'total_playlists': len(playlist_songs),
            'playlists_without_info': 0,
            'playlists_without_title': 0,
            'playlists_too_few_songs': 0,
            'total_songs': 0,
            'songs_with_semantic_ids': 0,
            'songs_without_semantic_ids': 0,
        }
        
        for glid, songs in tqdm(playlist_songs.items(), desc="Processing playlists"):
            if glid not in playlist_info:
                stats['playlists_without_info'] += 1
                continue
            
            if not songs:
                continue

            title = playlist_info[glid].get('listname', '')
            if not title:
                stats['playlists_without_title'] += 1
                continue

            # Per our "embrace collision" strategy, we do not de-duplicate songs or tokens here.
            sorted_songs = sorted(songs)
            semantic_tokens = []
            songs_with_semantic_ids = 0 # Count songs that actually have semantic IDs
            
            stats['total_songs'] += len(sorted_songs)
            
            for song_id in sorted_songs:
                if song_id in semantic_id_map:
                    # Create layer-specific tokens for each of the three layers
                    # semantic_ids = [layer1_id, layer2_id, layer3_id]
                    semantic_ids = semantic_id_map[song_id]
                    tokens = [
                        f"<id_l1_{semantic_ids[0]}>",  # Layer 1 token
                        f"<id_l2_{semantic_ids[1]}>",  # Layer 2 token
                        f"<id_l3_{semantic_ids[2]}>",  # Layer 3 token
                    ]
                    semantic_tokens.extend(tokens)
                    songs_with_semantic_ids += 1
                    stats['songs_with_semantic_ids'] += 1
                else:
                    stats['songs_without_semantic_ids'] += 1
            
            # Filter out playlists with too few songs after semantic ID filtering
            if songs_with_semantic_ids < self.data_config.min_songs_per_playlist:
                stats['playlists_too_few_songs'] += 1
                continue

            # If no semantic tokens were found (e.g., all songs filtered out), skip
            if not semantic_tokens:
                continue

            max_len = self.t5_config.max_target_length - 1
            truncated_tokens = semantic_tokens[:max_len]
            output_sequence = " ".join(truncated_tokens) + " <eos>"
            corpus.append((glid, title, output_sequence))
        
        logger.info(f"Successfully built corpus with {len(corpus)} entries.")
        logger.info("Corpus building statistics:")
        logger.info(f"  Total playlists: {stats['total_playlists']}")
        logger.info(f"  Valid corpus entries: {len(corpus)}")
        logger.info(f"  Playlists without info: {stats['playlists_without_info']}")
        logger.info(f"  Playlists without title: {stats['playlists_without_title']}")
        logger.info(f"  Playlists with too few songs: {stats['playlists_too_few_songs']}")
        logger.info(f"  Total songs processed: {stats['total_songs']}")
        logger.info(f"  Songs with semantic IDs: {stats['songs_with_semantic_ids']} ({stats['songs_with_semantic_ids']/stats['total_songs']*100:.2f}%)")
        logger.info(f"  Songs without semantic IDs: {stats['songs_without_semantic_ids']} ({stats['songs_without_semantic_ids']/stats['total_songs']*100:.2f}%)")
        
        if len(corpus) == 0:
            logger.error("FATAL: No valid corpus entries generated!")
            sys.exit(1)
        
        return corpus

    def _split_and_save(self, corpus: list):
        logger.info("Splitting data and saving to files...")
        # Set seed for reproducible splits
        random.seed(self.config.seed)
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
        
        # Validate split ratios
        if len(val_data) == 0:
            logger.warning("Validation set is empty! Consider adjusting split ratios.")
        if len(test_data) == 0:
            logger.warning("Test set is empty! Consider adjusting split ratios.")
        
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