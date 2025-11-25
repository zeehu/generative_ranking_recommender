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
import re

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config_optimized import Config
from src.common.utils import setup_logging

logger = logging.getLogger(__name__)

class CorpusBuilder:
    def __init__(self, config: Config):
        self.config = config
        self.data_config = config.data
        self.t5_config = config.generator_t5

    def run(self):
        logger.info("--- 开始步骤 G2: 生成器语料库生成 ---")
        # Set seed at the beginning for reproducibility
        random.seed(self.config.seed)
        logger.info(f"随机种子已设置为 {self.config.seed} 以确保数据处理的可重复性")
        
        semantic_id_map = self._load_semantic_ids()
        playlist_info = self._load_playlist_info()
        playlist_songs = self._load_playlist_songs()
        playlist_filter = self._load_playlist_filter()
        corpus = self._build_corpus(playlist_info, playlist_songs, semantic_id_map, playlist_filter)
        self._split_and_save(corpus)
        logger.info("--- 步骤 G2 成功完成 ---")

    def _load_semantic_ids(self) -> dict:
        logger.info(f"正在从 {self.data_config.semantic_ids_file} 加载语义ID...")
        if not os.path.exists(self.data_config.semantic_ids_file):
            logger.error(f"致命错误: 未找到语义ID文件。请先运行步骤 G1。")
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
                        logger.warning(f"第 {line_num} 行: 缺少必需字段")
                        error_count += 1
                        continue
                    
                    semantic_ids = item['semantic_ids']
                    if not isinstance(semantic_ids, list) or len(semantic_ids) != 3:
                        logger.warning(f"第 {line_num} 行: 无效的 semantic_ids 格式 (期望包含3个整数的列表)")
                        error_count += 1
                        continue
                    
                    mapping[item['song_id']] = semantic_ids
                except json.JSONDecodeError as e:
                    logger.warning(f"第 {line_num} 行: JSON 解码错误 - {e}")
                    error_count += 1
                    continue
        
        logger.info(f"从 {line_count} 行中加载了 {len(mapping)} 个歌曲到语义ID的映射。")
        if error_count > 0:
            logger.warning(f"加载语义ID时遇到 {error_count} 个错误")
        
        if len(mapping) == 0:
            logger.error("致命错误: 未加载到有效的语义ID!")
            sys.exit(1)
        
        return mapping

    def _load_playlist_info(self) -> dict:
        logger.info(f"正在从 {self.data_config.playlist_info_file} 加载歌单信息...")
        try:
            df = pd.read_csv(self.data_config.playlist_info_file, sep='\t')
            df.set_index('glid', inplace=True)
            return df.to_dict('index')
        except FileNotFoundError:
            logger.error(f"致命错误: 在 {self.data_config.playlist_info_file} 未找到歌单信息文件")
            sys.exit(1)

    def _load_playlist_songs(self) -> dict:
        logger.info(f"正在从 {self.data_config.playlist_songs_file} 加载歌单歌曲...")
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
            
            logger.info(f"已加载 {len(playlist_songs)} 个歌单")
            return playlist_songs
        except FileNotFoundError:
            logger.error(f"致命错误: 在 {self.data_config.playlist_songs_file} 未找到歌单歌曲文件")
            sys.exit(1)

    def _load_playlist_filter(self) -> dict:
        """
        Load playlist filter data from llm_filter_prompts.parse_res.
        Returns a dict mapping playlist_id to quality label (Good/Bad/N/A).
        
        File format: <label>\t<playlist_id>\t<title>\t<description>
        Example:
        Good    collection_1_1005752747_558564_0        黑执事专辑      明确指出了影视作品和音乐类型
        Bad     collection_1_1005942407_742147_0        嗨嗨嗨海海海    无意义的重复和无描述性
        """
        filter_file = os.path.join(self.data_config.data_dir, 'llm_filter_prompts.parse_res')
        logger.info(f"正在从 {filter_file} 加载歌单过滤数据...")
        
        if not os.path.exists(filter_file):
            logger.warning(f"在 {filter_file} 未找到歌单过滤文件。跳过过滤。")
            return {}
        
        playlist_filter = {}
        line_count = 0
        error_count = 0
        
        try:
            with open(filter_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line_count += 1
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        parts = line.split('\t')
                        if len(parts) < 2:
                            logger.warning(f"第 {line_num} 行: 无效格式 (期望至少2个制表符分隔的字段)")
                            error_count += 1
                            continue
                        
                        label = parts[0].strip()
                        playlist_id = parts[1].strip()
                        
                        if label not in ['Good', 'Bad', 'N/A']:
                            logger.warning(f"第 {line_num} 行: 歌单 {playlist_id} 的标签 '{label}' 未知")
                            error_count += 1
                            continue
                        
                        playlist_filter[playlist_id] = label
                    except Exception as e:
                        logger.warning(f"第 {line_num} 行: 解析行时出错 - {e}")
                        error_count += 1
                        continue
            
            logger.info(f"从 {line_count} 行中加载了 {len(playlist_filter)} 个歌单的过滤数据。")
            if error_count > 0:
                logger.warning(f"加载过滤数据时遇到 {error_count} 个错误")
            
            # Log statistics
            good_count = sum(1 for label in playlist_filter.values() if label == 'Good')
            bad_count = sum(1 for label in playlist_filter.values() if label == 'Bad')
            na_count = sum(1 for label in playlist_filter.values() if label == 'N/A')
            logger.info(f"过滤统计: Good={good_count}, Bad={bad_count}, N/A={na_count}")
            
        except Exception as e:
            logger.error(f"加载歌单过滤文件时出错: {e}")
            return {}
        
        return playlist_filter

    def _build_corpus(self, playlist_info: dict, playlist_songs: dict, semantic_id_map: dict, playlist_filter: dict) -> list:
        """
        Build text-to-text corpus with layer-specific semantic ID tokens.
        
        Implements a "chunking" strategy for long playlists to ensure all songs
        are used for training.
        
        Filters out playlists marked as 'Bad' or 'N/A' in the playlist_filter.
        """
        logger.info("正在使用分块策略构建文本到文本语料库...")
        corpus = []
        stats = {
            'total_playlists': len(playlist_songs),
            'playlists_without_info': 0,
            'playlists_without_title': 0,
            'playlists_too_few_songs': 0,
            'playlists_filtered_bad': 0,
            'playlists_filtered_na': 0,
            'total_songs': 0,
            'songs_with_semantic_ids': 0,
            'songs_without_semantic_ids': 0,
            'original_lengths': [], # For sequence length analysis
            'total_samples_after_chunking': 0,
        }
        
        max_len = self.t5_config.max_target_length - 1

        for glid, songs in tqdm(playlist_songs.items(), desc="处理歌单中"):
            # Filter out playlists marked as 'Bad' or 'N/A'
            if playlist_filter:
                if glid in playlist_filter:
                    label = playlist_filter[glid]
                    if label == 'Bad':
                        stats['playlists_filtered_bad'] += 1
                        continue
                    elif label == 'N/A':
                        stats['playlists_filtered_na'] += 1
                        continue
            
            if glid not in playlist_info:
                stats['playlists_without_info'] += 1
                continue
            
            if not songs:
                continue

            title = playlist_info[glid].get('listname', '')
            # Ensure title is a string (handle NaN or other non-string types)
            if pd.isna(title) or not isinstance(title, str):
                title = ''
            # Clean special identifiers from title (remove text wrapped in @...@)
            title = re.sub(r'@BI[^@]+@', '', title).strip()
            # Merge multiple consecutive spaces into a single space
            title = re.sub(r'\s+', ' ', title).strip()
            if not title:
                stats['playlists_without_title'] += 1
                continue

            # Per our "embrace collision" strategy, we do not de-duplicate songs or tokens here.
            # MODIFIED: Use random shuffle instead of sorting to prevent model from learning artificial ID order
            sorted_songs = list(songs)
            random.shuffle(sorted_songs)
            semantic_tokens = []
            songs_with_semantic_ids = 0 # Count songs that actually have semantic IDs
            
            stats['total_songs'] += len(sorted_songs)
            
            for song_id in sorted_songs:
                if song_id in semantic_id_map:
                    semantic_ids = semantic_id_map[song_id]
                    tokens = [
                        f"<id_l1_{semantic_ids[0]}>",
                        f"<id_l2_{semantic_ids[1]}>",
                        f"<id_l3_{semantic_ids[2]}>",
                    ]
                    semantic_tokens.extend(tokens)
                    songs_with_semantic_ids += 1
                    stats['songs_with_semantic_ids'] += 1
                else:
                    stats['songs_without_semantic_ids'] += 1
            
            if songs_with_semantic_ids < self.data_config.min_songs_per_playlist:
                stats['playlists_too_few_songs'] += 1
                continue

            if not semantic_tokens:
                continue
            
            stats['original_lengths'].append(len(semantic_tokens))

            # --- Chunking Logic ---
            if len(semantic_tokens) <= max_len:
                # If the sequence is short enough, create one sample.
                output_sequence = " ".join(semantic_tokens) + " <eos>"
                corpus.append((glid, title, output_sequence))
                stats['total_samples_after_chunking'] += 1
            else:
                # If the sequence is too long, split it into chunks.
                for i, chunk_start in enumerate(range(0, len(semantic_tokens), max_len)):
                    chunk_tokens = semantic_tokens[chunk_start : chunk_start + max_len]
                    output_sequence = " ".join(chunk_tokens) + " <eos>"
                    
                    # Create a unique ID for each chunk to avoid duplicates in TSV
                    chunk_glid = f"{glid}_chunk_{i}"
                    corpus.append((chunk_glid, title, output_sequence))
                    stats['total_samples_after_chunking'] += 1
        
        logger.info(f"成功构建了包含 {stats['total_samples_after_chunking']} 条记录的语料库 (分块后)。")
        logger.info("语料库构建统计:")
        logger.info(f"  原始歌单总数: {stats['total_playlists']}")
        logger.info(f"  有效原始歌单数: {len(stats['original_lengths'])}")
        logger.info(f"  生成的训练样本总数 (分块后): {stats['total_samples_after_chunking']}")
        logger.info(f"  过滤的歌单 (Bad): {stats['playlists_filtered_bad']}")
        logger.info(f"  过滤的歌单 (N/A): {stats['playlists_filtered_na']}")
        logger.info(f"  无信息的歌单: {stats['playlists_without_info']}")
        logger.info(f"  无标题的歌单: {stats['playlists_without_title']}")
        logger.info(f"  歌曲数过少的歌单: {stats['playlists_too_few_songs']}")
        logger.info(f"  处理的歌曲总数: {stats['total_songs']}")
        logger.info(f"  有语义ID的歌曲: {stats['songs_with_semantic_ids']} ({stats['songs_with_semantic_ids']/stats['total_songs']*100:.2f}%)" if stats['total_songs'] > 0 else "")
        logger.info(f"  无语义ID的歌曲: {stats['songs_without_semantic_ids']} ({stats['songs_without_semantic_ids']/stats['total_songs']*100:.2f}%)" if stats['total_songs'] > 0 else "")

        # Detailed sequence length analysis (on original lengths)
        if stats['original_lengths']:
            import numpy as np
            lengths = np.array(stats['original_lengths'])
            truncated_count = np.sum(lengths > max_len)
            
            logger.info("--- 原始目标序列长度分析 (分块前) ---")
            logger.info(f"  有效歌单总数: {len(lengths)}")
            logger.info(f"  最小长度: {np.min(lengths)}")
            logger.info(f"  最大长度: {np.max(lengths)}")
            logger.info(f"  平均长度: {np.mean(lengths):.2f}")
            logger.info(f"  中位数长度 (第50百分位): {np.median(lengths)}")
            logger.info(f"  第90百分位: {np.percentile(lengths, 90):.2f}")
            logger.info(f"  第95百分位: {np.percentile(lengths, 95):.2f}")
            logger.info(f"  第99百分位: {np.percentile(lengths, 99):.2f}")
            logger.info("--- 分块影响分析 ---")
            logger.info(f"  每个分块允许的最大长度: {max_len}")
            logger.info(f"  需要分块的原始歌单数: {truncated_count} ({truncated_count/len(lengths)*100:.2f}%)")
        
        if len(corpus) == 0:
            logger.error("致命错误: 未生成有效的语料库条目!")
            sys.exit(1)
        
        return corpus

    def _split_and_save(self, corpus: list):
        logger.info("正在拆分数据并保存到文件...")
        # Shuffle corpus for random split (seed already set in run())
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

        logger.info(f"数据拆分: {len(train_data)} 训练集, {len(val_data)} 验证集, {len(test_data)} 测试集。")
        
        # Validate split ratios
        if len(val_data) == 0:
            logger.warning("验证集为空! 请考虑调整拆分比例。")
        if len(test_data) == 0:
            logger.warning("测试集为空! 请考虑调整拆分比例。")
        
        output_dir = os.path.join(self.config.output_dir, "generator")
        os.makedirs(output_dir, exist_ok=True)
        self._save_to_tsv(train_data, os.path.join(output_dir, "train.tsv"))
        self._save_to_tsv(val_data, os.path.join(output_dir, "val.tsv"))
        self._save_to_tsv(test_data, os.path.join(output_dir, "test.tsv"))

    def _save_to_tsv(self, data: list, file_path: str):
        logger.info(f"正在保存 {len(data)} 条记录到 {file_path}...")
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