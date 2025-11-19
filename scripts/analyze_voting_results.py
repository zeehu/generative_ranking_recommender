"""
投票结果分析工具     

功能：
1. 加载投票结果数据，统计现有歌曲文件对投票结果中的歌曲覆盖率
2. 从投票结果中随机采样10000条数据，将语义query输入t5模型进行推理
3. 对比t5模型生成结果和投票结果（歌曲维度）
4. 对比t5模型生成结果和投票结果（语义ID维度）
"""

import os
import sys
import json
import random
import logging
import argparse
import numpy as np
from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter
from tqdm import tqdm

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config
from src.generator.inference_t5 import PlaylistGenerator
from src.common.utils import setup_logging

logger = logging.getLogger(__name__)


class VotingResultAnalyzer:
    """投票结果分析器"""
    
    def __init__(self, config: Config, voting_file: str, model_path: str = None):
        """
        初始化分析器
        
        Args:
            config: 配置对象
            voting_file: 投票结果文件路径
            model_path: T5模型路径
        """
        self.config = config
        self.voting_file = voting_file
        
        # 加载歌曲信息（歌名、歌手）
        self.song_info_map = self._load_song_info()
        logger.info(f"已加载 {len(self.song_info_map)} 首歌曲的信息")
        
        # 加载现有歌曲集合
        self.existing_songs = self._load_existing_songs()
        logger.info(f"已加载 {len(self.existing_songs)} 首现有歌曲")
        
        # 加载投票结果
        self.voting_data = self._load_voting_data()
        logger.info(f"已加载 {len(self.voting_data)} 条投票结果")
        
        # 加载语义ID映射
        self.song_to_semantic_id = self._load_semantic_id_mapping()
        logger.info(f"已加载 {len(self.song_to_semantic_id)} 首歌曲的语义ID映射")
        
        # 初始化T5生成器
        logger.info("正在初始化T5模型...")
        self.generator = PlaylistGenerator(config, model_path=model_path, use_trie_constraint=True)
        logger.info("T5模型初始化完成")
    
    def _load_song_info(self) -> Dict[str, Dict[str, str]]:
        """加载歌曲信息（歌名、歌手）"""
        import csv
        mapping = {}
        try:
            with open(self.config.data.song_info_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    if len(row) >= 3:
                        mapping[row[0]] = {"name": row[1], "singer": row[2]}
            logger.info(f"已从 {self.config.data.song_info_file} 加载歌曲信息")
        except FileNotFoundError:
            logger.warning(f"歌曲信息文件未找到: {self.config.data.song_info_file}")
        return mapping
    
    def _load_existing_songs(self) -> Set[str]:
        """加载现有歌曲集合"""
        songs = set()
        
        # 从语义ID文件加载
        semantic_ids_file = os.path.join(
            self.config.output_dir, "semantic_id", "song_semantic_ids.jsonl"
        )
        
        if os.path.exists(semantic_ids_file):
            with open(semantic_ids_file, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    songs.add(item['song_id'])
        else:
            logger.warning(f"语义ID文件不存在: {semantic_ids_file}")
        
        return songs
    
    def _load_voting_data(self) -> List[Dict]:
        """
        加载投票结果数据
        
        返回格式：
        [
            {
                'query': '2019电音神曲',
                'songs': [
                    {'song_id': '39667496', 'vote': 0},
                    {'song_id': '54003766', 'vote': 0},
                    ...
                ]
            },
            ...
        ]
        """
        voting_data = []
        
        with open(self.voting_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) != 2:
                    logger.warning(f"跳过格式错误的行: {line[:100]}")
                    continue
                
                query = parts[0]
                song_votes = parts[1]
                
                songs = []
                for item in song_votes.split(','):
                    try:
                        song_id, vote = item.split(':')
                        songs.append({
                            'song_id': song_id,
                            'vote': int(vote)
                        })
                    except ValueError:
                        logger.warning(f"跳过格式错误的歌曲投票: {item}")
                        continue
                
                voting_data.append({
                    'query': query,
                    'songs': songs
                })
        
        return voting_data
    
    def _load_semantic_id_mapping(self) -> Dict[str, Tuple[int, ...]]:
        """加载歌曲ID到语义ID的映射"""
        mapping = {}
        
        semantic_ids_file = os.path.join(
            self.config.output_dir, "semantic_id", "song_semantic_ids.jsonl"
        )
        
        if os.path.exists(semantic_ids_file):
            with open(semantic_ids_file, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    mapping[item['song_id']] = tuple(item['semantic_ids'])
        
        return mapping
    
    def analyze_coverage(self):
        """分析现有歌曲文件对投票结果的覆盖率"""
        logger.info("\n" + "="*80)
        logger.info("1. 分析歌曲覆盖率")
        logger.info("="*80)
        
        total_songs_in_voting = set()
        covered_songs = set()
        
        for item in self.voting_data:
            for song in item['songs']:
                song_id = song['song_id']
                total_songs_in_voting.add(song_id)
                if song_id in self.existing_songs:
                    covered_songs.add(song_id)
        
        coverage_rate = len(covered_songs) / len(total_songs_in_voting) * 100 if total_songs_in_voting else 0
        
        logger.info(f"投票结果中的歌曲总数: {len(total_songs_in_voting)}")
        logger.info(f"现有歌曲文件中的歌曲数: {len(self.existing_songs)}")
        logger.info(f"覆盖的歌曲数: {len(covered_songs)}")
        logger.info(f"覆盖率: {coverage_rate:.2f}%")
        
        # 过滤投票数据，只保留现有歌曲
        filtered_voting_data = []
        for item in self.voting_data:
            filtered_songs = [
                song for song in item['songs'] 
                if song['song_id'] in self.existing_songs
            ]
            if filtered_songs:
                filtered_voting_data.append({
                    'query': item['query'],
                    'songs': filtered_songs
                })
        
        logger.info(f"过滤后的投票数据条数: {len(filtered_voting_data)}")
        
        return filtered_voting_data, coverage_rate
    
    def sample_and_inference(self, filtered_voting_data: List[Dict], sample_size: int = 10000, seed: int = 42):
        """
        从投票结果中随机采样并进行T5推理
        
        Args:
            filtered_voting_data: 过滤后的投票数据
            sample_size: 采样数量
            seed: 随机种子，确保结果可复现
        
        Returns:
            采样数据和推理结果
        """
        logger.info("\n" + "="*80)
        logger.info(f"2. 随机采样 {sample_size} 条数据并进行T5推理（随机种子={seed}）")
        logger.info("="*80)
        
        # 设置随机种子以确保采样结果可复现
        random.seed(seed)
        
        # 随机采样
        if len(filtered_voting_data) > sample_size:
            sampled_data = random.sample(filtered_voting_data, sample_size)
            logger.info(f"从 {len(filtered_voting_data)} 条数据中采样 {sample_size} 条")
        else:
            sampled_data = filtered_voting_data
            logger.warning(f"数据量不足 {sample_size}，使用全部 {len(sampled_data)} 条数据")
        
        logger.info(f"实际采样数量: {len(sampled_data)}")
        
        # 进行推理
        inference_results = []
        logger.info("开始T5推理...")
        
        for item in tqdm(sampled_data, desc="推理进度"):
            query = item['query']
            
            # 使用T5生成歌单
            try:
                generated_results = self.generator.generate(
                    query, 
                    max_songs=50,  # 生成更多歌曲以便对比
                    temperature=0.8
                )
                
                # 提取生成的歌曲ID和语义ID
                generated_songs = [r['primary_song_id'] for r in generated_results]
                generated_semantic_ids = [r['semantic_id'] for r in generated_results]
                
                inference_results.append({
                    'query': query,
                    'voting_songs': item['songs'],
                    'generated_songs': generated_songs,
                    'generated_semantic_ids': generated_semantic_ids
                })
            except Exception as e:
                logger.error(f"推理失败 (query: {query}): {e}")
                inference_results.append({
                    'query': query,
                    'voting_songs': item['songs'],
                    'generated_songs': [],
                    'generated_semantic_ids': []
                })
        
        logger.info(f"推理完成，成功 {len([r for r in inference_results if r['generated_songs']])} 条")
        
        return inference_results
    
    def compare_songs(self, inference_results: List[Dict]):
        """
        从歌曲维度对比T5生成结果和投票结果
        
        Args:
            inference_results: 推理结果列表
        """
        logger.info("\n" + "="*80)
        logger.info("3. 歌曲维度对比分析")
        logger.info("="*80)
        
        # 计算每条数据的重合度
        overlap_scores = []
        
        for result in inference_results:
            voting_song_ids = set([s['song_id'] for s in result['voting_songs']])
            generated_song_ids = set(result['generated_songs'])
            
            if not voting_song_ids or not generated_song_ids:
                continue
            
            # 计算交集
            overlap = voting_song_ids & generated_song_ids
            
            # 计算重合度（Jaccard相似度）
            jaccard = len(overlap) / len(voting_song_ids | generated_song_ids)
            
            # 计算前N的精确率
            top_n_voting = [s['song_id'] for s in result['voting_songs'][:20]]
            top_n_generated = result['generated_songs'][:20]
            top_n_overlap = len(set(top_n_voting) & set(top_n_generated))
            precision_at_20 = top_n_overlap / len(top_n_generated) if top_n_generated else 0
            
            overlap_scores.append({
                'query': result['query'],
                'jaccard': jaccard,
                'precision_at_20': precision_at_20,
                'overlap_count': len(overlap),
                'voting_count': len(voting_song_ids),
                'generated_count': len(generated_song_ids),
                'voting_songs': result['voting_songs'],
                'generated_songs': result['generated_songs']
            })
        
        # 排序
        overlap_scores.sort(key=lambda x: x['jaccard'], reverse=True)
        
        # 统计平均重合度
        avg_jaccard = np.mean([s['jaccard'] for s in overlap_scores]) if overlap_scores else 0
        avg_precision = np.mean([s['precision_at_20'] for s in overlap_scores]) if overlap_scores else 0
        
        logger.info(f"平均Jaccard相似度: {avg_jaccard:.4f}")
        logger.info(f"平均Top-20精确率: {avg_precision:.4f}")
        
        # 打印指标说明
        logger.info("\n" + "-"*80)
        logger.info("指标说明:")
        logger.info("-"*80)
        logger.info("1. Jaccard相似度 = |投票歌曲 ∩ 生成歌曲| / |投票歌曲 ∪ 生成歌曲|")
        logger.info("   - 衡量两个歌曲集合的整体相似度")
        logger.info("   - 范围: [0, 1]，越高表示重合度越高")
        logger.info("   - 例如: 投票100首，生成50首，重合20首 → Jaccard = 20/(100+50-20) = 0.154")
        logger.info("")
        logger.info("2. Top-20精确率 = |投票Top20 ∩ 生成Top20| / 20")
        logger.info("   - 衡量生成的前20首歌曲中有多少在投票的前20首中")
        logger.info("   - 范围: [0, 1]，越高表示排序质量越好")
        logger.info("   - 例如: 生成的前20首中有5首在投票前20中 → P@20 = 5/20 = 0.25")
        
        # 打印最好的10条
        logger.info("\n" + "-"*80)
        logger.info("最好的10条结果:")
        logger.info("-"*80)
        for i, score in enumerate(overlap_scores[:10], 1):
            logger.info(f"\n[{i}] Query: {score['query']}")
            logger.info(f"    Jaccard: {score['jaccard']:.4f}, P@20: {score['precision_at_20']:.4f}")
            logger.info(f"    重合: {score['overlap_count']}, 投票: {score['voting_count']}, 生成: {score['generated_count']}")
            
            # 显示前10首歌曲对比（带歌名、歌手、vote值/语义ID）
            logger.info(f"\n    投票Top10:")
            for j, song in enumerate(score['voting_songs'][:10], 1):
                song_id = song['song_id']
                vote = song['vote']
                info = self.song_info_map.get(song_id, {"name": "未知", "singer": "未知"})
                semantic_id = self.song_to_semantic_id.get(song_id, (0, 0, 0))
                logger.info(f"      {j:2d}. {song_id} - {info['name']} - {info['singer']} (vote={vote}, sid={semantic_id})")
            
            logger.info(f"\n    生成Top10:")
            for j, song_id in enumerate(score['generated_songs'][:10], 1):
                info = self.song_info_map.get(song_id, {"name": "未知", "singer": "未知"})
                semantic_id = self.song_to_semantic_id.get(song_id, (0, 0, 0))
                logger.info(f"      {j:2d}. {song_id} - {info['name']} - {info['singer']} (sid={semantic_id})")
        
        # 打印最差的10条
        logger.info("\n" + "-"*80)
        logger.info("最差的10条结果:")
        logger.info("-"*80)
        for i, score in enumerate(overlap_scores[-10:], 1):
            logger.info(f"\n[{i}] Query: {score['query']}")
            logger.info(f"    Jaccard: {score['jaccard']:.4f}, P@20: {score['precision_at_20']:.4f}")
            logger.info(f"    重合: {score['overlap_count']}, 投票: {score['voting_count']}, 生成: {score['generated_count']}")
            
            # 显示前10首歌曲对比（带歌名、歌手、vote值/语义ID）
            logger.info(f"\n    投票Top10:")
            for j, song in enumerate(score['voting_songs'][:10], 1):
                song_id = song['song_id']
                vote = song['vote']
                info = self.song_info_map.get(song_id, {"name": "未知", "singer": "未知"})
                semantic_id = self.song_to_semantic_id.get(song_id, (0, 0, 0))
                logger.info(f"      {j:2d}. {song_id} - {info['name']} - {info['singer']} (vote={vote}, sid={semantic_id})")
            
            logger.info(f"\n    生成Top10:")
            for j, song_id in enumerate(score['generated_songs'][:10], 1):
                info = self.song_info_map.get(song_id, {"name": "未知", "singer": "未知"})
                semantic_id = self.song_to_semantic_id.get(song_id, (0, 0, 0))
                logger.info(f"      {j:2d}. {song_id} - {info['name']} - {info['singer']} (sid={semantic_id})")
        
        return overlap_scores
    
    def compare_semantic_ids(self, inference_results: List[Dict]):
        """
        从语义ID维度对比T5生成结果和投票结果，使用Jaccard相似度
        
        Args:
            inference_results: 推理结果列表
        """
        logger.info("\n" + "="*80)
        logger.info("4. 语义ID维度对比分析")
        logger.info("="*80)
        
        # 计算每条数据的语义ID Jaccard相似度
        semantic_scores = []
        
        for result in inference_results:
            # 将投票结果转换为语义ID序列
            voting_semantic_ids = []
            for song in result['voting_songs']:
                song_id = song['song_id']
                if song_id in self.song_to_semantic_id:
                    voting_semantic_ids.append(self.song_to_semantic_id[song_id])
            
            generated_semantic_ids = result['generated_semantic_ids']
            
            if not voting_semantic_ids or not generated_semantic_ids:
                continue
            
            # 转换为集合
            voting_set = set(voting_semantic_ids)
            generated_set = set(generated_semantic_ids)
            
            # 计算完全匹配的Jaccard相似度（3层都要相同）
            intersection = voting_set & generated_set
            union = voting_set | generated_set
            jaccard_full = len(intersection) / len(union) if union else 0
            
            # 计算前2层的Jaccard相似度
            voting_l1_l2 = set([(sid[0], sid[1]) for sid in voting_semantic_ids])
            generated_l1_l2 = set([(sid[0], sid[1]) for sid in generated_semantic_ids])
            intersection_l1_l2 = voting_l1_l2 & generated_l1_l2
            union_l1_l2 = voting_l1_l2 | generated_l1_l2
            jaccard_l1_l2 = len(intersection_l1_l2) / len(union_l1_l2) if union_l1_l2 else 0
            
            # 计算第1层的Jaccard相似度
            voting_l1 = set([sid[0] for sid in voting_semantic_ids])
            generated_l1 = set([sid[0] for sid in generated_semantic_ids])
            intersection_l1 = voting_l1 & generated_l1
            union_l1 = voting_l1 | generated_l1
            jaccard_l1 = len(intersection_l1) / len(union_l1) if union_l1 else 0
            
            semantic_scores.append({
                'query': result['query'],
                'jaccard_full': jaccard_full,
                'jaccard_l1_l2': jaccard_l1_l2,
                'jaccard_l1': jaccard_l1,
                'voting_semantic_ids': voting_semantic_ids,
                'generated_semantic_ids': generated_semantic_ids,
                'voting_songs': result['voting_songs'],
                'generated_songs': result['generated_songs']
            })
        
        # 排序（按完全匹配的Jaccard排序）
        semantic_scores.sort(key=lambda x: x['jaccard_full'], reverse=True)
        
        # 统计平均值
        avg_jaccard_full = np.mean([s['jaccard_full'] for s in semantic_scores]) if semantic_scores else 0
        avg_jaccard_l1_l2 = np.mean([s['jaccard_l1_l2'] for s in semantic_scores]) if semantic_scores else 0
        avg_jaccard_l1 = np.mean([s['jaccard_l1'] for s in semantic_scores]) if semantic_scores else 0
        
        logger.info(f"有效对比数量: {len(semantic_scores)}")
        logger.info(f"平均Jaccard相似度（3层完全匹配）: {avg_jaccard_full:.4f}")
        logger.info(f"平均Jaccard相似度（前2层匹配）: {avg_jaccard_l1_l2:.4f}")
        logger.info(f"平均Jaccard相似度（第1层匹配）: {avg_jaccard_l1:.4f}")
        
        # 打印指标说明
        logger.info("\n" + "-"*80)
        logger.info("指标说明:")
        logger.info("-"*80)
        logger.info("语义ID Jaccard相似度 = |投票语义IDs ∩ 生成语义IDs| / |投票语义IDs ∪ 生成语义IDs|")
        logger.info("  - 3层完全匹配: 比较完整的语义ID (L1, L2, L3)")
        logger.info("  - 前2层匹配: 只比较 (L1, L2)，忽略L3")
        logger.info("  - 第1层匹配: 只比较 L1，忽略L2和L3")
        logger.info("  - 范围: [0, 1]，越高表示语义理解越准确")
        
        # 打印最好的10条
        logger.info("\n" + "-"*80)
        logger.info("语义ID最好的10条结果:")
        logger.info("-"*80)
        for i, score in enumerate(semantic_scores[:10], 1):
            logger.info(f"\n[{i}] Query: {score['query']}")
            logger.info(f"    Jaccard(3层): {score['jaccard_full']:.4f}, Jaccard(L1+L2): {score['jaccard_l1_l2']:.4f}, Jaccard(L1): {score['jaccard_l1']:.4f}")
            logger.info(f"    投票语义ID数: {len(set(score['voting_semantic_ids']))}, 生成语义ID数: {len(set(score['generated_semantic_ids']))}")
            
            # 显示投票结果的语义ID分布（Top10）
            logger.info(f"\n    投票Top10语义ID:")
            for j, song in enumerate(score['voting_songs'][:10], 1):
                song_id = song['song_id']
                vote = song['vote']
                info = self.song_info_map.get(song_id, {"name": "未知", "singer": "未知"})
                semantic_id = self.song_to_semantic_id.get(song_id, (0, 0, 0))
                logger.info(f"      {j:2d}. {semantic_id} - {song_id} - {info['name']} - {info['singer']} (vote={vote})")
            
            # 显示生成结果的语义ID分布（Top10）
            logger.info(f"\n    生成Top10语义ID:")
            for j, song_id in enumerate(score['generated_songs'][:10], 1):
                info = self.song_info_map.get(song_id, {"name": "未知", "singer": "未知"})
                semantic_id = self.song_to_semantic_id.get(song_id, (0, 0, 0))
                logger.info(f"      {j:2d}. {semantic_id} - {song_id} - {info['name']} - {info['singer']}")
        
        # 打印最差的10条
        logger.info("\n" + "-"*80)
        logger.info("语义ID最差的10条结果:")
        logger.info("-"*80)
        for i, score in enumerate(semantic_scores[-10:], 1):
            logger.info(f"\n[{i}] Query: {score['query']}")
            logger.info(f"    Jaccard(3层): {score['jaccard_full']:.4f}, Jaccard(L1+L2): {score['jaccard_l1_l2']:.4f}, Jaccard(L1): {score['jaccard_l1']:.4f}")
            logger.info(f"    投票语义ID数: {len(set(score['voting_semantic_ids']))}, 生成语义ID数: {len(set(score['generated_semantic_ids']))}")
            
            # 显示投票结果的语义ID分布（Top10）
            logger.info(f"\n    投票Top10语义ID:")
            for j, song in enumerate(score['voting_songs'][:10], 1):
                song_id = song['song_id']
                vote = song['vote']
                info = self.song_info_map.get(song_id, {"name": "未知", "singer": "未知"})
                semantic_id = self.song_to_semantic_id.get(song_id, (0, 0, 0))
                logger.info(f"      {j:2d}. {semantic_id} - {song_id} - {info['name']} - {info['singer']} (vote={vote})")
            
            # 显示生成结果的语义ID分布（Top10）
            logger.info(f"\n    生成Top10语义ID:")
            for j, song_id in enumerate(score['generated_songs'][:10], 1):
                info = self.song_info_map.get(song_id, {"name": "未知", "singer": "未知"})
                semantic_id = self.song_to_semantic_id.get(song_id, (0, 0, 0))
                logger.info(f"      {j:2d}. {semantic_id} - {song_id} - {info['name']} - {info['singer']}")
        
        return {
            'total_count': len(semantic_scores),
            'avg_jaccard_full': float(avg_jaccard_full),
            'avg_jaccard_l1_l2': float(avg_jaccard_l1_l2),
            'avg_jaccard_l1': float(avg_jaccard_l1)
        }
    
    def run_analysis(self, sample_size: int = 10000, seed: int = 42):
        """运行完整的分析流程"""
        logger.info("开始投票结果分析...")
        
        # 1. 分析覆盖率
        filtered_voting_data, coverage_rate = self.analyze_coverage()
        
        # 2. 采样并推理
        inference_results = self.sample_and_inference(filtered_voting_data, sample_size, seed)
        
        # 3. 歌曲维度对比
        song_comparison = self.compare_songs(inference_results)
        
        # 4. 语义ID维度对比
        semantic_id_comparison = self.compare_semantic_ids(inference_results)
        
        # 保存结果
        output_file = os.path.join(self.config.output_dir, "voting_analysis_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'coverage_rate': coverage_rate,
                'sample_size': len(inference_results),
                'semantic_id_comparison': semantic_id_comparison,
                'song_comparison_summary': {
                    'avg_jaccard': float(np.mean([s['jaccard'] for s in song_comparison])) if song_comparison else 0,
                    'avg_precision_at_20': float(np.mean([s['precision_at_20'] for s in song_comparison])) if song_comparison else 0
                }
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n分析结果已保存到: {output_file}")
        logger.info("\n分析完成！")


def main():
    parser = argparse.ArgumentParser(description="投票结果分析工具")
    parser.add_argument(
        "-f", "--voting_file",
        type=str,
        default="data/query_tag_map_top_mixsongid_rain_2025-11-17.txt",
        help="投票结果文件路径"
    )
    parser.add_argument(
        "-m", "--model_path",
        type=str,
        default=None,
        help="T5模型路径（默认使用配置文件中的路径）"
    )
    parser.add_argument(
        "-s", "--sample_size",
        type=int,
        default=10000,
        help="采样数量（默认10000）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认42）"
    )
    parser.add_argument(
        "-l", "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别（默认INFO）"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    log_level = getattr(logging, args.log_level)
    setup_logging(level=log_level)
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 检查投票文件是否存在
    if not os.path.exists(args.voting_file):
        logger.error(f"投票结果文件不存在: {args.voting_file}")
        logger.error("请确保文件路径正确")
        sys.exit(1)
    
    # 加载配置
    config = Config()
    
    # 创建分析器并运行
    analyzer = VotingResultAnalyzer(config, args.voting_file, args.model_path)
    analyzer.run_analysis(sample_size=args.sample_size, seed=args.seed)


if __name__ == "__main__":
    main()
