"""
高级BadCase分析工具 - 支持多维度分析和可视化 

功能：
1. 关键词搜索和歌单匹配
2. 歌曲频次统计和排序
3. 歌单特征分析（创建者、标签等）
4. 歌曲-歌单关联分析
5. 数据导出和可视化
6. 对比分析（多个关键词）

使用场景：
- 深度分析模型badcase的语料原因
- 对比不同关键词的歌单数据分布
- 识别数据偏差和不平衡问题
- 为模型改进提供数据支持
"""

import os
import sys
import csv
import json
import logging
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Optional
from tqdm import tqdm
from tabulate import tabulate
import numpy as np

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config_optimized import Config
from src.common.utils import setup_logging

logger = logging.getLogger(__name__)


class AdvancedBadCaseAnalyzer:
    """高级BadCase分析工具类"""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_config = config.data
        
        # 数据容器
        self.playlist_info: Dict = {}
        self.playlist_songs: Dict[str, List[str]] = {}
        self.song_info: Dict = {}
        
        # 缓存
        self.analysis_cache: Dict = {}
        
        logger.info("初始化高级BadCase分析工具...")
        self._load_all_data()
    
    def _load_all_data(self):
        """加载所有必要的数据文件"""
        logger.info("=" * 100)
        logger.info("开始加载数据文件...")
        logger.info("=" * 100)
        
        self._load_playlist_info()
        self._load_playlist_songs()
        self._load_song_info()
        
        logger.info("=" * 100)
        logger.info("数据加载完成！")
        logger.info(f"  - 歌单数量: {len(self.playlist_info):,}")
        logger.info(f"  - 歌单-歌曲关系数: {sum(len(songs) for songs in self.playlist_songs.values()):,}")
        logger.info(f"  - 歌曲数量: {len(self.song_info):,}")
        logger.info("=" * 100)
    
    def _load_playlist_info(self):
        """加载歌单信息"""
        logger.info(f"加载歌单信息: {self.data_config.playlist_info_file}")
        try:
            with open(self.data_config.playlist_info_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                
                for row in tqdm(reader, desc="加载歌单信息", unit="条"):
                    glid = str(row.get('glid', ''))
                    if glid:
                        self.playlist_info[glid] = {
                            'glid': glid,
                            'listname': str(row.get('listname', '')),
                            'description': str(row.get('description', '')),
                            'creator': str(row.get('creator', '')),
                            'tags': str(row.get('tags', '')),
                        }
            
            logger.info(f"✓ 成功加载 {len(self.playlist_info):,} 个歌单信息")
        except FileNotFoundError:
            logger.error(f"✗ 歌单信息文件不存在: {self.data_config.playlist_info_file}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"✗ 加载歌单信息失败: {e}")
            sys.exit(1)
    
    def _load_playlist_songs(self):
        """加载歌单-歌曲关系数据"""
        logger.info(f"加载歌单-歌曲数据: {self.data_config.playlist_songs_file}")
        try:
            with open(self.data_config.playlist_songs_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                
                for row in tqdm(reader, desc="加载歌单-歌曲关系", unit="条"):
                    if len(row) < 2:
                        continue
                    
                    playlist_id = str(row[0])
                    song_id = str(row[1])
                    
                    if playlist_id not in self.playlist_songs:
                        self.playlist_songs[playlist_id] = []
                    self.playlist_songs[playlist_id].append(song_id)
            
            logger.info(f"✓ 成功加载 {len(self.playlist_songs):,} 个歌单的歌曲关系")
        except FileNotFoundError:
            logger.error(f"✗ 歌单-歌曲文件不存在: {self.data_config.playlist_songs_file}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"✗ 加载歌单-歌曲数据失败: {e}")
            sys.exit(1)
    
    def _load_song_info(self):
        """加载歌曲信息"""
        logger.info(f"加载歌曲信息: {self.data_config.song_info_file}")
        try:
            with open(self.data_config.song_info_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                
                for row in tqdm(reader, desc="加载歌曲信息", unit="条"):
                    if len(row) < 3:
                        continue
                    
                    song_id = str(row[0]).strip()
                    song_name = str(row[1]).strip()
                    artist = str(row[2]).strip()
                    
                    if song_id:
                        self.song_info[song_id] = {
                            'song_id': song_id,
                            'song_name': song_name,
                            'artist': artist,
                        }
            
            logger.info(f"✓ 成功加载 {len(self.song_info):,} 个歌曲信息")
        except FileNotFoundError:
            logger.error(f"✗ 歌曲信息文件不存在: {self.data_config.song_info_file}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"✗ 加载歌曲信息失败: {e}")
            sys.exit(1)
    
    def search_playlists_by_keyword(self, keyword: str, search_fields: Optional[List[str]] = None) -> List[str]:
        """
        根据关键词搜索歌单
        
        Args:
            keyword: 搜索关键词
            search_fields: 搜索字段列表，默认为['listname', 'description', 'tags']
            
        Returns:
            匹配的歌单ID列表
        """
        if search_fields is None:
            search_fields = ['listname', 'description', 'tags']
        
        keyword_lower = keyword.lower()
        matched_playlists = []
        
        for glid, info in self.playlist_info.items():
            for field in search_fields:
                field_value = info.get(field, '').lower()
                if keyword_lower in field_value:
                    matched_playlists.append(glid)
                    break
        
        return matched_playlists
    
    def analyze_keyword(self, keyword: str, top_n: int = 30, save_results: bool = True) -> Dict:
        """
        分析单个关键词
        
        Args:
            keyword: 搜索关键词
            top_n: 显示前N个高频歌曲
            save_results: 是否保存结果到文件
            
        Returns:
            分析结果字典
        """
        logger.info("=" * 100)
        logger.info(f"开始分析关键词: '{keyword}'")
        logger.info("=" * 100)
        
        # 搜索匹配的歌单
        matched_playlists = self.search_playlists_by_keyword(keyword)
        
        if not matched_playlists:
            logger.warning(f"未找到包含关键词 '{keyword}' 的歌单")
            return {}
        
        logger.info(f"\n✓ 找到 {len(matched_playlists):,} 个包含关键词的歌单")
        
        # 统计歌曲出现频次
        song_frequency = Counter()
        playlist_song_map = defaultdict(set)
        creator_counter = Counter()
        
        for glid in tqdm(matched_playlists, desc="统计歌曲频次", unit="个歌单"):
            if glid in self.playlist_songs:
                songs = self.playlist_songs[glid]
                for song_id in songs:
                    song_frequency[song_id] += 1
                    playlist_song_map[song_id].add(glid)
                
                # 统计创建者
                creator = self.playlist_info.get(glid, {}).get('creator', 'Unknown')
                creator_counter[creator] += 1
        
        logger.info(f"✓ 统计到 {len(song_frequency):,} 首不同的歌曲")
        
        # 按频次排序
        top_songs = song_frequency.most_common(top_n)
        
        # 显示高频歌曲
        self._display_top_songs(top_songs)
        
        # 显示歌单信息
        self._display_playlist_info(matched_playlists)
        
        # 显示创建者统计
        self._display_creator_stats(creator_counter)
        
        # 显示统计摘要
        self._display_summary_stats(keyword, matched_playlists, song_frequency, playlist_song_map)
        
        # 保存结果
        if save_results:
            self._save_analysis_results(keyword, matched_playlists, song_frequency, playlist_song_map)
        
        # 缓存结果
        result = {
            'keyword': keyword,
            'matched_playlists': matched_playlists,
            'song_frequency': song_frequency,
            'playlist_song_map': playlist_song_map,
            'creator_counter': creator_counter,
            'top_songs': top_songs
        }
        self.analysis_cache[keyword] = result
        
        return result
    
    def _display_top_songs(self, top_songs: List[Tuple[str, int]]):
        """显示高频歌曲表格"""
        logger.info("\n" + "=" * 100)
        logger.info(f"高频歌曲排行 (Top {len(top_songs)})")
        logger.info("=" * 100)
        
        table_data = []
        for rank, (song_id, freq) in enumerate(top_songs, 1):
            song_info = self.song_info.get(song_id, {})
            song_name = song_info.get('song_name', 'N/A')
            artist = song_info.get('artist', 'N/A')
            
            table_data.append([
                rank,
                song_id,
                song_name[:30],  # 截断长名称
                artist[:20],
                freq
            ])
        
        headers = ['排名', '歌曲ID', '歌曲名称', '歌手', '出现次数']
        print("\n" + tabulate(table_data, headers=headers, tablefmt='grid'))
    
    def _display_playlist_info(self, matched_playlists: List[str], display_count: int = 20):
        """显示歌单信息"""
        logger.info("\n" + "=" * 100)
        logger.info(f"歌单信息统计 (共 {len(matched_playlists):,} 个歌单，显示前 {display_count} 个)")
        logger.info("=" * 100)
        
        playlist_table_data = []
        for idx, glid in enumerate(matched_playlists[:display_count], 1):
            info = self.playlist_info.get(glid, {})
            listname = info.get('listname', 'N/A')
            song_count = len(self.playlist_songs.get(glid, []))
            creator = info.get('creator', 'N/A')
            
            playlist_table_data.append([
                idx,
                glid,
                listname[:40],
                song_count,
                creator[:15]
            ])
        
        playlist_headers = ['序号', '歌单ID', '歌单标题', '歌曲数', '创建者']
        print("\n" + tabulate(playlist_table_data, headers=playlist_headers, tablefmt='grid'))
        
        if len(matched_playlists) > display_count:
            logger.info(f"\n... 还有 {len(matched_playlists) - display_count:,} 个歌单未显示")
    
    def _display_creator_stats(self, creator_counter: Counter, top_n: int = 10):
        """显示创建者统计"""
        logger.info("\n" + "=" * 100)
        logger.info(f"创建者统计 (Top {top_n})")
        logger.info("=" * 100)
        
        top_creators = creator_counter.most_common(top_n)
        table_data = []
        for rank, (creator, count) in enumerate(top_creators, 1):
            table_data.append([rank, creator[:30], count])
        
        headers = ['排名', '创建者', '歌单数']
        print("\n" + tabulate(table_data, headers=headers, tablefmt='grid'))
    
    def _display_summary_stats(self, keyword: str, matched_playlists: List[str], 
                               song_frequency: Counter, playlist_song_map: Dict):
        """显示统计摘要"""
        logger.info("\n" + "=" * 100)
        logger.info("统计摘要")
        logger.info("=" * 100)
        
        total_songs_in_playlists = sum(len(self.playlist_songs.get(glid, [])) for glid in matched_playlists)
        avg_songs_per_playlist = total_songs_in_playlists / len(matched_playlists) if matched_playlists else 0
        
        # 计算频次分布
        frequencies = list(song_frequency.values())
        
        logger.info(f"关键词: '{keyword}'")
        logger.info(f"匹配歌单数: {len(matched_playlists):,}")
        logger.info(f"不同歌曲数: {len(song_frequency):,}")
        logger.info(f"歌单中歌曲总数: {total_songs_in_playlists:,}")
        logger.info(f"平均每个歌单的歌曲数: {avg_songs_per_playlist:.2f}")
        logger.info(f"最高频歌曲: {song_frequency.most_common(1)[0][0]} (出现 {song_frequency.most_common(1)[0][1]:,} 次)")
        logger.info(f"最低频歌曲: {song_frequency.most_common()[-1][0]} (出现 {song_frequency.most_common()[-1][1]:,} 次)")
        logger.info(f"平均歌曲出现频次: {np.mean(frequencies):.2f}")
        logger.info(f"中位数出现频次: {np.median(frequencies):.2f}")
        logger.info(f"标准差: {np.std(frequencies):.2f}")
    
    def compare_keywords(self, keywords: List[str], top_n: int = 10):
        """
        对比多个关键词的分析结果
        
        Args:
            keywords: 关键词列表
            top_n: 显示前N个高频歌曲
        """
        logger.info("=" * 100)
        logger.info(f"对比分析 {len(keywords)} 个关键词")
        logger.info("=" * 100)
        
        results = {}
        for keyword in keywords:
            results[keyword] = self.analyze_keyword(keyword, top_n=top_n, save_results=False)
        
        # 对比歌单数量
        logger.info("\n" + "=" * 100)
        logger.info("歌单数量对比")
        logger.info("=" * 100)
        
        comparison_data = []
        for keyword in keywords:
            if keyword in results:
                playlist_count = len(results[keyword].get('matched_playlists', []))
                song_count = len(results[keyword].get('song_frequency', {}))
                comparison_data.append([keyword, playlist_count, song_count])
        
        headers = ['关键词', '歌单数', '歌曲数']
        print("\n" + tabulate(comparison_data, headers=headers, tablefmt='grid'))
        
        # 对比高频歌曲
        logger.info("\n" + "=" * 100)
        logger.info("高频歌曲对比")
        logger.info("=" * 100)
        
        for keyword in keywords:
            if keyword in results:
                top_songs = results[keyword].get('top_songs', [])
                logger.info(f"\n{keyword} 的Top {len(top_songs)} 歌曲:")
                for rank, (song_id, freq) in enumerate(top_songs[:5], 1):
                    song_info = self.song_info.get(song_id, {})
                    song_name = song_info.get('song_name', 'N/A')
                    logger.info(f"  {rank}. {song_name} (ID: {song_id}, 频次: {freq:,})")
    
    def _save_analysis_results(self, keyword: str, matched_playlists: List[str], 
                               song_frequency: Counter, playlist_song_map: Dict):
        """保存分析结果到文件"""
        output_dir = os.path.join(self.config.output_dir, "badcase_analysis")
        os.makedirs(output_dir, exist_ok=True)
        
        safe_keyword = "".join(c if c.isalnum() else "_" for c in keyword)
        
        # 1. 保存歌曲频次统计
        songs_output_file = os.path.join(output_dir, f"{safe_keyword}_songs_frequency.csv")
        logger.info(f"\n保存歌曲频次统计...")
        with open(songs_output_file, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=['song_id', 'song_name', 'artist', 'frequency', 'playlist_count'])
            writer.writeheader()
            
            for song_id, freq in tqdm(song_frequency.most_common(), desc="保存歌曲数据", unit="首"):
                song_info = self.song_info.get(song_id, {})
                writer.writerow({
                    'song_id': song_id,
                    'song_name': song_info.get('song_name', ''),
                    'artist': song_info.get('artist', ''),
                    'frequency': freq,
                    'playlist_count': len(playlist_song_map[song_id])
                })
        
        logger.info(f"✓ 歌曲频次统计已保存: {songs_output_file}")
        
        # 2. 保存歌单信息
        playlists_output_file = os.path.join(output_dir, f"{safe_keyword}_playlists_info.csv")
        logger.info(f"保存歌单信息...")
        with open(playlists_output_file, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=['glid', 'listname', 'description', 'creator', 'tags', 'song_count'])
            writer.writeheader()
            
            for glid in tqdm(matched_playlists, desc="保存歌单数据", unit="个"):
                info = self.playlist_info.get(glid, {})
                writer.writerow({
                    'glid': glid,
                    'listname': info.get('listname', ''),
                    'description': info.get('description', ''),
                    'creator': info.get('creator', ''),
                    'tags': info.get('tags', ''),
                    'song_count': len(self.playlist_songs.get(glid, []))
                })
        
        logger.info(f"✓ 歌单信息已保存: {playlists_output_file}")
        
        # 3. 保存详细的歌单-歌曲映射
        mapping_output_file = os.path.join(output_dir, f"{safe_keyword}_playlist_song_mapping.json")
        logger.info(f"保存歌单-歌曲映射...")
        mapping_data = {}
        for glid in tqdm(matched_playlists, desc="保存映射数据", unit="个"):
            info = self.playlist_info.get(glid, {})
            songs = self.playlist_songs.get(glid, [])
            mapping_data[glid] = {
                'listname': info.get('listname', ''),
                'songs': songs,
                'song_count': len(songs)
            }
        
        with open(mapping_output_file, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ 歌单-歌曲映射已保存: {mapping_output_file}")
    
    def interactive_mode(self):
        """交互式模式"""
        logger.info("\n" + "=" * 100)
        logger.info("进入交互式分析模式")
        logger.info("命令:")
        logger.info("  analyze <keyword> [top_n]  - 分析单个关键词")
        logger.info("  compare <keyword1> <keyword2> ... - 对比多个关键词")
        logger.info("  exit/quit - 退出")
        logger.info("=" * 100 + "\n")
        
        while True:
            try:
                user_input = input("请输入命令: ").strip()
                
                if not user_input:
                    continue
                
                parts = user_input.split()
                command = parts[0].lower()
                
                if command in ['exit', 'quit']:
                    logger.info("退出分析工具")
                    break
                
                elif command == 'analyze':
                    if len(parts) < 2:
                        logger.warning("用法: analyze <keyword> [top_n]")
                        continue
                    
                    keyword = parts[1]
                    top_n = int(parts[2]) if len(parts) > 2 else 30
                    
                    self.analyze_keyword(keyword, top_n=top_n)
                
                elif command == 'compare':
                    if len(parts) < 3:
                        logger.warning("用法: compare <keyword1> <keyword2> ...")
                        continue
                    
                    keywords = parts[1:]
                    self.compare_keywords(keywords)
                
                else:
                    logger.warning(f"未知命令: {command}")
                    
            except KeyboardInterrupt:
                logger.info("\n用户中断，退出分析工具")
                break
            except Exception as e:
                logger.error(f"发生错误: {e}")
                continue


def main():
    """主函数"""
    config = Config()
    log_file_path = os.path.join(config.log_dir, "advanced_badcase_analysis.log")
    setup_logging(log_file=log_file_path)
    logger = logging.getLogger(__name__)
    
    # 创建分析工具
    analyzer = AdvancedBadCaseAnalyzer(config)
    
    # 进入交互式模式
    analyzer.interactive_mode()


if __name__ == "__main__":
    main()
