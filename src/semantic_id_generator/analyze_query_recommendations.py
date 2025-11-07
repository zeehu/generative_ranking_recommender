"""
分析查询推荐结果     
从semantic_query_vote_doc.csv加载查询数据，对每个query使用档位0或-1的歌曲，
在歌曲向量中查找距离最近的前20个歌曲，并展示对比结果。
"""

import os
import sys
import json
import csv
import numpy as np
import argparse
import pickle
from typing import Dict, List, Tuple
from tqdm import tqdm
from collections import defaultdict

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("警告: faiss 未安装，将使用较慢的numpy计算。建议安装: pip install faiss-cpu")

# Add project root to sys.path to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config


class QueryRecommendationAnalyzer:
    def __init__(self, config: Config, query_vote_file: str, use_faiss: bool = True):
        """
        初始化分析器
        
        Args:
            config: 配置对象
            query_vote_file: semantic_query_vote_doc.csv 文件路径
            use_faiss: 是否使用FAISS加速检索
        """
        print("--- 查询推荐分析工具 ---")
        self.config = config
        self.query_vote_file = query_vote_file
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        
        # FAISS索引相关
        self.faiss_index = None
        self.song_id_list = None  # 与FAISS索引对应的song_id列表
        self.index_cache_path = os.path.join(config.output_dir, "semantic_id", "faiss_index.bin")
        self.id_list_cache_path = os.path.join(config.output_dir, "semantic_id", "faiss_song_ids.pkl")
        self.index_meta_path = os.path.join(config.output_dir, "semantic_id", "faiss_index_meta.json")
        
        # 加载数据
        self.song_info = self._load_song_info(config.data.song_info_file)
        self.song_vectors = self._load_song_vectors(config.data.song_vectors_file)
        self.semantic_ids = self._load_semantic_ids(config.data.semantic_ids_file)
        self.query_data = self._load_query_vote_data(query_vote_file)
        
        if not self.song_info or not self.song_vectors or not self.semantic_ids:
            raise RuntimeError("加载必要文件失败，请检查 config.py 中的路径配置。")
        
        if not self.query_data:
            raise RuntimeError(f"加载查询数据失败，请检查文件: {query_vote_file}")
        
        # 构建或加载FAISS索引
        if self.use_faiss:
            self._build_or_load_faiss_index()
        
        print("\n初始化完成，分析器已就绪。\n")
    
    def _load_song_info(self, path: str) -> Dict[str, Dict[str, str]]:
        """
        加载歌曲信息
        格式: song_id\tsong_name\tsinger
        """
        print(f"加载歌曲信息: {path}")
        info = {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                for row in tqdm(reader, desc="读取歌曲信息"):
                    if len(row) >= 3:
                        info[row[0]] = {"name": row[1], "singer": row[2]}
            print(f"  ✓ 加载了 {len(info)} 首歌曲信息")
        except FileNotFoundError:
            print(f"  ⚠️  歌曲信息文件未找到: {path}")
        return info
    
    def _load_song_vectors(self, path: str) -> Dict[str, np.ndarray]:
        """
        加载歌曲向量
        格式: song_id,vec1,vec2,...,vecN
        """
        print(f"加载歌曲向量: {path}")
        vectors = {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in tqdm(reader, desc="读取歌曲向量"):
                    if len(row) > 1:
                        vectors[row[0]] = np.array(row[1:], dtype=np.float32)
            print(f"  ✓ 加载了 {len(vectors)} 个歌曲向量 (维度: {len(next(iter(vectors.values())))} )")
        except FileNotFoundError:
            print(f"  ⚠️  歌曲向量文件未找到: {path}")
        return vectors
    
    def _load_semantic_ids(self, path: str) -> Dict[str, Tuple]:
        """
        加载语义ID
        格式: {"song_id": "xxx", "semantic_ids": [1, 2, 3]}
        """
        print(f"加载语义ID: {path}")
        s_ids = {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="读取语义ID"):
                    item = json.loads(line)
                    song_id = item['song_id']
                    sem_id_tuple = tuple(item['semantic_ids'])
                    s_ids[song_id] = sem_id_tuple
            print(f"  ✓ 加载了 {len(s_ids)} 个语义ID")
        except FileNotFoundError:
            print(f"  ⚠️  语义ID文件未找到: {path}")
        return s_ids
    
    def _load_query_vote_data(self, path: str) -> List[Dict]:
        """
        加载查询投票数据
        格式: query,song_infos,search_pv,cnt (CSV格式，逗号分隔)
        song_infos格式: song_id:gear@@song_id:gear@@...
        """
        print(f"加载查询数据: {path}")
        data = []
        
        # 尝试不同的编码
        encodings = ['utf-8', 'gbk', 'gb18030', 'utf-8-sig', 'latin1']
        
        for encoding in encodings:
            try:
                with open(path, 'r', encoding=encoding) as f:
                    # 使用逗号作为分隔符（Excel导出的CSV格式）
                    reader = csv.DictReader(f, delimiter=',')
                    data = []
                    for row in tqdm(reader, desc=f"读取查询数据 (编码: {encoding})"):
                        # 检查必需的字段是否存在
                        if 'query' in row and 'song_infos' in row:
                            data.append({
                                'query': row['query'].strip(),
                                'song_infos': row['song_infos'].strip(),
                                'search_pv': row.get('search_pv', '').strip(),
                                'cnt': row.get('cnt', '').strip()
                            })
                
                if data:
                    print(f"  ✓ 使用 {encoding} 编码成功加载了 {len(data)} 条查询记录")
                    break
                else:
                    print(f"  ⚠️  使用 {encoding} 编码读取成功，但未找到有效数据")
                    
            except UnicodeDecodeError:
                if encoding == encodings[-1]:
                    print(f"  ⚠️  尝试了所有编码 {encodings} 都失败")
                    print(f"  💡 建议：使用以下命令转换文件编码：")
                    print(f"     iconv -f GBK -t UTF-8 {path} > {path}.utf8")
                continue
            except FileNotFoundError:
                print(f"  ⚠️  查询数据文件未找到: {path}")
                break
            except KeyError as e:
                print(f"  ⚠️  CSV文件缺少必需的列: {e}")
                print(f"  💡 期望的列名: query, song_infos, search_pv, cnt")
                break
            except Exception as e:
                print(f"  ⚠️  加载查询数据时出错 (编码: {encoding}): {e}")
                if encoding == encodings[-1]:
                    break
                continue
        
        return data
    
    def parse_song_infos(self, song_infos_str: str, top_n: int = 20) -> List[Tuple[str, int]]:
        """
        解析song_infos字符串
        
        Args:
            song_infos_str: 格式如 "32189764:-1@@27709143:-1@@111055831:-1..."
            top_n: 提取前N个歌曲
            
        Returns:
            [(song_id, gear), ...] 列表
        """
        songs = []
        items = song_infos_str.split('@@')
        
        for item in items[:top_n]:
            if ':' in item:
                parts = item.split(':')
                if len(parts) >= 2:
                    song_id = parts[0].strip()
                    try:
                        gear = int(parts[1])
                        songs.append((song_id, gear))
                    except ValueError:
                        continue
        
        return songs
    
    def get_seed_songs(self, songs: List[Tuple[str, int]]) -> List[str]:
        """
        获取档位为0或-1的歌曲作为种子歌曲
        
        Args:
            songs: [(song_id, gear), ...] 列表
            
        Returns:
            种子歌曲ID列表
        """
        seed_songs = []
        for song_id, gear in songs:
            if gear in [0, -1]:
                seed_songs.append(song_id)
        return seed_songs
    
    def _build_or_load_faiss_index(self):
        """
        构建或加载FAISS索引，确保使用最新的向量文件
        """
        vectors_file = self.config.data.song_vectors_file
        need_rebuild = False
        
        # 检查向量文件是否存在
        if not os.path.exists(vectors_file):
            raise FileNotFoundError(f"向量文件不存在: {vectors_file}")
        
        # 获取向量文件的修改时间
        vectors_mtime = os.path.getmtime(vectors_file)
        
        # 检查是否存在缓存的索引及元数据
        if (os.path.exists(self.index_cache_path) and 
            os.path.exists(self.id_list_cache_path) and 
            os.path.exists(self.index_meta_path)):
            
            # 读取元数据
            try:
                with open(self.index_meta_path, 'r') as f:
                    meta = json.load(f)
                
                cached_mtime = meta.get('vectors_mtime', 0)
                cached_file = meta.get('vectors_file', '')
                
                # 检查向量文件是否更新
                if cached_file != vectors_file:
                    print(f"\n⚠️  向量文件路径已变更: {cached_file} -> {vectors_file}")
                    need_rebuild = True
                elif vectors_mtime > cached_mtime:
                    print(f"\n⚠️  向量文件已更新（缓存时间: {cached_mtime}, 当前时间: {vectors_mtime}）")
                    need_rebuild = True
                else:
                    # 尝试加载缓存的索引
                    print(f"\n加载已缓存的FAISS索引: {self.index_cache_path}")
                    try:
                        self.faiss_index = faiss.read_index(self.index_cache_path)
                        with open(self.id_list_cache_path, 'rb') as f:
                            self.song_id_list = pickle.load(f)
                        print(f"  ✓ 成功加载FAISS索引，包含 {self.faiss_index.ntotal} 个向量")
                        print(f"  ✓ 索引基于向量文件: {cached_file}")
                        return
                    except Exception as e:
                        print(f"  ⚠️  加载索引失败: {e}")
                        need_rebuild = True
            except Exception as e:
                print(f"\n⚠️  读取索引元数据失败: {e}")
                need_rebuild = True
        else:
            print("\n未找到缓存的FAISS索引")
            need_rebuild = True
        
        # 构建新索引
        if need_rebuild:
            print("\n构建FAISS索引...")
            song_ids = list(self.song_vectors.keys())
            vectors = np.array([self.song_vectors[sid] for sid in song_ids], dtype=np.float32)
            
            print(f"  - 歌曲数量: {len(song_ids)}")
            print(f"  - 向量维度: {vectors.shape[1]}")
            
            # 归一化向量（用于余弦相似度）
            faiss.normalize_L2(vectors)
            
            # 创建索引（使用内积，因为向量已归一化，内积等于余弦相似度）
            dimension = vectors.shape[1]
            index = faiss.IndexFlatIP(dimension)  # Inner Product (余弦相似度)
            index.add(vectors)
            
            self.faiss_index = index
            self.song_id_list = song_ids
            
            print(f"  ✓ FAISS索引构建完成，包含 {index.ntotal} 个向量")
            
            # 保存索引到缓存
            print(f"\n保存FAISS索引到: {self.index_cache_path}")
            try:
                os.makedirs(os.path.dirname(self.index_cache_path), exist_ok=True)
                
                # 保存索引文件
                faiss.write_index(self.faiss_index, self.index_cache_path)
                
                # 保存歌曲ID列表
                with open(self.id_list_cache_path, 'wb') as f:
                    pickle.dump(self.song_id_list, f)
                
                # 保存元数据（包含向量文件路径和修改时间）
                meta = {
                    'vectors_file': vectors_file,
                    'vectors_mtime': vectors_mtime,
                    'num_vectors': len(song_ids),
                    'dimension': dimension,
                    'created_at': os.path.getmtime(self.index_cache_path)
                }
                with open(self.index_meta_path, 'w') as f:
                    json.dump(meta, f, indent=2)
                
                print("  ✓ 索引及元数据保存成功")
            except Exception as e:
                print(f"  ⚠️  保存索引失败: {e}")
    
    def calculate_average_vector(self, song_ids: List[str]) -> np.ndarray:
        """
        计算多个歌曲的平均向量
        
        Args:
            song_ids: 歌曲ID列表
            
        Returns:
            平均向量，如果没有有效向量则返回None
        """
        vectors = []
        for song_id in song_ids:
            if song_id in self.song_vectors:
                vectors.append(self.song_vectors[song_id])
        
        if not vectors:
            return None
        
        return np.mean(vectors, axis=0)
    
    def find_nearest_songs(
        self,
        query_vector: np.ndarray,
        exclude_ids: List[str],
        top_n: int = 20
    ) -> List[Tuple[str, float]]:
        """
        找到与查询向量最近的歌曲（使用余弦相似度）
        
        Args:
            query_vector: 查询向量
            exclude_ids: 要排除的歌曲ID列表
            top_n: 返回前N个最近的歌曲
            
        Returns:
            [(song_id, similarity), ...] 列表，按相似度降序排列
        """
        if self.use_faiss and self.faiss_index is not None:
            return self._find_nearest_songs_faiss(query_vector, exclude_ids, top_n)
        else:
            return self._find_nearest_songs_numpy(query_vector, exclude_ids, top_n)
    
    def _find_nearest_songs_faiss(
        self,
        query_vector: np.ndarray,
        exclude_ids: List[str],
        top_n: int = 20
    ) -> List[Tuple[str, float]]:
        """
        使用FAISS找到最近的歌曲
        """
        exclude_set = set(exclude_ids)
        
        # 归一化查询向量
        query_vec = query_vector.copy().reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_vec)
        
        # 搜索更多结果以便过滤排除的歌曲
        k = min(top_n + len(exclude_ids) + 100, self.faiss_index.ntotal)
        distances, indices = self.faiss_index.search(query_vec, k)
        
        # 过滤排除的歌曲并返回top_n
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            song_id = self.song_id_list[idx]
            if song_id not in exclude_set:
                results.append((song_id, float(dist)))
                if len(results) >= top_n:
                    break
        
        return results
    
    def _find_nearest_songs_numpy(
        self,
        query_vector: np.ndarray,
        exclude_ids: List[str],
        top_n: int = 20
    ) -> List[Tuple[str, float]]:
        """
        使用numpy计算最近的歌曲（较慢，用于没有FAISS的情况）
        """
        similarities = []
        exclude_set = set(exclude_ids)
        
        # 归一化查询向量
        query_vector_norm = query_vector / np.linalg.norm(query_vector)
        
        for song_id, vector in self.song_vectors.items():
            if song_id not in exclude_set:
                # 归一化歌曲向量并计算余弦相似度
                vector_norm = vector / np.linalg.norm(vector)
                cosine_sim = np.dot(query_vector_norm, vector_norm)
                similarities.append((song_id, float(cosine_sim)))
        
        # 按相似度降序排序并返回前N个
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]
    
    def get_song_display_info(self, song_id: str) -> str:
        """
        获取歌曲的展示信息
        
        Args:
            song_id: 歌曲ID
            
        Returns:
            格式化的歌曲信息字符串
        """
        info = self.song_info.get(song_id, {"name": "未知", "singer": "未知"})
        semantic_id = self.semantic_ids.get(song_id, ())
        semantic_id_str = str(semantic_id) if semantic_id else "N/A"
        return f"ID:{song_id} | {info['name']} - {info['singer']} | 语义ID:{semantic_id_str}"
    
    def analyze_query(self, query_data: Dict, query_idx: int = 0):
        """
        分析单个查询
        
        Args:
            query_data: 查询数据字典
            query_idx: 查询索引（用于显示）
        """
        query = query_data['query']
        song_infos_str = query_data['song_infos']
        
        # 1. 解析前20个歌曲及其档位
        top_20_songs = self.parse_song_infos(song_infos_str, top_n=20)
        
        # 2. 获取种子歌曲（档位0或-1）
        seed_songs = self.get_seed_songs(top_20_songs)
        
        # 3. 计算种子歌曲的平均向量并检索
        nearest_songs = []
        if seed_songs:
            query_vector = self.calculate_average_vector(seed_songs)
            if query_vector is not None:
                exclude_ids = [song_id for song_id, _ in top_20_songs]
                nearest_songs = self.find_nearest_songs(query_vector, exclude_ids, top_n=20)
        
        # 4. 并列展示结果
        self._display_side_by_side(query, query_idx, top_20_songs, seed_songs, nearest_songs)
    
    def _display_side_by_side(
        self, 
        query: str, 
        query_idx: int, 
        original_songs: List[Tuple[str, int]], 
        seed_songs: List[str],
        vector_songs: List[Tuple[str, float]]
    ):
        """
        并列展示原始推荐和向量检索结果
        
        Args:
            query: 查询文本
            query_idx: 查询索引
            original_songs: 原始推荐歌曲列表 [(song_id, gear), ...]
            seed_songs: 种子歌曲ID列表
            vector_songs: 向量检索歌曲列表 [(song_id, similarity), ...]
        """
        # 打印查询
        print("\n" + "=" * 220)
        print(f"查询 #{query_idx + 1}: {query} (种子歌曲数: {len(seed_songs)})")
        print("=" * 220)
        
        # 表头
        left_header = "【原始推荐 Top20】"
        right_header = "【向量检索 Top20】"
        print(f"\n{left_header:<100} | {right_header}")
        print("-" * 100 + " | " + "-" * 100)
        
        # 并列展示20行
        for i in range(20):
            # 左侧：原始推荐
            if i < len(original_songs):
                song_id, gear = original_songs[i]
                info = self.song_info.get(song_id, {"name": "未知", "singer": "未知"})
                semantic_id = self.semantic_ids.get(song_id, ())
                sem_str = str(semantic_id) if semantic_id else "N/A"
                
                # 标记种子歌曲
                marker = "✓" if gear in [0, -1] else " "
                left_line = f"{i+1:2d}.[档{gear:2d}]{marker} ID:{song_id:<10} {info['name'][:12]:<12} - {info['singer'][:8]:<8} {sem_str}"
            else:
                left_line = ""
            
            # 右侧：向量检索
            if i < len(vector_songs):
                song_id, similarity = vector_songs[i]
                info = self.song_info.get(song_id, {"name": "未知", "singer": "未知"})
                semantic_id = self.semantic_ids.get(song_id, ())
                sem_str = str(semantic_id) if semantic_id else "N/A"
                
                right_line = f"{i+1:2d}.[{similarity:.3f}] ID:{song_id:<10} {info['name'][:12]:<12} - {info['singer'][:8]:<8} {sem_str}"
            else:
                right_line = ""
            
            # 打印一行
            print(f"{left_line:<100} | {right_line}")
        
        print()
    
    def analyze_all_queries(self, max_queries: int = None):
        """
        分析所有查询
        
        Args:
            max_queries: 最多分析的查询数量，None表示分析所有
        """
        total_queries = len(self.query_data)
        if max_queries:
            total_queries = min(total_queries, max_queries)
        
        print(f"\n开始分析 {total_queries} 个查询...\n")
        
        for idx in range(total_queries):
            self.analyze_query(self.query_data[idx], idx)
    
    def run(self, max_queries: int = None):
        """
        运行完整的分析流程
        
        Args:
            max_queries: 最多分析的查询数量
        """
        self.analyze_all_queries(max_queries)


def main():
    parser = argparse.ArgumentParser(description='分析查询推荐结果')
    parser.add_argument(
        '--query_vote',
        type=str,
        default='semantic_query_vote_doc.csv',
        help='查询投票数据文件路径'
    )
    parser.add_argument(
        '--max_queries',
        type=int,
        default=None,
        help='最多分析的查询数量（默认分析所有）'
    )
    parser.add_argument(
        '--no_faiss',
        action='store_true',
        help='不使用FAISS加速（使用numpy计算）'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config = Config()
    
    # 创建分析器并运行
    analyzer = QueryRecommendationAnalyzer(
        config=config,
        query_vote_file=args.query_vote,
        use_faiss=not args.no_faiss
    )
    
    analyzer.run(max_queries=args.max_queries)


if __name__ == '__main__':
    main()
