"""
Trie树实现，用于约束T5模型生成有效的语义ID序列
支持三层语义ID结构：(l1, l2, l3)
"""
import json
import logging
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class TrieNode:
    """Trie树节点"""
    
    def __init__(self):
        self.children = {}  # key: token_id, value: TrieNode
        self.is_end = False  # 标记是否是完整语义ID序列的结束
        self.depth = 0  # 当前节点深度（0=root, 1=l1, 2=l2, 3=l3）


class SemanticIDTrie:
    """
    语义ID的Trie树，用于约束生成
    
    支持三层语义ID结构，每个完整的语义ID由3个token组成：
    <id_l1_X> <id_l2_Y> <id_l3_Z>
    """
    
    def __init__(self, tokenizer, semantic_ids_file: str):
        """
        初始化Trie树
        
        Args:
            tokenizer: TIGER tokenizer实例
            semantic_ids_file: 语义ID文件路径 (JSONL格式)
        """
        self.tokenizer = tokenizer
        self.root = TrieNode()
        self.valid_semantic_ids = set()  # 存储所有有效的语义ID元组
        
        # 构建Trie树
        self._build_trie(semantic_ids_file)
        
        logger.info(f"Trie树构建完成，包含 {len(self.valid_semantic_ids)} 个有效语义ID序列")
    
    def _build_trie(self, semantic_ids_file: str):
        """
        从语义ID文件构建Trie树
        
        Args:
            semantic_ids_file: 语义ID文件路径
        """
        logger.info(f"正在从 {semantic_ids_file} 构建Trie树...")
        
        with open(semantic_ids_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                semantic_ids = item['semantic_ids']
                
                if len(semantic_ids) != 3:
                    logger.warning(f"跳过无效的语义ID: {semantic_ids} (长度不为3)")
                    continue
                
                # 将语义ID转换为token ID序列
                l1_id, l2_id, l3_id = semantic_ids
                token_sequence = self._semantic_ids_to_token_ids(l1_id, l2_id, l3_id)
                
                if token_sequence is None:
                    continue
                
                # 插入到Trie树
                self._insert(token_sequence)
                
                # 记录有效的语义ID
                self.valid_semantic_ids.add(tuple(semantic_ids))
        
        logger.info(f"Trie树构建完成，共插入 {len(self.valid_semantic_ids)} 个有效序列")
    
    def _semantic_ids_to_token_ids(self, l1_id: int, l2_id: int, l3_id: int) -> Optional[List[int]]:
        """
        将语义ID三元组转换为token ID序列
        
        Args:
            l1_id: 第一层ID
            l2_id: 第二层ID
            l3_id: 第三层ID
            
        Returns:
            token ID列表，如果转换失败则返回None
        """
        try:
            l1_token = f"<id_l1_{l1_id}>"
            l2_token = f"<id_l2_{l2_id}>"
            l3_token = f"<id_l3_{l3_id}>"
            
            # 转换为token ID
            l1_token_id = self.tokenizer.base_tokenizer.convert_tokens_to_ids(l1_token)
            l2_token_id = self.tokenizer.base_tokenizer.convert_tokens_to_ids(l2_token)
            l3_token_id = self.tokenizer.base_tokenizer.convert_tokens_to_ids(l3_token)
            
            # 检查是否成功转换（未知token会返回unk_token_id）
            unk_id = self.tokenizer.base_tokenizer.unk_token_id
            if l1_token_id == unk_id or l2_token_id == unk_id or l3_token_id == unk_id:
                logger.warning(f"无法转换语义ID: ({l1_id}, {l2_id}, {l3_id})")
                return None
            
            return [l1_token_id, l2_token_id, l3_token_id]
        
        except Exception as e:
            logger.warning(f"转换语义ID时出错 ({l1_id}, {l2_id}, {l3_id}): {e}")
            return None
    
    def _insert(self, token_sequence: List[int]):
        """
        将token序列插入Trie树
        
        Args:
            token_sequence: token ID序列（长度为3）
        """
        node = self.root
        
        for depth, token_id in enumerate(token_sequence, start=1):
            if token_id not in node.children:
                child = TrieNode()
                child.depth = depth
                node.children[token_id] = child
            
            node = node.children[token_id]
        
        # 标记序列结束
        node.is_end = True
    
    def get_valid_next_tokens(self, prefix_tokens: List[int]) -> Set[int]:
        """
        获取给定前缀后的所有有效下一个token
        
        Args:
            prefix_tokens: 前缀token序列（可以是空列表、长度1或长度2）
            
        Returns:
            有效的下一个token ID集合
        """
        # 从根节点开始遍历
        node = self.root
        
        # 沿着前缀路径遍历
        for token_id in prefix_tokens:
            if token_id not in node.children:
                # 前缀不在Trie树中，返回空集合
                return set()
            node = node.children[token_id]
        
        # 返回当前节点的所有子节点（即有效的下一个token）
        return set(node.children.keys())
    
    def is_valid_sequence(self, token_sequence: List[int]) -> bool:
        """
        检查token序列是否是有效的完整语义ID序列
        
        Args:
            token_sequence: token ID序列
            
        Returns:
            是否有效
        """
        if len(token_sequence) != 3:
            return False
        
        node = self.root
        for token_id in token_sequence:
            if token_id not in node.children:
                return False
            node = node.children[token_id]
        
        return node.is_end
    
    def get_all_valid_sequences(self) -> List[List[int]]:
        """
        获取所有有效的token序列
        
        Returns:
            所有有效序列的列表
        """
        sequences = []
        
        def dfs(node: TrieNode, current_path: List[int]):
            if node.is_end:
                sequences.append(current_path.copy())
                return
            
            for token_id, child in node.children.items():
                current_path.append(token_id)
                dfs(child, current_path)
                current_path.pop()
        
        dfs(self.root, [])
        return sequences
    
    def get_statistics(self) -> Dict:
        """
        获取Trie树的统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            'total_valid_sequences': len(self.valid_semantic_ids),
            'unique_l1_tokens': len(self.root.children),
            'l2_distribution': defaultdict(int),
            'l3_distribution': defaultdict(int)
        }
        
        # 统计L2和L3的分布
        for l1_token, l1_node in self.root.children.items():
            stats['l2_distribution'][len(l1_node.children)] += 1
            
            for l2_token, l2_node in l1_node.children.items():
                stats['l3_distribution'][len(l2_node.children)] += 1
        
        return stats


class ConstrainedLogitsProcessor:
    """
    约束logits处理器，用于在生成过程中强制模型只生成有效的语义ID序列
    
    这个处理器会在每一步生成时，根据已生成的前缀，将无效的token的logits设置为负无穷，
    从而确保模型只能生成Trie树中存在的有效序列。
    """
    
    def __init__(self, trie: SemanticIDTrie, tokenizer, eos_token_id: int):
        """
        初始化约束处理器
        
        Args:
            trie: SemanticIDTrie实例
            tokenizer: TIGER tokenizer实例
            eos_token_id: EOS token的ID
        """
        self.trie = trie
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id
        
        # 获取所有L1 token的ID（用于识别新序列的开始）
        self.l1_token_ids = set()
        for i in range(tokenizer.layer_vocab_sizes['l1']):
            token = f"<id_l1_{i}>"
            token_id = tokenizer.base_tokenizer.convert_tokens_to_ids(token)
            self.l1_token_ids.add(token_id)
        
        logger.info(f"约束处理器初始化完成，L1 token数量: {len(self.l1_token_ids)}")
    
    def __call__(self, input_ids, scores):
        """
        处理logits，约束生成
        
        Args:
            input_ids: 当前已生成的token序列 [batch_size, seq_len]
            scores: 当前步的logits [batch_size, vocab_size]
            
        Returns:
            处理后的scores
        """
        import torch
        
        batch_size = input_ids.shape[0]
        
        for batch_idx in range(batch_size):
            # 获取当前序列
            current_sequence = input_ids[batch_idx].tolist()
            
            # 提取当前正在生成的语义ID前缀
            prefix = self._extract_current_prefix(current_sequence)
            
            # 获取有效的下一个token
            valid_next_tokens = self.trie.get_valid_next_tokens(prefix)
            
            # 如果当前前缀长度为3（完整的语义ID），允许生成新的L1 token或EOS
            if len(prefix) == 3:
                valid_next_tokens = self.l1_token_ids.copy()
                valid_next_tokens.add(self.eos_token_id)
            
            # 如果没有有效的下一个token，只允许EOS
            if not valid_next_tokens:
                valid_next_tokens = {self.eos_token_id}
            
            # 将无效token的logits设置为负无穷
            mask = torch.ones_like(scores[batch_idx], dtype=torch.bool)
            for valid_token in valid_next_tokens:
                mask[valid_token] = False
            
            scores[batch_idx] = scores[batch_idx].masked_fill(mask, float('-inf'))
        
        return scores
    
    def _extract_current_prefix(self, sequence: List[int]) -> List[int]:
        """
        从完整序列中提取当前正在生成的语义ID前缀
        
        Args:
            sequence: 完整的token序列
            
        Returns:
            当前语义ID的前缀（长度0-2）
        """
        # 从后往前找，找到最近的L1 token
        prefix = []
        
        for i in range(len(sequence) - 1, -1, -1):
            token_id = sequence[i]
            
            # 如果是L1 token，说明找到了当前语义ID的开始
            if token_id in self.l1_token_ids:
                # 收集从L1开始的所有token（最多3个）
                for j in range(i, len(sequence)):
                    prefix.append(sequence[j])
                    if len(prefix) >= 3:
                        break
                break
        
        # 如果没有找到L1 token，说明还没开始生成语义ID，返回空前缀
        return prefix


def create_constrained_logits_processor(trie: SemanticIDTrie, tokenizer) -> ConstrainedLogitsProcessor:
    """
    创建约束logits处理器的工厂函数
    
    Args:
        trie: SemanticIDTrie实例
        tokenizer: TIGER tokenizer实例
        
    Returns:
        ConstrainedLogitsProcessor实例
    """
    eos_token_id = tokenizer.eos_token_id
    return ConstrainedLogitsProcessor(trie, tokenizer, eos_token_id)
