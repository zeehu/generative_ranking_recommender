"""
分析语义ID的分布情况，诊断覆盖率低的原因
"""
import json
import numpy as np
from collections import Counter, defaultdict

def analyze_semantic_ids(semantic_ids_file):
    """分析语义ID文件"""
    
    # 读取语义ID
    semantic_ids = []
    with open(semantic_ids_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            semantic_ids.append(tuple(data['semantic_ids']))
    
    print(f"总歌曲数: {len(semantic_ids)}")
    print(f"唯一语义ID数: {len(set(semantic_ids))}")
    print(f"覆盖率: {len(set(semantic_ids)) / len(semantic_ids) * 100:.2f}%")
    print()
    
    # 分析每层的分布
    layer1_ids = [sid[0] for sid in semantic_ids]
    layer2_ids = [sid[1] for sid in semantic_ids]
    layer3_ids = [sid[2] for sid in semantic_ids]
    
    print("=" * 80)
    print("单层分布分析")
    print("=" * 80)
    print(f"第1层唯一ID数: {len(set(layer1_ids))}")
    print(f"第2层唯一ID数: {len(set(layer2_ids))}")
    print(f"第3层唯一ID数: {len(set(layer3_ids))}")
    print()
    
    # 分析前两层的组合
    layer12_combinations = [(sid[0], sid[1]) for sid in semantic_ids]
    unique_layer12 = set(layer12_combinations)
    print("=" * 80)
    print("前两层组合分析")
    print("=" * 80)
    print(f"理论组合数: 128 × 128 = 16,384")
    print(f"实际组合数: {len(unique_layer12)}")
    print(f"组合利用率: {len(unique_layer12) / 16384 * 100:.2f}%")
    print()
    
    # 统计每个前两层组合对应的第3层ID数量
    layer12_to_layer3 = defaultdict(set)
    for sid in semantic_ids:
        layer12_to_layer3[(sid[0], sid[1])].add(sid[2])
    
    layer3_counts = [len(v) for v in layer12_to_layer3.values()]
    print("=" * 80)
    print("第3层分布分析（按前两层组合）")
    print("=" * 80)
    print(f"平均每个[layer1,layer2]组合使用的layer3 ID数: {np.mean(layer3_counts):.2f}")
    print(f"最少: {np.min(layer3_counts)}")
    print(f"最多: {np.max(layer3_counts)}")
    print(f"中位数: {np.median(layer3_counts):.2f}")
    print(f"标准差: {np.std(layer3_counts):.2f}")
    print()
    
    # 统计每个组合的样本数
    combination_counts = Counter(semantic_ids)
    print("=" * 80)
    print("语义ID重复度分析")
    print("=" * 80)
    print(f"平均每个语义ID对应的歌曲数: {len(semantic_ids) / len(set(semantic_ids)):.2f}")
    
    # 找出最常见的语义ID
    most_common = combination_counts.most_common(10)
    print("\n最常见的10个语义ID:")
    for sid, count in most_common:
        print(f"  {sid}: {count}首歌曲")
    
    # 找出只出现一次的语义ID
    unique_once = sum(1 for count in combination_counts.values() if count == 1)
    print(f"\n只出现一次的语义ID数: {unique_once}")
    print(f"占比: {unique_once / len(set(semantic_ids)) * 100:.2f}%")
    
    # 分析每个[layer1, layer2]组合的样本数分布
    layer12_sample_counts = defaultdict(int)
    for sid in semantic_ids:
        layer12_sample_counts[(sid[0], sid[1])] += 1
    
    sample_counts = list(layer12_sample_counts.values())
    print()
    print("=" * 80)
    print("[Layer1, Layer2]组合的样本数分布")
    print("=" * 80)
    print(f"平均样本数: {np.mean(sample_counts):.2f}")
    print(f"最少: {np.min(sample_counts)}")
    print(f"最多: {np.max(sample_counts)}")
    print(f"中位数: {np.median(sample_counts):.2f}")
    print(f"标准差: {np.std(sample_counts):.2f}")
    
    # 找出样本数最少的组合
    sorted_by_count = sorted(layer12_sample_counts.items(), key=lambda x: x[1])
    print("\n样本数最少的10个[layer1, layer2]组合:")
    for (l1, l2), count in sorted_by_count[:10]:
        layer3_used = len(layer12_to_layer3[(l1, l2)])
        print(f"  [{l1}, {l2}]: {count}首歌曲, 使用了{layer3_used}个layer3 ID")


if __name__ == '__main__':
    semantic_ids_file = "outputs/semantic_id/song_semantic_ids.jsonl"
    analyze_semantic_ids(semantic_ids_file)
