"""
预处理脚本: 内存优化版本 - 修复OOM问题
主要改进:
1. 减小chunk_size (20000 -> 10000) 降低内存峰值
2. 流式合并parquet文件，边处理边合并，避免文件堆积
3. 修复输出长度统计：按语义ID粒度统计
4. 增强内存清理和垃圾回收
"""
import os
import sys
import logging
import shutil
from tqdm import tqdm
from datasets import Dataset as HFDataset, concatenate_datasets
import gc
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List, Tuple
from multiprocessing import Pool, cpu_count
import time
import re

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from config_optimized import Config
except ImportError:
    from config import Config
from src.generator.tiger_model import TIGERTokenizer
from src.common.utils import setup_logging

logger = logging.getLogger(__name__)

# 全局变量：worker进程的tokenizer
worker_tokenizer = None


def safe_remove_dir(path: str, max_retries: int = 3, retry_delay: float = 1.0):
    """
    安全删除目录，处理可能的文件锁定问题
    如果目录不存在，直接返回（这是正常情况）
    
    Args:
        path: 要删除的目录路径
        max_retries: 最大重试次数
        retry_delay: 重试延迟（秒）
    """
    if not os.path.exists(path):
        logger.debug(f"目录不存在，跳过删除: {path}")
        return
    
    for attempt in range(max_retries):
        try:
            shutil.rmtree(path)
            logger.info(f"✅ 已删除目录: {path}")
            return
        except OSError as e:
            if attempt < max_retries - 1:
                logger.warning(f"⚠️ 删除目录失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                logger.info(f"等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
                gc.collect()  # 强制垃圾回收，释放可能的文件句柄
            else:
                logger.error(f"❌ 无法删除目录 {path} (已重试 {max_retries} 次): {e}")
                logger.warning(f"⚠️ 请手动删除该目录: {path}")
                raise


def init_worker(model_name: str, layer_vocab_sizes: dict):
    """
    初始化worker进程的tokenizer（每个进程一个独立的tokenizer）
    """
    global worker_tokenizer
    from src.generator.tiger_model_new import TIGERTokenizer
    worker_tokenizer = TIGERTokenizer(base_model=model_name, layer_vocab_sizes=layer_vocab_sizes)


def count_semantic_ids(text: str) -> int:
    """
    统计文本中的语义ID数量
    
    Args:
        text: 包含语义ID的文本，如 "<id_l1_3> <id_l2_99> ..."
    
    Returns:
        语义ID的数量
    """
    # 匹配 <id_l1_xxx> <id_l2_xxx> <id_l3_xxx> 格式
    pattern = r'<id_l[123]_\d+>'
    matches = re.findall(pattern, text)
    return len(matches)


def tokenize_chunk_worker(args: Tuple[List[str], List[str], int, int, int, str]) -> Tuple[str, int]:
    """
    Worker函数：tokenize一个数据块
    
    Args:
        args: (input_texts, target_texts, max_input_len, max_target_len, chunk_id, temp_dir)
    
    Returns:
        (parquet_file_path, num_samples)
    """
    input_texts, target_texts, max_input_len, max_target_len, chunk_id, temp_dir = args
    
    # 使用worker的tokenizer
    global worker_tokenizer
    
    # 批量tokenize（HuggingFace tokenizer内部已优化）
    input_encodings = worker_tokenizer.base_tokenizer(
        input_texts,
        max_length=max_input_len,
        truncation=True,
        padding='max_length',
        return_tensors=None
    )
    
    target_encodings = worker_tokenizer.base_tokenizer(
        target_texts,
        max_length=max_target_len,
        truncation=True,
        padding='max_length',
        return_tensors=None
    )
    
    # 采样打印：只在第一个chunk（chunk_id=0）打印5条样本
    if chunk_id == 0:
        print("\n" + "=" * 100)
        print(f"📋 采样检查 - Chunk {chunk_id} 的前5条数据")
        print("=" * 100)
        
        num_samples_to_print = min(5, len(input_texts))
        for i in range(num_samples_to_print):
            print(f"\n{'─' * 100}")
            print(f"样本 #{i+1}")
            print(f"{'─' * 100}")
            
            # 原始输入
            print(f"\n【原始输入】")
            print(f"  文本: {input_texts[i][:200]}{'...' if len(input_texts[i]) > 200 else ''}")
            print(f"  长度: {len(input_texts[i])} 字符")
            
            # 原始输出 - 修复：按语义ID粒度统计
            num_semantic_ids = count_semantic_ids(target_texts[i])
            print(f"\n【原始输出】")
            print(f"  文本: {target_texts[i][:200]}{'...' if len(target_texts[i]) > 200 else ''}")
            print(f"  字符长度: {len(target_texts[i])} 字符")
            print(f"  语义ID数量: {num_semantic_ids} 个")
            
            # Tokenize后的输入
            input_ids = input_encodings['input_ids'][i]
            attention_mask = input_encodings['attention_mask'][i]
            print(f"\n【Tokenize后的输入】")
            print(f"  input_ids: {input_ids[:50]}{'...' if len(input_ids) > 50 else ''}")
            print(f"  input_ids长度: {len(input_ids)}")
            print(f"  有效token数: {sum(attention_mask)}")
            print(f"  padding数: {len(attention_mask) - sum(attention_mask)}")
            
            # Tokenize后的输出
            label_ids = target_encodings['input_ids'][i]
            label_attention = target_encodings['attention_mask'][i]
            print(f"\n【Tokenize后的输出】")
            print(f"  label_ids: {label_ids[:50]}{'...' if len(label_ids) > 50 else ''}")
            print(f"  label_ids长度: {len(label_ids)}")
            print(f"  有效token数: {sum(label_attention)}")
            print(f"  padding数: {len(label_attention) - sum(label_attention)}")
            
            # 解码验证（前50个token）
            decoded_input = worker_tokenizer.base_tokenizer.decode(
                [tid for tid in input_ids[:50] if tid != worker_tokenizer.base_tokenizer.pad_token_id],
                skip_special_tokens=False
            )
            decoded_target = worker_tokenizer.base_tokenizer.decode(
                [tid for tid in label_ids[:50] if tid != worker_tokenizer.base_tokenizer.pad_token_id],
                skip_special_tokens=False
            )
            print(f"\n【解码验证（前50个token）】")
            print(f"  输入解码: {decoded_input}")
            print(f"  输出解码: {decoded_target}")
        
        print(f"\n{'=' * 100}")
        print(f"✅ 采样检查完成")
        print(f"{'=' * 100}\n")
    
    # 构建Arrow Table（零拷贝）
    schema = pa.schema([
        ('input_ids', pa.list_(pa.int64())),
        ('attention_mask', pa.list_(pa.int64())),
        ('labels', pa.list_(pa.int64()))
    ])
    
    arrays = [
        pa.array(input_encodings['input_ids']),
        pa.array(input_encodings['attention_mask']),
        pa.array(target_encodings['input_ids'])
    ]
    table = pa.Table.from_arrays(arrays, schema=schema)
    
    # 写入parquet文件（snappy压缩）
    parquet_file = os.path.join(temp_dir, f"chunk_{chunk_id:06d}.parquet")
    pq.write_table(table, parquet_file, compression='snappy')
    
    # 立即释放内存
    del input_encodings, target_encodings, arrays, table
    gc.collect()
    
    return parquet_file, len(input_texts)


def read_and_split_data(data_path: str, chunk_size: int) -> List[Tuple[List[str], List[str]]]:
    """
    快速读取TSV文件并分割成多个chunk（用于多进程处理）
    
    Args:
        data_path: TSV文件路径
        chunk_size: 每个chunk的大小
    
    Returns:
        List of (input_texts, target_texts) tuples
    """
    logger.info(f"读取文件: {data_path}")
    logger.info(f"Chunk大小: {chunk_size:,} 样本/chunk")
    
    chunks = []
    input_texts = []
    target_texts = []
    total_lines = 0
    
    # 使用大缓冲区加速读取
    with open(data_path, 'r', encoding='utf-8', buffering=32*1024*1024) as f:  # 32MB缓冲
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            
            glid, input_text, target_text = parts
            input_texts.append(input_text)
            target_texts.append(target_text)
            total_lines += 1
            
            if len(input_texts) >= chunk_size:
                chunks.append((input_texts, target_texts))
                input_texts = []
                target_texts = []
    
    # 添加剩余数据
    if input_texts:
        chunks.append((input_texts, target_texts))
    
    logger.info(f"总样本数: {total_lines:,}")
    logger.info(f"分割成 {len(chunks)} 个chunks")
    
    return chunks


def merge_parquet_files_streaming(parquet_files: List[str], output_dir: str, batch_size: int = 30) -> List[str]:
    """
    流式合并parquet文件，避免内存堆积
    
    Args:
        parquet_files: parquet文件列表
        output_dir: 输出目录
        batch_size: 每批合并的文件数（降低到30以减少内存）
    
    Returns:
        合并后的Arrow文件列表
    """
    logger.info(f"\n💾 流式合并parquet文件...")
    logger.info(f"  总文件数: {len(parquet_files)}")
    logger.info(f"  批次大小: {batch_size} 文件/批")
    
    # 排序文件
    parquet_files.sort()
    
    # 分批合并
    num_batches = (len(parquet_files) + batch_size - 1) // batch_size
    logger.info(f"  分 {num_batches} 批合并")
    
    arrow_shards = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(parquet_files))
        batch_files = parquet_files[start_idx:end_idx]
        
        logger.info(f"  合并批次 {i+1}/{num_batches} ({len(batch_files)} 文件)...")
        
        # 读取并合并当前批次
        tables = []
        for pf in batch_files:
            tables.append(pq.read_table(pf, memory_map=True))
        
        combined_table = pa.concat_tables(tables)
        del tables
        gc.collect()
        
        # 保存为Arrow shard
        shard_path = os.path.join(output_dir, f"data-{i:05d}-of-{num_batches:05d}.arrow")
        with pa.OSFile(shard_path, 'wb') as sink:
            with pa.ipc.RecordBatchStreamWriter(sink, combined_table.schema) as writer:
                writer.write_table(combined_table)
        
        arrow_shards.append(shard_path)
        
        # 立即删除已合并的parquet文件，释放磁盘空间
        for pf in batch_files:
            try:
                os.remove(pf)
            except:
                pass
        
        del combined_table
        gc.collect()
        
        logger.info(f"  ✅ 批次 {i+1}/{num_batches} 完成")
    
    logger.info(f"✅ 流式合并完成，生成 {len(arrow_shards)} 个Arrow文件")
    return arrow_shards


def tokenize_dataset_multiproc(
    data_path: str,
    model_name: str,
    layer_vocab_sizes: dict,
    max_input_len: int,
    max_target_len: int,
    output_path: str,
    chunk_size: int = 10000,  # 降低到10000以减少内存峰值
    num_proc: int = None
):
    """
    多进程并行tokenize - 内存优化版本
    
    核心优化：
    1. 减小chunk_size (20000 -> 10000) 降低内存峰值
    2. 流式合并parquet文件，边处理边合并
    3. 及时清理临时文件
    4. 增强垃圾回收
    
    Args:
        data_path: TSV文件路径
        model_name: 模型名称
        layer_vocab_sizes: 层级词表大小
        max_input_len: 输入最大长度
        max_target_len: 目标最大长度
        output_path: 输出路径
        chunk_size: 每个chunk的样本数（降低到10000）
        num_proc: 并行进程数（默认CPU核心数-2）
    """
    if num_proc is None:
        num_proc = max(1, cpu_count() - 2)
    
    logger.info("=" * 80)
    logger.info("🚀 多进程并行Tokenization（内存优化版本）")
    logger.info("=" * 80)
    logger.info(f"文件路径: {data_path}")
    logger.info(f"Chunk大小: {chunk_size:,} 样本/chunk (降低以减少内存)")
    logger.info(f"并行进程数: {num_proc} (CPU核心数: {cpu_count()})")
    logger.info(f"内存优化: 流式合并 + 及时清理")
    
    start_time = time.time()
    
    # 准备临时目录
    temp_parquet_dir = output_path + "_temp_parquet"
    safe_remove_dir(temp_parquet_dir)
    os.makedirs(temp_parquet_dir, exist_ok=True)
    
    # 步骤1: 快速读取并分割数据
    logger.info("\n📖 步骤1: 读取并分割数据...")
    read_start = time.time()
    chunks = read_and_split_data(data_path, chunk_size)
    read_time = time.time() - read_start
    logger.info(f"✅ 读取完成，耗时: {read_time:.1f}秒")
    
    # 步骤2: 多进程并行tokenization
    logger.info(f"\n⚡ 步骤2: 启动 {num_proc} 个进程进行并行tokenization...")
    
    # 准备参数
    chunk_args = [
        (input_texts, target_texts, max_input_len, max_target_len, i, temp_parquet_dir)
        for i, (input_texts, target_texts) in enumerate(chunks)
    ]
    
    # 创建进程池并并行处理
    tokenize_start = time.time()
    parquet_files = []
    total_samples = 0
    
    with Pool(
        processes=num_proc,
        initializer=init_worker,
        initargs=(model_name, layer_vocab_sizes)
    ) as pool:
        # 使用imap_unordered获得更好的性能
        with tqdm(total=len(chunks), desc="🔥 Tokenizing", unit="chunk") as pbar:
            for parquet_file, num_samples in pool.imap_unordered(tokenize_chunk_worker, chunk_args, chunksize=1):
                parquet_files.append(parquet_file)
                total_samples += num_samples
                pbar.update(1)
                pbar.set_postfix({
                    "samples": f"{total_samples:,}",
                    "speed": f"{total_samples/(time.time()-tokenize_start):.0f} samples/s"
                })
    
    tokenize_time = time.time() - tokenize_start
    speed = total_samples / tokenize_time
    
    logger.info(f"\n✅ Tokenization完成！")
    logger.info(f"   总样本数: {total_samples:,}")
    logger.info(f"   耗时: {tokenize_time:.1f}秒")
    logger.info(f"   速度: {speed:.0f} samples/s")
    logger.info(f"   生成文件: {len(parquet_files)} 个parquet文件")
    
    # 步骤3: 流式合并parquet文件
    temp_arrow_dir = output_path + "_temp_arrow"
    safe_remove_dir(temp_arrow_dir)
    os.makedirs(temp_arrow_dir, exist_ok=True)
    
    merge_start = time.time()
    arrow_shards = merge_parquet_files_streaming(
        parquet_files, 
        temp_arrow_dir, 
        batch_size=30  # 降低批次大小以减少内存
    )
    merge_time = time.time() - merge_start
    logger.info(f"✅ 合并完成，耗时: {merge_time:.1f}秒")
    
    # 步骤4: 加载为HuggingFace Dataset
    logger.info("\n📦 步骤4: 加载为HuggingFace Dataset...")
    load_start = time.time()
    
    # 使用内存映射加载
    datasets_list = []
    for shard in arrow_shards:
        datasets_list.append(HFDataset.from_file(shard))
    
    dataset = concatenate_datasets(datasets_list)
    del datasets_list
    gc.collect()
    
    logger.info(f"✅ 数据集大小: {len(dataset):,} 样本")
    
    # 保存最终数据集
    logger.info("\n💿 步骤5: 保存最终数据集...")
    save_start = time.time()
    
    # 清理旧的输出目录
    safe_remove_dir(output_path)
    
    # 直接保存到最终位置
    dataset.save_to_disk(output_path)
    
    del dataset
    gc.collect()
    
    save_time = time.time() - save_start
    logger.info(f"✅ 保存完成，耗时: {save_time:.1f}秒")
    
    # 清理所有临时文件
    logger.info("\n🧹 清理临时文件...")
    try:
        safe_remove_dir(temp_parquet_dir)
        safe_remove_dir(temp_arrow_dir)
        logger.info("✅ 临时文件清理完成")
    except Exception as e:
        logger.warning(f"⚠️ 清理临时文件时出现警告: {e}")
    
    # 总结
    total_time = time.time() - start_time
    logger.info("\n" + "=" * 80)
    logger.info("🎉 处理完成！")
    logger.info("=" * 80)
    logger.info(f"总耗时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
    logger.info(f"平均速度: {total_samples/total_time:.0f} samples/s")
    logger.info(f"数据保存位置: {output_path}")
    logger.info("=" * 80)


def preprocess_and_save(config: Config):
    """
    主函数: 预处理训练集、验证集和测试集
    """
    logger.info("=" * 80)
    logger.info("🚀 开始预处理tokenization（内存优化版本）")
    logger.info("=" * 80)
    
    model_config = config.generator_t5
    rq_config = config.h_rqkmeans
    
    # 准备参数
    layer_vocab_sizes = {
        'l1': rq_config.need_clusters[0],
        'l2': rq_config.need_clusters[1],
        'l3': rq_config.need_clusters[2],
    }
    
    logger.info(f"\n📊 Tokenizer配置:")
    logger.info(f"  模型: {model_config.model_name}")
    logger.info(f"  Layer 1 词表大小: {layer_vocab_sizes['l1']}")
    logger.info(f"  Layer 2 词表大小: {layer_vocab_sizes['l2']}")
    logger.info(f"  Layer 3 词表大小: {layer_vocab_sizes['l3']}")
    logger.info(f"  总语义ID tokens: {sum(layer_vocab_sizes.values())}")
    
    # 定义输入输出路径
    train_tsv = os.path.join(config.output_dir, "generator", "train.tsv")
    val_tsv = os.path.join(config.output_dir, "generator", "val.tsv")
    test_tsv = os.path.join(config.output_dir, "generator", "test.tsv")
    
    train_output = os.path.join(config.output_dir, "generator", "train_tokenized")
    val_output = os.path.join(config.output_dir, "generator", "val_tokenized")
    test_output = os.path.join(config.output_dir, "generator", "test_tokenized")
    
    # 处理训练集
    if os.path.exists(train_tsv):
        logger.info("\n" + "=" * 80)
        logger.info("处理训练集")
        logger.info("=" * 80)
        
        if os.path.exists(train_output):
            logger.info(f"检测到旧的预处理数据，正在删除: {train_output}")
            try:
                safe_remove_dir(train_output)
            except Exception as e:
                logger.error(f"❌ 无法删除旧数据: {e}")
                logger.info("尝试使用新的目录名...")
                train_output = train_output + f"_new_{int(time.time())}"
                logger.info(f"新的输出目录: {train_output}")
        
        tokenize_dataset_multiproc(
            train_tsv,
            model_config.model_name,
            layer_vocab_sizes,
            model_config.max_input_length,
            model_config.max_target_length,
            train_output,
            chunk_size=10000,  # 降低chunk_size以减少内存
            num_proc=16  # 降低进程数（从18降到16）以减少内存压力
        )
        
        # 验证数据集
        logger.info("\n验证训练集...")
        train_dataset = HFDataset.load_from_disk(train_output)
        logger.info(f"✅ 训练集大小: {len(train_dataset):,} 样本")
        logger.info(f"✅ 数据集特征: {train_dataset.features}")
        
        # 显示样本
        logger.info("\n样本示例:")
        logger.info(f"  input_ids长度: {len(train_dataset[0]['input_ids'])}")
        logger.info(f"  labels长度: {len(train_dataset[0]['labels'])}")
        
        del train_dataset
        gc.collect()
    else:
        logger.warning(f"❌ 训练集文件不存在: {train_tsv}")
    
    # 处理验证集
    if os.path.exists(val_tsv):
        logger.info("\n" + "=" * 80)
        logger.info("处理验证集")
        logger.info("=" * 80)
        
        if os.path.exists(val_output):
            logger.info(f"检测到旧的预处理数据，正在删除: {val_output}")
            try:
                safe_remove_dir(val_output)
            except Exception as e:
                logger.error(f"❌ 无法删除旧数据: {e}")
                logger.info("尝试使用新的目录名...")
                val_output = val_output + f"_new_{int(time.time())}"
                logger.info(f"新的输出目录: {val_output}")
        
        tokenize_dataset_multiproc(
            val_tsv,
            model_config.model_name,
            layer_vocab_sizes,
            model_config.max_input_length,
            model_config.max_target_length,
            val_output,
            chunk_size=10000,
            num_proc=16
        )
        
        logger.info("\n验证验证集...")
        val_dataset = HFDataset.load_from_disk(val_output)
        logger.info(f"✅ 验证集大小: {len(val_dataset):,} 样本")
        logger.info(f"✅ 数据集特征: {val_dataset.features}")
        del val_dataset
        gc.collect()
    else:
        logger.warning(f"❌ 验证集文件不存在: {val_tsv}")
    
    # 处理测试集
    if os.path.exists(test_tsv):
        logger.info("\n" + "=" * 80)
        logger.info("处理测试集")
        logger.info("=" * 80)
        
        if os.path.exists(test_output):
            logger.info(f"检测到旧的预处理数据，正在删除: {test_output}")
            try:
                safe_remove_dir(test_output)
            except Exception as e:
                logger.error(f"❌ 无法删除旧数据: {e}")
                logger.info("尝试使用新的目录名...")
                test_output = test_output + f"_new_{int(time.time())}"
                logger.info(f"新的输出目录: {test_output}")
        
        tokenize_dataset_multiproc(
            test_tsv,
            model_config.model_name,
            layer_vocab_sizes,
            model_config.max_input_length,
            model_config.max_target_length,
            test_output,
            chunk_size=10000,
            num_proc=16
        )
        
        logger.info("\n验证测试集...")
        test_dataset = HFDataset.load_from_disk(test_output)
        logger.info(f"✅ 测试集大小: {len(test_dataset):,} 样本")
        logger.info(f"✅ 数据集特征: {test_dataset.features}")
        del test_dataset
        gc.collect()
    else:
        logger.warning(f"❌ 测试集文件不存在: {test_tsv}")
    
    logger.info("\n" + "=" * 80)
    logger.info("🎉 预处理完成！")
    logger.info("=" * 80)
    logger.info(f"✅ 训练集保存位置: {train_output}")
    logger.info(f"✅ 验证集保存位置: {val_output}")
    if os.path.exists(test_tsv):
        logger.info(f"✅ 测试集保存位置: {test_output}")
    logger.info("\n现在可以使用优化后的训练脚本进行训练")


if __name__ == "__main__":
    config = Config()
    log_file_path = os.path.join(config.log_dir, "preprocess_tokenize_fixed.log")
    setup_logging(log_file=log_file_path)
    logger = logging.getLogger(__name__)
    
    preprocess_and_save(config)
