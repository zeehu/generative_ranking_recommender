"""
预处理脚本: 极速多进程版本 - 充分利用20核CPU + 50GB内存 
预计速度: 10,000-15,000 samples/s (10-15倍提升)
预计时间: 4-7分钟 (vs 原来71分钟)
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

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from config_optimized import Config
except ImportError:
    from config import Config
from src.generator.tiger_model import TIGERModel
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
    from src.generator.tiger_model import TIGERModel
    model = TIGERModel(base_model=model_name, layer_vocab_sizes=layer_vocab_sizes)
    worker_tokenizer = model.tokenizer


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
    logger.info(f"预计每个进程处理 {len(chunks)//cpu_count()} 个chunks")
    
    return chunks


def tokenize_dataset_multiproc(
    data_path: str,
    model_name: str,
    layer_vocab_sizes: dict,
    max_input_len: int,
    max_target_len: int,
    output_path: str,
    chunk_size: int = 20000,
    num_proc: int = None
):
    """
    多进程并行tokenize - 充分利用20核CPU + 50GB内存
    
    核心优化：
    1. 多进程并行tokenization（每个进程独立tokenizer）
    2. 大chunk size（20000样本/chunk，充分利用内存）
    3. 分批合并parquet（避免OOM）
    4. Arrow零拷贝 + Snappy压缩
    
    Args:
        data_path: TSV文件路径
        model_name: 模型名称
        layer_vocab_sizes: 层级词表大小
        max_input_len: 输入最大长度
        max_target_len: 目标最大长度
        output_path: 输出路径
        chunk_size: 每个chunk的样本数（建议20000-50000）
        num_proc: 并行进程数（默认CPU核心数-2）
    """
    if num_proc is None:
        num_proc = max(1, cpu_count() - 2)
    
    logger.info("=" * 80)
    logger.info("🚀 多进程并行Tokenization（极速版本）")
    logger.info("=" * 80)
    logger.info(f"文件路径: {data_path}")
    logger.info(f"Chunk大小: {chunk_size:,} 样本/chunk")
    logger.info(f"并行进程数: {num_proc} (CPU核心数: {cpu_count()})")
    logger.info(f"预计速度: 10,000-15,000 samples/s (vs 原来915 samples/s)")
    logger.info(f"预计提升: 10-15倍")
    
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
    
    # 步骤3: 分批合并parquet文件（避免OOM）
    logger.info("\n💾 步骤3: 分批合并parquet文件...")
    merge_start = time.time()
    
    # 创建临时Arrow目录
    temp_arrow_dir = output_path + "_temp_arrow"
    safe_remove_dir(temp_arrow_dir)
    os.makedirs(temp_arrow_dir, exist_ok=True)
    
    # 排序parquet文件
    parquet_files.sort()
    
    # 分批合并：每次50个文件（约100万样本，~2-3GB内存）
    merge_batch_size = 50
    num_merge_batches = (len(parquet_files) + merge_batch_size - 1) // merge_batch_size
    
    logger.info(f"分 {num_merge_batches} 批合并，每批 {merge_batch_size} 个文件")
    
    all_shards = []
    for i in range(num_merge_batches):
        start_idx = i * merge_batch_size
        end_idx = min((i + 1) * merge_batch_size, len(parquet_files))
        batch_files = parquet_files[start_idx:end_idx]
        
        # 读取并合并当前批次
        tables = [pq.read_table(pf, memory_map=True) for pf in batch_files]
        combined_table = pa.concat_tables(tables)
        del tables
        gc.collect()
        
        # 保存为Arrow shard（使用RecordBatchStreamWriter，兼容HFDataset.from_file）
        shard_path = os.path.join(temp_arrow_dir, f"data-{i:05d}-of-{num_merge_batches:05d}.arrow")
        with pa.OSFile(shard_path, 'wb') as sink:
            with pa.ipc.RecordBatchStreamWriter(sink, combined_table.schema) as writer:
                writer.write_table(combined_table)
        
        all_shards.append(shard_path)
        del combined_table
        gc.collect()
    
    merge_time = time.time() - merge_start
    logger.info(f"✅ 合并完成，耗时: {merge_time:.1f}秒")
    
    # 步骤4: 加载为HuggingFace Dataset
    logger.info("\n📦 步骤4: 加载为HuggingFace Dataset...")
    load_start = time.time()
    
    # 使用内存映射加载（不会OOM）
    datasets_list = [HFDataset.from_file(shard) for shard in all_shards]
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
        # 删除parquet临时文件
        safe_remove_dir(temp_parquet_dir)
        
        # 删除arrow临时文件
        safe_remove_dir(temp_arrow_dir)
        
        logger.info("✅ 临时文件清理完成")
    except Exception as e:
        logger.warning(f"⚠️ 清理临时文件时出现警告: {e}")
        logger.warning("临时文件未完全清理，但不影响最终结果")
    
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
    主函数: 预处理训练集和验证集
    """
    logger.info("=" * 80)
    logger.info("🚀 开始预处理tokenization（极速多进程版本）")
    logger.info("=" * 80)
    
    model_config = config.generator_t5
    rq_config = config.h_rqkmeans
    
    # 准备参数
    layer_vocab_sizes = {
        'l1': rq_config.need_clusters[0],
        'l2': rq_config.need_clusters[1],
        'l3': rq_config.need_clusters[2],
    }
    
    # 定义输入输出路径
    train_tsv = os.path.join(config.output_dir, "generator", "train.tsv")
    val_tsv = os.path.join(config.output_dir, "generator", "val.tsv")
    
    train_output = os.path.join(config.output_dir, "generator", "train_tokenized")
    val_output = os.path.join(config.output_dir, "generator", "val_tokenized")
    
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
            chunk_size=20000,  # 20000样本/chunk（平衡内存和速度）
            num_proc=18  # 20核CPU：使用18个进程（保留2核给系统）
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
        
        # 计算磁盘占用
        import subprocess
        try:
            size = subprocess.check_output(['du', '-sh', train_output]).split()[0].decode('utf-8')
            logger.info(f"💾 磁盘占用: {size}")
        except:
            pass
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
            chunk_size=20000,
            num_proc=18
        )
        
        logger.info("\n验证验证集...")
        val_dataset = HFDataset.load_from_disk(val_output)
        logger.info(f"✅ 验证集大小: {len(val_dataset):,} 样本")
        logger.info(f"✅ 数据集特征: {val_dataset.features}")
        del val_dataset
        gc.collect()
        
        try:
            size = subprocess.check_output(['du', '-sh', val_output]).split()[0].decode('utf-8')
            logger.info(f"💾 磁盘占用: {size}")
        except:
            pass
    else:
        logger.warning(f"❌ 验证集文件不存在: {val_tsv}")
    
    logger.info("\n" + "=" * 80)
    logger.info("🎉 预处理完成！")
    logger.info("=" * 80)
    logger.info(f"✅ 训练集保存位置: {train_output}")
    logger.info(f"✅ 验证集保存位置: {val_output}")
    logger.info("\n现在可以使用优化后的训练脚本进行训练")


if __name__ == "__main__":
    config = Config()
    log_file_path = os.path.join(config.log_dir, "preprocess_tokenize_fast.log")
    setup_logging(log_file=log_file_path)
    logger = logging.getLogger(__name__)
    
    preprocess_and_save(config)
