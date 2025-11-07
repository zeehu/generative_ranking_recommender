"""
预处理脚本: 将训练数据tokenize并保存为Arrow格式     
使用内存映射技术，大幅提升训练速度，降低内存占用
"""
import os
import sys
import logging
import shutil
from tqdm import tqdm
from datasets import Dataset as HFDataset
import gc

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


def tokenize_dataset_streaming(data_path: str, tokenizer, max_input_len: int, max_target_len: int, output_path: str, batch_size: int = 10000):
    """
    流式处理TSV文件并tokenize，分批保存以避免内存溢出
    
    Args:
        data_path: TSV文件路径
        tokenizer: TIGERTokenizer
        max_input_len: 输入最大长度
        max_target_len: 目标最大长度
        output_path: 输出路径
        batch_size: 每批处理的样本数
    """
    logger.info(f"开始流式处理文件: {data_path}")
    logger.info(f"批处理大小: {batch_size:,} 样本/批")
    
    # 首先统计总行数用于进度条
    logger.info("统计样本数量...")
    with open(data_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    logger.info(f"总样本数: {total_lines:,}")
    logger.info("开始分批tokenization...")
    
    batch_data = []
    total_processed = 0
    batch_num = 0
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Tokenizing"):
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            
            glid, input_text, target_text = parts
            
            # Tokenize input
            input_encoding = tokenizer.base_tokenizer(
                input_text,
                max_length=max_input_len,
                truncation=True,
                padding='max_length',
                return_tensors=None
            )
            
            # Tokenize target
            target_encoding = tokenizer.base_tokenizer(
                target_text,
                max_length=max_target_len,
                truncation=True,
                padding='max_length',
                return_tensors=None
            )
            
            batch_data.append({
                'input_ids': input_encoding['input_ids'],
                'attention_mask': input_encoding['attention_mask'],
                'labels': target_encoding['input_ids']
            })
            
            # 当达到batch_size时，保存这一批
            if len(batch_data) >= batch_size:
                logger.info(f"\n保存第 {batch_num + 1} 批数据 ({len(batch_data):,} 样本)...")
                
                # 转换为Dataset并保存
                batch_dataset = HFDataset.from_list(batch_data)
                
                if batch_num == 0:
                    # 第一批：创建新数据集
                    batch_dataset.save_to_disk(output_path)
                else:
                    # 后续批次：使用临时目录避免覆盖问题
                    temp_path = output_path + "_temp"
                    existing_dataset = HFDataset.load_from_disk(output_path)
                    from datasets import concatenate_datasets
                    combined_dataset = concatenate_datasets([existing_dataset, batch_dataset])
                    combined_dataset.save_to_disk(temp_path)
                    
                    # 删除旧数据，重命名新数据
                    del existing_dataset, combined_dataset
                    gc.collect()
                    shutil.rmtree(output_path)
                    shutil.move(temp_path, output_path)
                
                total_processed += len(batch_data)
                logger.info(f"已处理: {total_processed:,} / {total_lines:,} 样本")
                
                # 清理内存
                del batch_data, batch_dataset
                batch_data = []
                batch_num += 1
                gc.collect()
    
    # 处理剩余的数据
    if batch_data:
        logger.info(f"\n保存最后一批数据 ({len(batch_data):,} 样本)...")
        batch_dataset = HFDataset.from_list(batch_data)
        
        if batch_num == 0:
            batch_dataset.save_to_disk(output_path)
        else:
            temp_path = output_path + "_temp"
            existing_dataset = HFDataset.load_from_disk(output_path)
            from datasets import concatenate_datasets
            combined_dataset = concatenate_datasets([existing_dataset, batch_dataset])
            combined_dataset.save_to_disk(temp_path)
            
            # 删除旧数据，重命名新数据
            del existing_dataset, combined_dataset
            gc.collect()
            shutil.rmtree(output_path)
            shutil.move(temp_path, output_path)
        
        total_processed += len(batch_data)
        del batch_data, batch_dataset
        gc.collect()
    
    logger.info(f"\n成功tokenize {total_processed:,} 个样本")
    logger.info(f"数据已保存到: {output_path}")


def preprocess_and_save(config: Config):
    """
    主函数: 预处理训练集和验证集
    """
    logger.info("=" * 80)
    logger.info("开始预处理tokenization")
    logger.info("=" * 80)
    
    model_config = config.generator_t5
    rq_config = config.h_rqkmeans
    
    # 初始化tokenizer
    logger.info("初始化TIGER模型和tokenizer...")
    layer_vocab_sizes = {
        'l1': rq_config.need_clusters[0],
        'l2': rq_config.need_clusters[1],
        'l3': rq_config.need_clusters[2],
    }
    
    model = TIGERModel(base_model=model_config.model_name, layer_vocab_sizes=layer_vocab_sizes)
    tokenizer = model.tokenizer
    
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")
    
    # 定义输入输出路径
    train_tsv = os.path.join(config.output_dir, "generator", "train.tsv")
    val_tsv = os.path.join(config.output_dir, "generator", "val.tsv")
    
    train_output = os.path.join(config.output_dir, "generator", "train_tokenized")
    val_output = os.path.join(config.output_dir, "generator", "val_tokenized")
    
    # 处理训练集
    logger.info("\n" + "=" * 80)
    logger.info("处理训练集")
    logger.info("=" * 80)
    
    if os.path.exists(train_tsv):
        # 清理旧的预处理数据（如果存在）
        if os.path.exists(train_output):
            logger.info(f"删除旧的预处理数据: {train_output}")
            shutil.rmtree(train_output)
        
        # 使用流式处理，避免内存溢出
        tokenize_dataset_streaming(
            train_tsv,
            tokenizer,
            model_config.max_input_length,
            model_config.max_target_length,
            train_output,
            batch_size=10000  # 每批处理1万个样本
        )
        
        # 显示数据集信息
        logger.info("\n加载数据集以验证...")
        train_dataset = HFDataset.load_from_disk(train_output)
        logger.info(f"训练集大小: {len(train_dataset):,} 样本")
        logger.info(f"数据集特征: {train_dataset.features}")
        del train_dataset
        gc.collect()
        
        # 计算磁盘占用
        import subprocess
        try:
            size = subprocess.check_output(['du', '-sh', train_output]).split()[0].decode('utf-8')
            logger.info(f"磁盘占用: {size}")
        except:
            pass
    else:
        logger.warning(f"训练集文件不存在: {train_tsv}")
    
    # 处理验证集
    logger.info("\n" + "=" * 80)
    logger.info("处理验证集")
    logger.info("=" * 80)
    
    if os.path.exists(val_tsv):
        # 清理旧的预处理数据（如果存在）
        if os.path.exists(val_output):
            logger.info(f"删除旧的预处理数据: {val_output}")
            shutil.rmtree(val_output)
        
        # 验证集较小，可以一次性处理
        tokenize_dataset_streaming(
            val_tsv,
            tokenizer,
            model_config.max_input_length,
            model_config.max_target_length,
            val_output,
            batch_size=10000
        )
        
        logger.info("\n加载数据集以验证...")
        val_dataset = HFDataset.load_from_disk(val_output)
        logger.info(f"验证集大小: {len(val_dataset):,} 样本")
        logger.info(f"数据集特征: {val_dataset.features}")
        del val_dataset
        gc.collect()
        
        try:
            size = subprocess.check_output(['du', '-sh', val_output]).split()[0].decode('utf-8')
            logger.info(f"磁盘占用: {size}")
        except:
            pass
    else:
        logger.warning(f"验证集文件不存在: {val_tsv}")
    
    logger.info("\n" + "=" * 80)
    logger.info("预处理完成！")
    logger.info("=" * 80)
    logger.info(f"训练集保存位置: {train_output}")
    logger.info(f"验证集保存位置: {val_output}")
    logger.info("\n现在可以使用优化后的训练脚本进行训练")


if __name__ == "__main__":
    config = Config()
    log_file_path = os.path.join(config.log_dir, "preprocess_tokenize.log")
    setup_logging(log_file=log_file_path)
    logger = logging.getLogger(__name__)
    
    preprocess_and_save(config)
