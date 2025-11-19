import os
import random    
import numpy as np
import torch
from typing import Optional
import logging

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def is_main_process() -> bool:
    """检查当前进程是否为主进程（rank 0）
    
    在分布式训练中，通过以下方式判断：
    1. 如果torch.distributed已初始化，使用dist.get_rank()
    2. 如果未初始化但设置了环境变量，使用LOCAL_RANK或RANK
    3. 否则认为是主进程（单进程训练）
    """
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
        
        # 检查环境变量（在分布式启动时会设置）
        # torchrun 和 torch.distributed.launch 都会设置这些变量
        local_rank = int(os.environ.get('LOCAL_RANK', -1))
        if local_rank != -1:
            return local_rank == 0
        
        rank = int(os.environ.get('RANK', -1))
        if rank != -1:
            return rank == 0
    except:
        pass
    
    # 默认情况（单进程训练）
    return True


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Setup root logger to capture all logs and write to both console and file.
    Removes any existing handlers to prevent duplicate logs.
    
    在分布式训练环境中，只有rank 0进程会输出到控制台，避免重复打印。
    
    注意：此函数应该在程序启动时只调用一次。
    """
    # 检查是否在分布式训练环境中
    import torch.distributed as dist
    is_distributed = dist.is_available() and dist.is_initialized()
    is_main = is_main_process()
    
    root_logger = logging.getLogger()
    
    # 防止重复配置：如果已经配置过且有handlers，先清除
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    # 设置日志级别
    root_logger.setLevel(level)
    
    # 防止日志向上传播导致重复
    root_logger.propagate = False

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 只有主进程输出到控制台
    if is_main:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # 文件日志：只有主进程写入主日志文件，其他进程写入带rank后缀的文件
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # 在分布式训练中，为每个进程创建单独的日志文件
        if is_distributed and not is_main:
            rank = dist.get_rank()
            base_name, ext = os.path.splitext(log_file)
            log_file = f"{base_name}_rank{rank}{ext}"
        
        file_handler = logging.FileHandler(log_file, mode='a')  # 追加模式
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # 设置第三方库的日志级别，减少噪音
    # 同时防止这些库的日志重复
    for lib_name in ['transformers', 'datasets', 'torch', 'transformers.trainer', 'accelerate']:
        lib_logger = logging.getLogger(lib_name)
        lib_logger.setLevel(logging.WARNING)
        lib_logger.propagate = True  # 允许向上传播，但会被root logger的级别过滤
    
    # 返回一个模块级别的logger
    return root_logger


def load_song_vectors(path: str) -> dict:
    """
    Loads song vectors from a project-standard CSV file.

    Args:
        path (str): The path to the song vectors file.

    Returns:
        dict: A dictionary mapping song_id (str) to its vector (np.ndarray).
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading song vectors from: {path}")
    vectors = {}
    try:
        from tqdm import tqdm
        import csv
        
        # Get total number of lines for tqdm progress bar
        with open(path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for line in f)

        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in tqdm(reader, total=total_lines, desc="Reading song vectors"):
                if len(row) > 1:
                    vectors[row[0]] = np.array(row[1:], dtype=np.float32)
    except FileNotFoundError:
        logger.error(f"FATAL: Song vectors file not found at {path}.")
        return {}
    except Exception as e:
        logger.error(f"An error occurred while loading song vectors: {e}")
        return {}
    
    logger.info(f"Successfully loaded {len(vectors)} song vectors.")
    return vectors
