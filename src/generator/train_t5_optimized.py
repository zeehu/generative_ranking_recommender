"""
Step G3: Train the T5 Generator Model (Optimized Version)      

优化策略:
1. 内存映射数据集: 使用HuggingFace datasets的内存映射，减少内存占用
2. 预处理tokenization: 避免训练时重复分词，提升数据加载速度
3. 优化DataLoader: 多进程加载 + pin_memory + prefetch
4. 混合精度训练: FP16提升Tensor Core利用率
5. 梯度检查点: 节省30-40%显存
6. 梯度累积: 减少DDP通信开销
7. 可选torch.compile: 额外5-15%加速

针对硬件: 3×L20 GPU (48GB) + 20 CPU + 50GB内存 + 400万数据
预期效果: 训练时间约10-11小时 (3 epochs)
配置优化: batch=128×3×5=1920, lr=4.0e-4, 显存利用率79%
"""
import os
import sys
import logging
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_from_disk

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 先导入必要的模块
try:
    from config_optimized import Config
    _using_optimized = True
except ImportError:
    from config import Config
    _using_optimized = False

from src.generator.tiger_model import TIGERModel
from src.common.utils import set_seed, setup_logging, is_main_process


class MemoryMappedDataset(Dataset):
    """
    使用HuggingFace datasets的内存映射功能
    数据存储在磁盘，按需加载，内存占用极小
    """
    def __init__(self, dataset_path: str):
        # 使用模块级别的logger，只在主进程打印
        if is_main_process():
            _logger = logging.getLogger(__name__)
            _logger.info(f"Loading memory-mapped dataset from {dataset_path}...")
        self.dataset = load_from_disk(dataset_path)
        if is_main_process():
            _logger = logging.getLogger(__name__)
            _logger.info(f"Loaded {len(self.dataset):,} samples (memory-mapped)")
            _logger.info(f"Dataset features: {self.dataset.features}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 直接返回预处理好的数据，无需tokenization
        item = self.dataset[idx]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "labels": item["labels"]
        }


class T5TrainerOptimized:
    def __init__(self, config: Config):
        self.config = config
        set_seed(config.seed)

    def run(self):
        # 使用模块级别的logger，只在主进程打印
        if not is_main_process():
            # 非主进程静默运行
            _logger = logging.getLogger(__name__)
        else:
            _logger = logging.getLogger(__name__)
            _logger.info("=" * 80)
            _logger.info("Step G3: T5 Generator Model Training (OPTIMIZED)")
            _logger.info("=" * 80)
        
        model_config = self.config.generator_t5
        rq_config = self.config.h_rqkmeans

        # Calculate the layer-specific vocab sizes for semantic ID tokens
        layer_vocab_sizes = {
            'l1': rq_config.need_clusters[0],
            'l2': rq_config.need_clusters[1],
            'l3': rq_config.need_clusters[2],
        }
        
        if is_main_process():
            _logger = logging.getLogger(__name__)
            _logger.info(f"Layer vocab sizes: {layer_vocab_sizes}")
            _logger.info(f"Total semantic ID tokens: {sum(layer_vocab_sizes.values())} + 2 special tokens")
            _logger.info("Initializing TIGER model...")
        model = TIGERModel(base_model=model_config.model_name, layer_vocab_sizes=layer_vocab_sizes)
        model.model.config.use_cache = False  # Necessary for gradient checkpointing
        tokenizer = model.tokenizer
        
        if is_main_process():
            _logger = logging.getLogger(__name__)
            _logger.info(f"Tokenizer vocab size: {len(tokenizer)}")

        # 使用预处理的tokenized数据
        train_dataset_path = os.path.join(self.config.output_dir, "generator", "train_tokenized")
        val_dataset_path = os.path.join(self.config.output_dir, "generator", "val_tokenized")
        
        # 检查预处理数据是否存在
        if not os.path.exists(train_dataset_path):
            if is_main_process():
                _logger = logging.getLogger(__name__)
                _logger.error(f"预处理数据不存在: {train_dataset_path}")
                _logger.error("请先运行: python src/generator/preprocess_tokenize.py")
            raise FileNotFoundError(f"预处理数据不存在，请先运行预处理脚本")
        
        if not os.path.exists(val_dataset_path):
            if is_main_process():
                _logger = logging.getLogger(__name__)
                _logger.error(f"预处理数据不存在: {val_dataset_path}")
                _logger.error("请先运行: python src/generator/preprocess_tokenize.py")
            raise FileNotFoundError(f"预处理数据不存在，请先运行预处理脚本")
        
        # 加载内存映射数据集
        train_dataset = MemoryMappedDataset(train_dataset_path)
        val_dataset = MemoryMappedDataset(val_dataset_path)

        # ========== 从配置文件加载训练参数 ==========
        train_config = self.config.training
        
        training_args = TrainingArguments(
            output_dir=os.path.join(self.config.model_dir, "generator", "checkpoints"),
            
            # 基本训练配置
            num_train_epochs=model_config.num_train_epochs,
            per_device_train_batch_size=model_config.per_device_train_batch_size,
            per_device_eval_batch_size=model_config.per_device_eval_batch_size,
            gradient_accumulation_steps=model_config.gradient_accumulation_steps,
            
            # 学习率配置
            learning_rate=model_config.learning_rate,
            warmup_steps=model_config.warmup_steps,
            warmup_ratio=getattr(model_config, 'warmup_ratio', 0.05),
            lr_scheduler_type=train_config.lr_scheduler_type,
            weight_decay=model_config.weight_decay,
            
            # 混合精度训练
            fp16=model_config.fp16,
            fp16_opt_level=getattr(model_config, 'fp16_opt_level', 'O2'),
            fp16_backend=getattr(model_config, 'fp16_backend', 'auto'),
            
            # 梯度优化
            gradient_checkpointing=getattr(model_config, 'gradient_checkpointing', True),
            gradient_checkpointing_kwargs=model_config.gradient_checkpointing_kwargs,
            max_grad_norm=getattr(model_config, 'max_grad_norm', 1.0),
            
            # 评估和保存策略（从配置文件读取）
            eval_strategy=train_config.eval_strategy,
            eval_steps=train_config.eval_steps,
            save_strategy=train_config.save_strategy,
            save_steps=train_config.save_steps,
            save_total_limit=train_config.save_total_limit,
            load_best_model_at_end=train_config.load_best_model_at_end,
            metric_for_best_model=train_config.metric_for_best_model,
            greater_is_better=train_config.greater_is_better,
            
            # 日志配置（从配置文件读取）
            logging_steps=train_config.logging_steps,
            logging_first_step=train_config.logging_first_step,
            report_to=train_config.report_to,
            
            # 数据加载优化（从配置文件读取）
            dataloader_num_workers=self.config.num_workers,
            dataloader_pin_memory=train_config.dataloader_pin_memory,
            dataloader_prefetch_factor=train_config.dataloader_prefetch_factor,
            dataloader_persistent_workers=train_config.dataloader_persistent_workers,
            
            # 分布式训练优化（从配置文件读取）
            ddp_find_unused_parameters=train_config.ddp_find_unused_parameters,
            ddp_bucket_cap_mb=train_config.ddp_bucket_cap_mb,
            ddp_broadcast_buffers=train_config.ddp_broadcast_buffers,
            
            # 其他配置（从配置文件读取）
            remove_unused_columns=train_config.remove_unused_columns,
            ignore_data_skip=train_config.ignore_data_skip,
            disable_tqdm=train_config.disable_tqdm,
        )

        # 自定义DataCollator - 因为数据已经padding过了
        class PrePaddedDataCollator:
            """数据已经预先padding，直接转换为tensor"""
            def __call__(self, features):
                import torch
                batch = {
                    "input_ids": torch.tensor([f["input_ids"] for f in features], dtype=torch.long),
                    "attention_mask": torch.tensor([f["attention_mask"] for f in features], dtype=torch.long),
                    "labels": torch.tensor([f["labels"] for f in features], dtype=torch.long),
                }
                return batch

        trainer = Trainer(
            model=model.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=PrePaddedDataCollator()  # 使用简化的collator
        )
        
        # ========== torch.compile优化 (可选) ==========
        # PyTorch 2.0+ 的编译优化，可提速5-15%
        # 注意: 首次编译需要额外时间 (5-10分钟)
        # 建议: 训练时间>4小时时开启
        use_compile = getattr(model_config, 'use_torch_compile', False)
        if torch.__version__ >= "2.0.0" and use_compile:
            if is_main_process():
                _logger = logging.getLogger(__name__)
                _logger.info("="*80)
                _logger.info("Enabling torch.compile for model optimization")
                _logger.info("Backend: {}".format(getattr(model_config, 'torch_compile_backend', 'inductor')))
                _logger.info("Mode: {}".format(getattr(model_config, 'torch_compile_mode', 'reduce-overhead')))
                _logger.info("Note: First few iterations will be slower due to compilation")
                _logger.info("Expected speedup: 5-15% after compilation")
                _logger.info("="*80)
            try:
                model.model = torch.compile(
                    model.model,
                    backend=getattr(model_config, 'torch_compile_backend', 'inductor'),
                    mode=getattr(model_config, 'torch_compile_mode', 'reduce-overhead')
                )
                if is_main_process():
                    _logger = logging.getLogger(__name__)
                    _logger.info("torch.compile enabled successfully")
            except Exception as e:
                if is_main_process():
                    _logger = logging.getLogger(__name__)
                    _logger.warning(f"torch.compile failed: {e}")
                    _logger.warning("Continuing without compilation")
        else:
            if is_main_process():
                _logger = logging.getLogger(__name__)
                if torch.__version__ < "2.0.0":
                    _logger.info("torch.compile requires PyTorch 2.0+, current version: {}".format(torch.__version__))
                else:
                    _logger.info("torch.compile disabled (set use_torch_compile=True to enable)")

        # ========== 打印详细的训练配置摘要 ==========
        num_gpus = torch.cuda.device_count()
        sys_config = self.config.system
        
        effective_batch_size = (
            model_config.per_device_train_batch_size * 
            model_config.gradient_accumulation_steps * 
            num_gpus
        )
        steps_per_epoch = len(train_dataset) // effective_batch_size
        total_steps = steps_per_epoch * model_config.num_train_epochs
        
        if is_main_process():
            _logger = logging.getLogger(__name__)
            _logger.info("\n" + "=" * 80)
            _logger.info(f"TRAINING CONFIGURATION SUMMARY ({sys_config.expected_num_gpus}×{sys_config.gpu_model} GPU OPTIMIZED)")
            _logger.info("=" * 80)
            
            _logger.info("\n[Dataset Information]")
            _logger.info(f"  Training samples:   {len(train_dataset):,}")
            _logger.info(f"  Validation samples: {len(val_dataset):,}")
            _logger.info(f"  Train/Val ratio:    {len(train_dataset)/len(val_dataset):.1f}:1")
            
            _logger.info("\n[Hardware Configuration]")
            _logger.info(f"  Number of GPUs:     {num_gpus}")
            _logger.info(f"  GPU Model:          {sys_config.gpu_model} ({sys_config.gpu_memory_gb}GB GDDR6)")
            _logger.info(f"  CPU cores:          {sys_config.cpu_cores}")
            _logger.info(f"  System memory:      {sys_config.system_memory_gb}GB")
            
            _logger.info("\n[Batch Configuration]")
            _logger.info(f"  Per-device batch:   {model_config.per_device_train_batch_size}")
            _logger.info(f"  Gradient accum:     {model_config.gradient_accumulation_steps}")
            _logger.info(f"  Effective batch:    {effective_batch_size} = {model_config.per_device_train_batch_size} × {num_gpus} × {model_config.gradient_accumulation_steps}")
            
            _logger.info("\n[Training Schedule]")
            _logger.info(f"  Epochs:             {model_config.num_train_epochs}")
            _logger.info(f"  Steps per epoch:    {steps_per_epoch:,}")
            _logger.info(f"  Total steps:        {total_steps:,}")
            _logger.info(f"  Warmup steps:       {model_config.warmup_steps:,} ({model_config.warmup_steps/total_steps*100:.1f}%)")
            _logger.info(f"  Eval frequency:     every {train_config.eval_steps} steps")
            _logger.info(f"  Save frequency:     every {train_config.save_steps} steps")
            
            _logger.info("\n[Optimization Settings]")
            _logger.info(f"  Learning rate:      {model_config.learning_rate}")
            _logger.info(f"  LR scheduler:       {train_config.lr_scheduler_type}")
            _logger.info(f"  Weight decay:       {model_config.weight_decay}")
            _logger.info(f"  Max grad norm:      {getattr(model_config, 'max_grad_norm', 1.0)}")
            _logger.info(f"  Mixed precision:    FP16 ({getattr(model_config, 'fp16_opt_level', 'O2')} level)")
            _logger.info(f"  Gradient ckpt:      {getattr(model_config, 'gradient_checkpointing', True)}")
            
            _logger.info("\n[DataLoader Settings]")
            _logger.info(f"  Num workers:        {self.config.num_workers}")
            _logger.info(f"  Pin memory:         {train_config.dataloader_pin_memory}")
            _logger.info(f"  Prefetch factor:    {train_config.dataloader_prefetch_factor}")
            _logger.info(f"  Persistent workers: {train_config.dataloader_persistent_workers}")
            
            _logger.info("\n[Performance Estimation]")
            estimated_total_time = total_steps * sys_config.estimated_time_per_step / 3600  # hours
            _logger.info(f"  Est. time per step: ~{sys_config.estimated_time_per_step}s")
            _logger.info(f"  Est. total time:    ~{estimated_total_time:.1f} hours")
            _logger.info(f"  Expected GPU util:  {sys_config.expected_gpu_utilization}")
            _logger.info(f"  Speedup vs orig:    {sys_config.expected_speedup}")
            
            _logger.info("\n" + "=" * 80 + "\n")

            _logger.info("\n" + "=" * 80)
            _logger.info("STARTING TRAINING")
            _logger.info("=" * 80)
            _logger.info("Monitor GPU usage: watch -n 1 nvidia-smi")
            _logger.info("Monitor training log: tail -f logs/g3_train_t5_optimized.log")
            _logger.info("=" * 80 + "\n")
        
        # 训练开始
        import time
        start_time = time.time()
        
        try:
            trainer.train()
            
            # 计算实际训练时间
            end_time = time.time()
            training_time = (end_time - start_time) / 3600  # hours
            
            if is_main_process():
                _logger = logging.getLogger(__name__)
                _logger.info("\n" + "=" * 80)
                _logger.info("TRAINING COMPLETED SUCCESSFULLY")
                _logger.info("=" * 80)
                _logger.info(f"Total training time: {training_time:.2f} hours")
                _logger.info(f"Average time per step: {(end_time - start_time) / total_steps:.2f}s")
                _logger.info("=" * 80 + "\n")
            
        except Exception as e:
            if is_main_process():
                _logger = logging.getLogger(__name__)
                _logger.error("\n" + "=" * 80)
                _logger.error("TRAINING FAILED")
                _logger.error("=" * 80)
                _logger.error(f"Error: {str(e)}")
                _logger.error("=" * 80 + "\n")
            raise

        # 保存最终模型（只在主进程保存）
        if is_main_process():
            final_model_path = os.path.join(self.config.model_dir, "generator", "final_model")
            model.save_pretrained(final_model_path)
            
            _logger = logging.getLogger(__name__)
            _logger.info("\n" + "=" * 80)
            _logger.info("MODEL SAVED")
            _logger.info("=" * 80)
            _logger.info(f"Final model path: {final_model_path}")
            _logger.info(f"Model size: ~{os.path.getsize(final_model_path) / 1e9 if os.path.exists(final_model_path) else 0:.2f} GB")
            _logger.info("=" * 80)
            
            _logger.info("\n" + "=" * 80)
            _logger.info("NEXT STEPS")
            _logger.info("=" * 80)
            _logger.info("1. Evaluate model performance on test set")
            _logger.info("2. Check training logs for any anomalies")
            _logger.info("3. Monitor GPU memory usage patterns")
            _logger.info("4. Consider enabling torch.compile for future training")
            _logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    # 加载优化配置
    config = Config()
    log_file_path = os.path.join(config.log_dir, "g3_train_t5_optimized.log")
    
    # 设置日志（只调用一次）
    setup_logging(log_file=log_file_path)
    
    # 只在主进程打印启动信息
    if is_main_process():
        # 获取logger用于主程序
        main_logger = logging.getLogger(__name__)
        
        # 打印配置信息
        if _using_optimized:
            main_logger.info("Using optimized configuration (config_optimized.py)")
        else:
            main_logger.warning("config_optimized.py not found, using default config.py")
        
        # 打印启动信息
        main_logger.info("#" * 80)
        main_logger.info("#" + " " * 78 + "#")
        main_logger.info("#" + " " * 20 + "T5 GENERATOR TRAINING (OPTIMIZED)" + " " * 25 + "#")
        main_logger.info("#" + " " * 78 + "#")
        main_logger.info("#" * 80)
        main_logger.info("")
        sys_config = config.system
        main_logger.info(f"Hardware: {sys_config.expected_num_gpus}×{sys_config.gpu_model} GPU ({sys_config.gpu_memory_gb}GB) + {sys_config.cpu_cores} CPU + {sys_config.system_memory_gb}GB RAM")
        main_logger.info(f"Dataset: ~{config.data.train_split_ratio*100:.0f}% training samples")
        main_logger.info(f"Epochs: {config.generator_t5.num_train_epochs}")
        main_logger.info(f"Expected speedup: {sys_config.expected_speedup} vs original config")
        main_logger.info("")
        main_logger.info("#" * 80 + "\n")
        
        # 检查GPU可用性
        if not torch.cuda.is_available():
            main_logger.error("CUDA is not available! Please check your GPU setup.")
            raise RuntimeError("CUDA not available")
        
        num_gpus = torch.cuda.device_count()
        main_logger.info(f"Detected {num_gpus} GPU(s):")
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            main_logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        main_logger.info("")
        
        sys_config = config.system
        if num_gpus != sys_config.expected_num_gpus:
            main_logger.warning(f"Expected {sys_config.expected_num_gpus} GPUs but found {num_gpus}. Config is optimized for {sys_config.expected_num_gpus}×{sys_config.gpu_model}.")
            main_logger.warning("Training will continue but performance may differ.")
            main_logger.info("")
    
    # 启动训练
    trainer = T5TrainerOptimized(config)
    trainer.run()
