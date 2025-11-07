"""
Step G3: Train the T5 Generator Model (Optimized Version)
使用预处理的tokenized数据 + 内存映射技术，大幅提升训练速度
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

try:
    from config_optimized import Config
    logger = logging.getLogger(__name__)
    logger.info("Using optimized configuration")
except ImportError:
    from config import Config
    logger = logging.getLogger(__name__)
    logger.warning("config_optimized.py not found, using default config.py")
from src.generator.tiger_model import TIGERModel
from src.common.utils import set_seed, setup_logging

logger = logging.getLogger(__name__)


class MemoryMappedDataset(Dataset):
    """
    使用HuggingFace datasets的内存映射功能
    数据存储在磁盘，按需加载，内存占用极小
    """
    def __init__(self, dataset_path: str):
        logger.info(f"Loading memory-mapped dataset from {dataset_path}...")
        self.dataset = load_from_disk(dataset_path)
        logger.info(f"Loaded {len(self.dataset):,} samples (memory-mapped)")
        logger.info(f"Dataset features: {self.dataset.features}")

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
        logger.info("=" * 80)
        logger.info("Step G3: T5 Generator Model Training (OPTIMIZED)")
        logger.info("=" * 80)
        
        model_config = self.config.generator_t5
        rq_config = self.config.h_rqkmeans

        # Calculate the layer-specific vocab sizes for semantic ID tokens
        layer_vocab_sizes = {
            'l1': rq_config.need_clusters[0],
            'l2': rq_config.need_clusters[1],
            'l3': rq_config.need_clusters[2],
        }
        
        logger.info(f"Layer vocab sizes: {layer_vocab_sizes}")
        logger.info(f"Total semantic ID tokens: {sum(layer_vocab_sizes.values())} + 2 special tokens")
        
        # Initialize TIGERModel and TIGERTokenizer
        logger.info("Initializing TIGER model...")
        model = TIGERModel(base_model=model_config.model_name, layer_vocab_sizes=layer_vocab_sizes)
        model.model.config.use_cache = False  # Necessary for gradient checkpointing
        tokenizer = model.tokenizer
        
        logger.info(f"Tokenizer vocab size: {len(tokenizer)}")

        # 使用预处理的tokenized数据
        train_dataset_path = os.path.join(self.config.output_dir, "generator", "train_tokenized")
        val_dataset_path = os.path.join(self.config.output_dir, "generator", "val_tokenized")
        
        # 检查预处理数据是否存在
        if not os.path.exists(train_dataset_path):
            logger.error(f"预处理数据不存在: {train_dataset_path}")
            logger.error("请先运行: python src/generator/preprocess_tokenize.py")
            raise FileNotFoundError(f"预处理数据不存在，请先运行预处理脚本")
        
        if not os.path.exists(val_dataset_path):
            logger.error(f"预处理数据不存在: {val_dataset_path}")
            logger.error("请先运行: python src/generator/preprocess_tokenize.py")
            raise FileNotFoundError(f"预处理数据不存在，请先运行预处理脚本")
        
        # 加载内存映射数据集
        train_dataset = MemoryMappedDataset(train_dataset_path)
        val_dataset = MemoryMappedDataset(val_dataset_path)

        # 优化后的训练参数
        training_args = TrainingArguments(
            output_dir=os.path.join(self.config.model_dir, "generator", "checkpoints"),
            num_train_epochs=model_config.num_train_epochs,
            per_device_train_batch_size=model_config.per_device_train_batch_size,
            per_device_eval_batch_size=model_config.per_device_eval_batch_size,
            gradient_accumulation_steps=model_config.gradient_accumulation_steps,
            learning_rate=model_config.learning_rate,
            warmup_steps=model_config.warmup_steps,
            weight_decay=model_config.weight_decay,
            fp16=model_config.fp16,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs=model_config.gradient_checkpointing_kwargs,
            max_grad_norm=1.0,
            
            # 评估和保存策略
            eval_strategy="steps",
            eval_steps=2000,  # 优化: 减少评估频率
            save_strategy="steps",
            save_steps=2000,  # 优化: 减少保存频率
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # 日志
            logging_steps=100,
            report_to="none",
            
            # 数据加载优化
            dataloader_num_workers=self.config.num_workers,
            dataloader_pin_memory=True,  # 优化: 启用pin_memory加速GPU传输
            
            # 分布式训练优化
            ddp_find_unused_parameters=False,  # 优化: 加速DDP
            
            # 其他
            remove_unused_columns=False,
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
        
        # torch.compile优化 (可选)
        # 注意: 首次运行会有编译开销，但长期训练会提速5-15%
        if torch.__version__ >= "2.0.0" and model_config.use_torch_compile:
            logger.info("Enabling torch.compile for model optimization...")
            logger.info("Note: First few iterations will be slower due to compilation")
            model.model = torch.compile(model.model)
        else:
            logger.info("torch.compile disabled for faster startup")

        # 打印训练配置摘要
        logger.info("\n" + "=" * 80)
        logger.info("Training Configuration Summary")
        logger.info("=" * 80)
        logger.info(f"Training samples: {len(train_dataset):,}")
        logger.info(f"Validation samples: {len(val_dataset):,}")
        logger.info(f"Per-device batch size: {model_config.per_device_train_batch_size}")
        logger.info(f"Gradient accumulation steps: {model_config.gradient_accumulation_steps}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        
        effective_batch_size = (
            model_config.per_device_train_batch_size * 
            model_config.gradient_accumulation_steps * 
            torch.cuda.device_count()
        )
        logger.info(f"Effective batch size: {effective_batch_size}")
        
        steps_per_epoch = len(train_dataset) // effective_batch_size
        total_steps = steps_per_epoch * model_config.num_train_epochs
        logger.info(f"Steps per epoch: {steps_per_epoch}")
        logger.info(f"Total training steps: {total_steps}")
        logger.info(f"DataLoader workers: {self.config.num_workers}")
        logger.info("=" * 80 + "\n")

        logger.info("Starting training...")
        trainer.train()

        # 保存最终模型
        final_model_path = os.path.join(self.config.model_dir, "generator", "final_model")
        model.save_pretrained(final_model_path)
        
        logger.info("\n" + "=" * 80)
        logger.info("Training Complete!")
        logger.info("=" * 80)
        logger.info(f"Final model saved to: {final_model_path}")
        logger.info("=" * 80)


if __name__ == "__main__":
    config = Config()
    log_file_path = os.path.join(config.log_dir, "g3_train_t5_optimized.log")
    setup_logging(log_file=log_file_path)
    logger = logging.getLogger(__name__)
    
    trainer = T5TrainerOptimized(config)
    trainer.run()
