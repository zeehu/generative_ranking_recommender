"""
Step R3: Train the Cross-Encoder Ranker Model.
"""
import os
import sys
import logging
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config
from src.ranker.ranker_model import CrossEncoder
from src.common.utils import set_seed, setup_logging

logger = logging.getLogger(__name__)

class RankingDataset(Dataset):
    """Dataset for the ranker model training."""
    def __init__(self, data: pd.DataFrame, tokenizer: AutoTokenizer, song_id_to_sem_id: dict, max_len: int):
        self.data = data
        self.tokenizer = tokenizer
        self.song_id_to_sem_id = song_id_to_sem_id
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text, song_id, label = row['text'], str(row['song_id']), row['label']
        
        sem_ids = self.song_id_to_sem_id.get(song_id, [])
        sem_id_tokens = " ".join([f"<id_{sid}>" for sid in sem_ids])
        
        combined_text = f"query: {text} document: {sem_id_tokens}"
        
        encoding = self.tokenizer(combined_text, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")
        
        return {
            "input_ids": encoding.input_ids.flatten(),
            "attention_mask": encoding.attention_mask.flatten(),
            "labels": torch.tensor(label, dtype=torch.float)
        }

class RankerTrainer:
    def __init__(self, config: Config):
        self.config = config
        set_seed(config.seed)

    def run(self):
        logger.info("--- Starting Step R3: Cross-Encoder Ranker Training ---")
        model_config = self.config.generator_t5 # Reuse T5 config for base model path
        
        # Load the tokenizer from the GENERATOR model, as it has the custom <id_..> tokens
        generator_model_path = os.path.join(self.config.model_dir, "generator", "final_model")
        if not os.path.exists(generator_model_path):
            logger.error(f"Generator model not found at {generator_model_path}. Please run G3 first.")
            sys.exit(1)
        tokenizer = AutoTokenizer.from_pretrained(generator_model_path)

        # Load the song->semantic_id map
        with open(self.config.data.semantic_ids_file, 'r') as f:
            song_id_to_sem_id = {item['song_id']: item['semantic_ids'] for item in (json.loads(line) for line in f)}

        # Load training data
        data_path = os.path.join(self.config.output_dir, "ranker", "ranking_train_data.tsv")
        if not os.path.exists(data_path):
            logger.error(f"Ranker training data not found at {data_path}. Please run R1 first.")
            sys.exit(1)
        df = pd.read_csv(data_path, sep='\t')

        # Split data
        train_df, eval_df = train_test_split(df, test_size=0.1, random_state=self.config.seed)

        # Create datasets
        train_dataset = RankingDataset(train_df, tokenizer, song_id_to_sem_id, model_config.max_input_length)
        eval_dataset = RankingDataset(eval_df, tokenizer, song_id_to_sem_id, model_config.max_input_length)

        # Initialize model
        model = CrossEncoder(base_model=model_config.model_name, tokenizer_len=len(tokenizer))

        def compute_metrics(p):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = (torch.sigmoid(torch.tensor(preds)) > 0.5).float().numpy()
            precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='binary')
            acc = accuracy_score(p.label_ids, preds)
            return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

        training_args = TrainingArguments(
            output_dir=os.path.join(self.config.model_dir, "ranker", "checkpoints"),
            num_train_epochs=3, # Ranker fine-tuning usually needs fewer epochs
            per_device_train_batch_size=model_config.per_device_train_batch_size,
            gradient_accumulation_steps=model_config.gradient_accumulation_steps,
            learning_rate=1e-5, # Use a smaller LR for fine-tuning rankers
            fp16=model_config.fp16,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs=model_config.gradient_checkpointing_kwargs,
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_steps=100,
            report_to="none",
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )

        logger.info("Starting ranker training...")
        trainer.train()

        final_model_path = os.path.join(self.config.model_dir, "ranker", "final_model")
        model.save_pretrained(final_model_path)
        logger.info(f"Training complete. Final ranker model saved to {final_model_path}")
        logger.info("--- Step R3 Completed Successfully ---")

if __name__ == "__main__":
    config = Config()
    log_file_path = os.path.join(config.log_dir, "r3_train_ranker.log")
    setup_logging(log_file=log_file_path)
    logger = logging.getLogger(__name__)
    trainer = RankerTrainer(config)
    trainer.run()