
"""
Step R3 (Refactored): Train the Hybrid Cross-Encoder Ranker Model.
"""
import os
import sys
import logging
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, AutoTokenizer, default_data_collator
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
    """Dataset for the hybrid ranker model training."""
    def __init__(self, data: pd.DataFrame, tokenizer: AutoTokenizer, song_id_to_vector: dict, max_len: int):
        self.data = data
        self.tokenizer = tokenizer
        self.song_id_to_vector = song_id_to_vector
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text, song_id, label = row['text'], str(row['song_id']), row['label']
        
        # Get text encoding
        encoding = self.tokenizer(text, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")
        
        # Get song vector
        song_vector = self.song_id_to_vector.get(song_id, np.zeros(self.song_id_to_vector['__dim__'], dtype=np.float32))

        return {
            "input_ids": encoding.input_ids.flatten(),
            "attention_mask": encoding.attention_mask.flatten(),
            "song_vector": torch.tensor(song_vector, dtype=torch.float),
            "labels": torch.tensor(label, dtype=torch.float)
        }

class RankerTrainer:
    def __init__(self, config: Config):
        self.config = config
        set_seed(config.seed)

    def run(self):
        logger.info("--- Starting Step R3: Hybrid Cross-Encoder Ranker Training ---")
        model_config = self.config.generator_t5 # Reuse T5 config for base model path
        w2v_config = self.config.word2vec

        # Load tokenizer (can be any T5 tokenizer, as we don't use custom tokens here)
        tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)

        # Load song vectors
        logger.info(f"Loading song vectors from {self.config.data.song_vectors_file}...")
        vectors_df = pd.read_csv(self.config.data.song_vectors_file, dtype={'mixsongid': str}).set_index('mixsongid')
        song_id_to_vector = {idx: row.to_numpy() for idx, row in vectors_df.iterrows()}
        song_id_to_vector['__dim__'] = w2v_config.vector_size # Store dimension for zero vector

        # Load training data
        data_path = os.path.join(self.config.output_dir, "ranker", "ranking_train_data.tsv")
        df = pd.read_csv(data_path, sep='\t')

        train_df, eval_df = train_test_split(df, test_size=0.1, random_state=self.config.seed)

        train_dataset = RankingDataset(train_df, tokenizer, song_id_to_vector, model_config.max_input_length)
        eval_dataset = RankingDataset(eval_df, tokenizer, song_id_to_vector, model_config.max_input_length)

        model = CrossEncoder(base_model=model_config.model_name, song_vector_dim=w2v_config.vector_size)

        def compute_metrics(p):
            logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).float().numpy()
            precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='binary')
            acc = accuracy_score(p.label_ids, preds)
            return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

        training_args = TrainingArguments(
            output_dir=os.path.join(self.config.model_dir, "ranker", "checkpoints"),
            num_train_epochs=3,
            per_device_train_batch_size=64, # Can use a larger batch size for ranker
            learning_rate=1e-5,
            fp16=model_config.fp16,
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
            data_collator=default_data_collator # Default collator works for this structure
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
