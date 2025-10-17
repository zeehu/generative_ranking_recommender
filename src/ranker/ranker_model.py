"""
Step R2 (Part 1): Defines the Cross-Encoder model for ranking.
"""
import torch
import torch.nn as nn
from transformers import T5EncoderModel, T5Config
import os
import json

class CrossEncoder(nn.Module):
    """A Cross-Encoder model based on a T5 Encoder for relevance scoring."""
    def __init__(self, base_model: str, tokenizer_len: int):
        super().__init__()
        self.base_model_path = base_model
        self.config = T5Config.from_pretrained(base_model)
        self.t5_encoder = T5EncoderModel.from_pretrained(base_model)
        
        # Resize embeddings for the semantic ID tokens
        self.t5_encoder.resize_token_embeddings(tokenizer_len)

        # A simple classification head for relevance scoring (outputting a single logit)
        self.classifier = nn.Linear(self.config.d_model, 1)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.t5_encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask
        ).last_hidden_state
        
        # Use mean pooling over the sequence length for a fixed-size representation
        pooled_output = outputs.mean(dim=1)
        
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            # Use BCEWithLogitsLoss for binary classification
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.squeeze(), labels.squeeze())

        return {"loss": loss, "logits": logits}

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        self.t5_encoder.save_pretrained(os.path.join(save_directory, "t5_encoder"))
        torch.save(self.classifier.state_dict(), os.path.join(save_directory, "classifier.pt"))
        with open(os.path.join(save_directory, 'ranker_config.json'), 'w') as f:
            json.dump({'base_model': self.base_model_path}, f)

    @classmethod
    def from_pretrained(cls, load_directory: str, tokenizer_len: int):
        with open(os.path.join(load_directory, 'ranker_config.json'), 'r') as f:
            config = json.load(f)
        
        model = cls(base_model=config['base_model'], tokenizer_len=tokenizer_len)
        model.t5_encoder = T5EncoderModel.from_pretrained(os.path.join(load_directory, "t5_encoder"))
        model.classifier.load_state_dict(torch.load(os.path.join(load_directory, "classifier.pt")))
        return model