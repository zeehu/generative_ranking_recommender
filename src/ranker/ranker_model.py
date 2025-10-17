import torch
import torch.nn as nn
from transformers import T5EncoderModel, T5Config
import os
import json

class CrossEncoder(nn.Module):
    def __init__(self, base_model: str, vocab_size: int):
        super().__init__()
        self.base_model_path = base_model
        self.config = T5Config.from_pretrained(base_model)
        self.t5_encoder = T5EncoderModel.from_pretrained(base_model)
        
        # We need to resize embeddings for the semantic ID tokens
        # Note: This assumes the tokenizer used for preparing data has the same custom tokens
        current_embeddings = self.t5_encoder.get_input_embeddings().weight.shape[0]
        if vocab_size > current_embeddings:
            self.t5_encoder.resize_token_embeddings(vocab_size)

        # A simple classification head
        self.classifier = nn.Linear(self.config.d_model, 1)

    def forward(self, input_ids, attention_mask):
        # The last hidden state of the [CLS] token (or the mean of all tokens) is often used
        outputs = self.t5_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # Mean pooling
        pooled_output = outputs.mean(dim=1)
        logits = self.classifier(pooled_output)
        return logits

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        self.t5_encoder.save_pretrained(os.path.join(save_directory, "t5_encoder"))
        torch.save(self.classifier.state_dict(), os.path.join(save_directory, "classifier.pt"))
        with open(os.path.join(save_directory, 'ranker_config.json'), 'w') as f:
            json.dump({'base_model': self.base_model_path, 'vocab_size': self.t5_encoder.get_input_embeddings().weight.shape[0]}, f)

    @classmethod
    def from_pretrained(cls, load_directory: str):
        with open(os.path.join(load_directory, 'ranker_config.json'), 'r') as f:
            config = json.load(f)
        
        model = cls(base_model=config['base_model'], vocab_size=config['vocab_size'])
        model.t5_encoder = T5EncoderModel.from_pretrained(os.path.join(load_directory, "t5_encoder"))
        model.classifier.load_state_dict(torch.load(os.path.join(load_directory, "classifier.pt")))
        return model
