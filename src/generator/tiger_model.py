import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
import json
import os

class TIGERTokenizer:
    def __init__(self, base_model: str, vocab_size: int):
        self.base_tokenizer = T5Tokenizer.from_pretrained(base_model)
        semantic_tokens = [f"<id_{i}>" for i in range(vocab_size)]
        self.base_tokenizer.add_tokens(semantic_tokens)
        
    def __len__(self):
        return len(self.base_tokenizer)
    
    @property
    def pad_token_id(self):
        return self.base_tokenizer.pad_token_id

    def save_pretrained(self, save_directory: str):
        self.base_tokenizer.save_pretrained(save_directory)
        with open(os.path.join(save_directory, 'custom_tokenizer_config.json'), 'w') as f:
            json.dump({'vocab_size': len(self.base_tokenizer) - self.base_tokenizer.vocab_size}, f)

    @classmethod
    def from_pretrained(cls, load_directory: str):
        base_tokenizer = T5Tokenizer.from_pretrained(load_directory)
        with open(os.path.join(load_directory, 'custom_tokenizer_config.json'), 'r') as f:
            config = json.load(f)
        tokenizer = cls.__new__(cls)
        tokenizer.base_tokenizer = base_tokenizer
        return tokenizer

class TIGERModel(nn.Module):
    def __init__(self, base_model: str, vocab_size: int):
        super().__init__()
        self.base_model_path = base_model
        self.config = T5Config.from_pretrained(base_model)
        self.model = T5ForConditionalGeneration.from_pretrained(base_model)
        self.tokenizer = TIGERTokenizer(base_model, vocab_size)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
    def forward(self, **kwargs):
        return self.model(**kwargs)
    
    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        with open(os.path.join(save_directory, 'tiger_config.json'), 'w') as f:
            json.dump({'base_model': self.base_model_path, 'vocab_size': self.tokenizer.vocab_size}, f)

    @classmethod
    def from_pretrained(cls, load_directory: str):
        with open(os.path.join(load_directory, 'tiger_config.json'), 'r') as f:
            config = json.load(f)
        model = cls(base_model=config['base_model'], vocab_size=config['vocab_size'])
        model.model = T5ForConditionalGeneration.from_pretrained(load_directory)
        model.tokenizer = TIGERTokenizer.from_pretrained(load_directory)
        return model
