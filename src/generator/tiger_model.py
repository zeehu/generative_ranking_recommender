"""
Defines the TIGER Model (T5-based) and its custom Tokenizer. 
"""
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
import json
import os

class TIGERTokenizer:
    """Custom tokenizer that adds layer-specific semantic ID tokens to a base T5 tokenizer."""
    def __init__(self, base_model_path: str, layer_vocab_sizes: dict):
        """
        Args:
            base_model_path: Path to the base T5 model
            layer_vocab_sizes: Dict with keys 'l1', 'l2', 'l3' and their respective vocab sizes
                Example: {'l1': 128, 'l2': 128, 'l3': 256}
        """
        self.base_tokenizer = T5Tokenizer.from_pretrained(base_model_path)
        self.base_vocab_size = self.base_tokenizer.vocab_size

        # Add layer-specific semantic ID tokens
        semantic_tokens = []
        
        # Layer 1 tokens: <id_l1_0>, <id_l1_1>, ..., <id_l1_127>
        for i in range(layer_vocab_sizes['l1']):
            semantic_tokens.append(f"<id_l1_{i}>")
        
        # Layer 2 tokens: <id_l2_0>, <id_l2_1>, ..., <id_l2_127>
        for i in range(layer_vocab_sizes['l2']):
            semantic_tokens.append(f"<id_l2_{i}>")
        
        # Layer 3 tokens: <id_l3_0>, <id_l3_1>, ..., <id_l3_255>
        for i in range(layer_vocab_sizes['l3']):
            semantic_tokens.append(f"<id_l3_{i}>")
        
        # Add special tokens
        special_tokens = ["<eos>", "<sep>"]
        semantic_tokens.extend(special_tokens)
        
        # Add all tokens to tokenizer
        self.base_tokenizer.add_tokens(semantic_tokens)
        
        self.custom_vocab_size = len(self.base_tokenizer)
        self.layer_vocab_sizes = layer_vocab_sizes
        self.num_semantic_tokens = len(semantic_tokens)

    def __len__(self):
        return self.custom_vocab_size
    
    @property
    def pad_token_id(self):
        return self.base_tokenizer.pad_token_id

    def save_pretrained(self, save_directory: str):
        self.base_tokenizer.save_pretrained(save_directory)
        # Save our custom config
        with open(os.path.join(save_directory, 'custom_tokenizer_config.json'), 'w') as f:
            json.dump({
                'custom_vocab_size': self.custom_vocab_size,
                'layer_vocab_sizes': self.layer_vocab_sizes,
                'num_semantic_tokens': self.num_semantic_tokens
            }, f)

    @classmethod
    def from_pretrained(cls, load_directory: str):
        base_tokenizer = T5Tokenizer.from_pretrained(load_directory)
        tokenizer = cls.__new__(cls)
        tokenizer.base_tokenizer = base_tokenizer
        
        # Load custom config
        custom_config_path = os.path.join(load_directory, 'custom_tokenizer_config.json')
        if os.path.exists(custom_config_path):
            with open(custom_config_path, 'r') as f:
                config = json.load(f)
            tokenizer.custom_vocab_size = config['custom_vocab_size']
            tokenizer.layer_vocab_sizes = config['layer_vocab_sizes']
            tokenizer.num_semantic_tokens = config['num_semantic_tokens']
        else:
            tokenizer.custom_vocab_size = len(base_tokenizer)
        
        return tokenizer

class TIGERModel(nn.Module):
    """A wrapper for the T5 model and its custom tokenizer."""
    
    def __init__(self, base_model: str, layer_vocab_sizes: dict):
        """
        Args:
            base_model: Path to the base T5 model
            layer_vocab_sizes: Dict with keys 'l1', 'l2', 'l3' and their respective vocab sizes
        """
        super().__init__()
        self.base_model_path = base_model
        self.config = T5Config.from_pretrained(base_model)
        self.model = T5ForConditionalGeneration.from_pretrained(base_model)
        self.tokenizer = TIGERTokenizer(base_model, layer_vocab_sizes)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.layer_vocab_sizes = layer_vocab_sizes
        
    def forward(self, **kwargs):
        return self.model(**kwargs)
    
    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        # Save our main config
        config_dict = {
            'base_model': self.base_model_path,
            'layer_vocab_sizes': self.layer_vocab_sizes
        }
        with open(os.path.join(save_directory, 'tiger_config.json'), 'w') as f:
            json.dump(config_dict, f)

    @classmethod
    def from_pretrained(cls, load_directory: str):
        with open(os.path.join(load_directory, 'tiger_config.json'), 'r') as f:
            config = json.load(f)
        
        model = cls(base_model=config['base_model'], layer_vocab_sizes=config['layer_vocab_sizes'])
        model.model = T5ForConditionalGeneration.from_pretrained(load_directory)
        model.tokenizer = TIGERTokenizer.from_pretrained(load_directory)
        return model