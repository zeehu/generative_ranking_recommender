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
        # Load custom config first to get layer_vocab_sizes
        custom_config_path = os.path.join(load_directory, 'custom_tokenizer_config.json')
        if not os.path.exists(custom_config_path):
            raise FileNotFoundError(
                f"Custom tokenizer config not found at {custom_config_path}. "
                "Make sure the model was saved with TIGERTokenizer.save_pretrained()"
            )
        
        with open(custom_config_path, 'r') as f:
            config = json.load(f)
        
        # Load base tokenizer
        base_tokenizer = T5Tokenizer.from_pretrained(load_directory)
        
        # Verify vocab size matches
        if len(base_tokenizer) != config['custom_vocab_size']:
            raise ValueError(
                f"Vocab size mismatch: loaded tokenizer has {len(base_tokenizer)} tokens, "
                f"but config expects {config['custom_vocab_size']} tokens"
            )
        
        # Create tokenizer instance without calling __init__
        tokenizer = cls.__new__(cls)
        tokenizer.base_tokenizer = base_tokenizer
        tokenizer.base_vocab_size = base_tokenizer.vocab_size - config['num_semantic_tokens']
        tokenizer.custom_vocab_size = config['custom_vocab_size']
        tokenizer.layer_vocab_sizes = config['layer_vocab_sizes']
        tokenizer.num_semantic_tokens = config['num_semantic_tokens']
        
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
        """Load a saved TIGER model.
        
        Args:
            load_directory: Directory containing the saved model
            
        Returns:
            TIGERModel instance with loaded weights
        """
        tiger_config_path = os.path.join(load_directory, 'tiger_config.json')
        if not os.path.exists(tiger_config_path):
            raise FileNotFoundError(
                f"TIGER config not found at {tiger_config_path}. "
                "Make sure the model was saved with TIGERModel.save_pretrained()"
            )
        
        with open(tiger_config_path, 'r') as f:
            config = json.load(f)
        
        # Load tokenizer first
        tokenizer = TIGERTokenizer.from_pretrained(load_directory)
        
        # Load the T5 model directly from the saved directory
        t5_model = T5ForConditionalGeneration.from_pretrained(load_directory)
        
        # Verify vocab size matches
        if t5_model.config.vocab_size != len(tokenizer):
            raise ValueError(
                f"Model vocab size ({t5_model.config.vocab_size}) doesn't match "
                f"tokenizer vocab size ({len(tokenizer)})"
            )
        
        # Create TIGER model instance without calling __init__ to avoid re-initialization
        tiger_model = cls.__new__(cls)
        super(TIGERModel, tiger_model).__init__()
        tiger_model.base_model_path = config['base_model']
        tiger_model.config = t5_model.config
        tiger_model.model = t5_model
        tiger_model.tokenizer = tokenizer
        tiger_model.layer_vocab_sizes = config['layer_vocab_sizes']
        
        return tiger_model
