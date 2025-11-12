"""
TIGER Model (T5-based) with 3-layer Semantic IDs
Defines the TIGER Model and its custom Tokenizer with layer-specific semantic ID tokens.
"""
import torch
import torch.nn as nn
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer, 
    T5Config
)
from typing import List, Dict, Optional, Tuple
import json
import os


class TIGERTokenizer:
    """Custom tokenizer for TIGER model with layer-specific semantic IDs"""
    
    def __init__(self, base_model: str, layer_vocab_sizes: dict):
        """
        Args:
            base_model: Path to the base T5 model (e.g., "t5-small")
            layer_vocab_sizes: Dict with keys 'l1', 'l2', 'l3' and their respective vocab sizes
                Example: {'l1': 128, 'l2': 128, 'l3': 256}
        """
        self.base_tokenizer = T5Tokenizer.from_pretrained(base_model)
        self.base_vocab_size = self.base_tokenizer.vocab_size
        self.layer_vocab_sizes = layer_vocab_sizes
        
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
        
        # Add special tokens (aligned with GR-movie-recommendation)
        special_tokens = ["<user>", "<eos>", "<unk>", "<pad>", "<mask>", "<sep>"]
        semantic_tokens.extend(special_tokens)
        
        # Add all tokens to tokenizer
        self.base_tokenizer.add_tokens(semantic_tokens)
        
        self.custom_vocab_size = len(self.base_tokenizer)
        self.num_semantic_tokens = len(semantic_tokens)
        
        # Create token mappings for layer-specific semantic IDs
        self.semantic_id_to_token = {}
        self.token_to_semantic_id = {}
        for layer, size in layer_vocab_sizes.items():
            for i in range(size):
                token = f"<id_{layer}_{i}>"
                self.semantic_id_to_token[(layer, i)] = token
                self.token_to_semantic_id[token] = (layer, i)
    
    def encode(self, text: str, **kwargs) -> List[int]:
        """Encode text to token IDs"""
        return self.base_tokenizer.encode(text, **kwargs)
    
    def decode(self, token_ids: List[int], **kwargs) -> str:
        """Decode token IDs to text"""
        return self.base_tokenizer.decode(token_ids, **kwargs)
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text"""
        return self.base_tokenizer.tokenize(text)
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to IDs"""
        return self.base_tokenizer.convert_tokens_to_ids(tokens)
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert IDs to tokens"""
        return self.base_tokenizer.convert_ids_to_tokens(ids)
    
    def __len__(self):
        return self.custom_vocab_size
    
    @property
    def pad_token_id(self):
        return self.base_tokenizer.pad_token_id
    
    @property
    def eos_token_id(self):
        return self.base_tokenizer.eos_token_id
    
    def save_pretrained(self, save_directory: str):
        """Save tokenizer"""
        self.base_tokenizer.save_pretrained(save_directory)
        
        # Save custom mappings
        mappings = {
            'semantic_id_to_token': {str(k): v for k, v in self.semantic_id_to_token.items()},
            'token_to_semantic_id': {k: list(v) for k, v in self.token_to_semantic_id.items()},
            'custom_vocab_size': self.custom_vocab_size,
            'layer_vocab_sizes': self.layer_vocab_sizes,
            'num_semantic_tokens': self.num_semantic_tokens
        }
        
        with open(os.path.join(save_directory, 'custom_mappings.json'), 'w') as f:
            json.dump(mappings, f)
    
    @classmethod
    def from_pretrained(cls, load_directory: str):
        """Load tokenizer"""
        # Load base tokenizer
        base_tokenizer = T5Tokenizer.from_pretrained(load_directory)
        
        # Load custom mappings
        mappings_path = os.path.join(load_directory, 'custom_mappings.json')
        if not os.path.exists(mappings_path):
            raise FileNotFoundError(
                f"Custom mappings not found at {mappings_path}. "
                "Make sure the model was saved with TIGERTokenizer.save_pretrained()"
            )
        
        with open(mappings_path, 'r') as f:
            mappings = json.load(f)
        
        # Verify vocab size matches
        if len(base_tokenizer) != mappings['custom_vocab_size']:
            raise ValueError(
                f"Vocab size mismatch: loaded tokenizer has {len(base_tokenizer)} tokens, "
                f"but config expects {mappings['custom_vocab_size']} tokens"
            )
        
        # Create instance
        tokenizer = cls.__new__(cls)
        tokenizer.base_tokenizer = base_tokenizer
        tokenizer.base_vocab_size = base_tokenizer.vocab_size - mappings['num_semantic_tokens']
        tokenizer.custom_vocab_size = mappings['custom_vocab_size']
        tokenizer.layer_vocab_sizes = mappings['layer_vocab_sizes']
        tokenizer.num_semantic_tokens = mappings['num_semantic_tokens']
        
        # Restore mappings (convert string keys back to tuples for semantic_id_to_token)
        tokenizer.semantic_id_to_token = {eval(k): v for k, v in mappings['semantic_id_to_token'].items()}
        tokenizer.token_to_semantic_id = {k: tuple(v) for k, v in mappings['token_to_semantic_id'].items()}
        
        return tokenizer


class TIGERModel(nn.Module):
    """TIGER: T5-based Generative Recommendation Model with 3-layer Semantic IDs"""
    
    def __init__(self, base_model: str, layer_vocab_sizes: dict):
        """
        Args:
            base_model: Path to the base T5 model (e.g., "t5-small")
            layer_vocab_sizes: Dict with keys 'l1', 'l2', 'l3' and their respective vocab sizes
        """
        super().__init__()
        
        # Load base T5 model
        self.config = T5Config.from_pretrained(base_model)
        self.model = T5ForConditionalGeneration.from_pretrained(base_model)
        
        # Initialize tokenizer and resize embeddings
        self.tokenizer = TIGERTokenizer(base_model, layer_vocab_sizes)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Store configuration
        self.base_model_path = base_model
        self.layer_vocab_sizes = layer_vocab_sizes
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None,
                labels: torch.Tensor = None, **kwargs):
        """Forward pass"""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
    
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None,
                max_new_tokens: int = 10, num_beams: int = 5, 
                num_return_sequences: int = 1, **kwargs):
        """Generate recommendations"""
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            **kwargs
        )
    
    def recommend(self, user_sequence: List[str], num_recommendations: int = 10,
                 num_beams: int = 10) -> List[List[Tuple[int, int, int]]]:
        """
        Generate recommendations for a user sequence
        
        Args:
            user_sequence: List of input text tokens
            num_recommendations: Number of recommendations to generate
            num_beams: Number of beams for beam search
            
        Returns:
            List of semantic ID tuples (l1, l2, l3)
        """
        self.eval()
        
        # Prepare input
        input_text = " ".join(user_sequence)
        input_ids = torch.tensor([self.tokenizer.encode(input_text)]).to(next(self.parameters()).device)
        
        # Generate
        with torch.no_grad():
            outputs = self.generate(
                input_ids=input_ids,
                max_new_tokens=6,  # Generate 6 tokens (2 semantic ID tuples: 3 tokens each)
                num_beams=num_beams,
                num_return_sequences=num_recommendations
            )
        
        # Decode recommendations
        recommendations = []
        for output in outputs:
            # Get generated tokens (skip input)
            generated_tokens = output[input_ids.shape[1]:]
            decoded_tokens = self.tokenizer.convert_ids_to_tokens(generated_tokens)
            
            # Extract semantic ID tuples
            semantic_ids = []
            i = 0
            while i < len(decoded_tokens):
                token = decoded_tokens[i]
                
                # Check if this is a layer 1 token
                if token.startswith("<id_l1_"):
                    # Try to extract complete 3-layer semantic ID
                    if i + 2 < len(decoded_tokens):
                        l1_token = decoded_tokens[i]
                        l2_token = decoded_tokens[i + 1]
                        l3_token = decoded_tokens[i + 2]
                        
                        # Validate all three are semantic ID tokens
                        if (l1_token.startswith("<id_l1_") and 
                            l2_token.startswith("<id_l2_") and 
                            l3_token.startswith("<id_l3_")):
                            try:
                                l1_id = int(l1_token.split('_')[2].rstrip('>'))
                                l2_id = int(l2_token.split('_')[2].rstrip('>'))
                                l3_id = int(l3_token.split('_')[2].rstrip('>'))
                                semantic_ids.append((l1_id, l2_id, l3_id))
                                i += 3
                                continue
                            except (ValueError, IndexError):
                                pass
                i += 1
            
            if semantic_ids:
                recommendations.append(semantic_ids)
        
        return recommendations
    
    def save_pretrained(self, save_directory: str):
        """Save model and tokenizer"""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(save_directory)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_directory)
        
        # Save config
        config_dict = {
            'base_model': self.base_model_path,
            'layer_vocab_sizes': self.layer_vocab_sizes
        }
        
        with open(os.path.join(save_directory, 'tiger_config.json'), 'w') as f:
            json.dump(config_dict, f)
    
    @classmethod
    def from_pretrained(cls, load_directory: str):
        """Load model and tokenizer"""
        # Load config
        config_path = os.path.join(load_directory, 'tiger_config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"TIGER config not found at {config_path}. "
                "Make sure the model was saved with TIGERModel.save_pretrained()"
            )
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
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
        tiger_model.base_model_path = config_dict['base_model']
        tiger_model.config = t5_model.config
        tiger_model.model = t5_model
        tiger_model.tokenizer = tokenizer
        tiger_model.layer_vocab_sizes = config_dict['layer_vocab_sizes']
        
        return tiger_model
