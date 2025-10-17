"""
Step R2 (Refactored): Defines the Hybrid Cross-Encoder model for ranking.

This model takes separate inputs for text tokens and a dense song vector,
projects the song vector, and fuses it with text embeddings for deep interaction.
"""
import torch
import torch.nn as nn
from transformers import T5EncoderModel, T5Config
import os
import json

class CrossEncoder(nn.Module):
    """A Hybrid Cross-Encoder that fuses text embeddings with a projected song vector."""
    def __init__(self, base_model: str, song_vector_dim: int):
        super().__init__()
        self.base_model_path = base_model
        self.config = T5Config.from_pretrained(base_model)
        self.t5_encoder = T5EncoderModel.from_pretrained(base_model)
        
        # Projection layer to map song vector to the same dimension as text embeddings
        self.song_projector = nn.Linear(song_vector_dim, self.config.d_model)

        # A simple classification head for relevance scoring
        self.classifier = nn.Linear(self.config.d_model, 1)

    def forward(self, input_ids, attention_mask, song_vector, labels=None):
        # 1. Get text token embeddings
        # Shape: (batch_size, text_seq_len, hidden_dim)
        text_embeds = self.t5_encoder.get_input_embeddings()(input_ids)
        
        # 2. Project song vector into the same embedding space
        # Shape: (batch_size, song_vector_dim) -> (batch_size, hidden_dim)
        song_embed = self.song_projector(song_vector)
        # Add a sequence dimension: (batch_size, 1, hidden_dim)
        song_embed = song_embed.unsqueeze(1)

        # 3. Concatenate text and song embeddings
        # Shape: (batch_size, text_seq_len + 1, hidden_dim)
        combined_embeds = torch.cat([text_embeds, song_embed], dim=1)

        # 4. Create a combined attention mask
        # The song embedding is always attended to
        song_attention_mask = torch.ones(song_embed.shape[0], 1, device=attention_mask.device, dtype=attention_mask.dtype)
        combined_attention_mask = torch.cat([attention_mask, song_attention_mask], dim=1)

        # 5. Pass the combined embeddings through the T5 encoder
        outputs = self.t5_encoder(
            inputs_embeds=combined_embeds, 
            attention_mask=combined_attention_mask
        ).last_hidden_state
        
        # 6. Pool the output and classify
        pooled_output = outputs.mean(dim=1)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.squeeze(), labels.squeeze())

        return {"loss": loss, "logits": logits}

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        self.t5_encoder.save_pretrained(os.path.join(save_directory, "t5_encoder"))
        torch.save(self.song_projector.state_dict(), os.path.join(save_directory, "song_projector.pt"))
        torch.save(self.classifier.state_dict(), os.path.join(save_directory, "classifier.pt"))
        with open(os.path.join(save_directory, 'ranker_config.json'), 'w') as f:
            json.dump({'base_model': self.base_model_path, 'song_vector_dim': self.song_projector.in_features}, f)

    @classmethod
    def from_pretrained(cls, load_directory: str):
        with open(os.path.join(load_directory, 'ranker_config.json'), 'r') as f:
            config = json.load(f)
        
        model = cls(base_model=config['base_model'], song_vector_dim=config['song_vector_dim'])
        model.t5_encoder = T5EncoderModel.from_pretrained(os.path.join(load_directory, "t5_encoder"))
        model.song_projector.load_state_dict(torch.load(os.path.join(load_directory, "song_projector.pt")))
        model.classifier.load_state_dict(torch.load(os.path.join(load_directory, "classifier.pt")))
        return model
