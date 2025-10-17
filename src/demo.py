
"""
Step F1: Final Inference Demo for the Generate-and-Rank System.

This script chains the Generator and Ranker models to produce a final,
ranked list of recommendations from a user-provided text prompt.
"""
import os
import sys
import logging
import torch
import random
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config
from src.common.utils import setup_logging
from src.generator.tiger_model import TIGERModel
from src.ranker.ranker_model import CrossEncoder

logger = logging.getLogger(__name__)

class Demo:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        logger.info("Loading all models and mappings...")
        self.generator = self._load_generator()
        self.ranker = self._load_ranker()
        self.song_to_sem_id_map, self.sem_id_to_songs_map = self._create_semantic_mappings()
        self.song_to_vector_map = self._load_song_vectors()
        self.song_info_map = self._load_song_info()
        logger.info("All models and data loaded successfully.")

    def _load_generator(self) -> TIGERModel:
        path = os.path.join(self.config.model_dir, "generator", "final_model")
        logger.info(f"Loading Generator from {path}...")
        return TIGERModel.from_pretrained(path).to(self.device).eval()

    def _load_ranker(self) -> CrossEncoder:
        path = os.path.join(self.config.model_dir, "ranker", "final_model")
        logger.info(f"Loading Ranker from {path}...")
        # The ranker needs to know the full tokenizer length (with custom tokens)
        tokenizer_len = len(self.generator.tokenizer)
        return CrossEncoder.from_pretrained(path, tokenizer_len=tokenizer_len).to(self.device).eval()

    def predict(self, text: str, num_candidates: int = 100, top_k: int = 10) -> List[str]:
        # === Step 1: Candidate Generation ===
        logger.info(f"Generating {num_candidates} candidates with T5 Generator...")
        input_ids = self.generator.tokenizer.base_tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            generated_ids = self.generator.model.generate(
                input_ids,
                max_new_tokens=self.config.generator_t5.max_target_length,
                num_beams=num_candidates,
                num_return_sequences=num_candidates,
                early_stopping=True
            )
        
        decoded_preds = self.generator.tokenizer.base_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        candidate_song_ids = set()
        for pred_str in decoded_preds:
            sem_id_tuples = self._get_semantic_tuples(pred_str)
            for sem_id in sem_id_tuples:
                candidate_song_ids.update(self.sem_id_to_songs_map.get(sem_id, []))
        
        candidate_song_ids = list(candidate_song_ids)
        logger.info(f"Generated {len(candidate_song_ids)} unique candidate songs.")
        if not candidate_song_ids: return []

        # === Step 2: Precise Ranking ===
        logger.info(f"Ranking {len(candidate_song_ids)} candidates with Cross-Encoder...")
        scores = []
        ranker_tokenizer = self.generator.tokenizer.base_tokenizer
        
        for song_id in candidate_song_ids:
            song_vector = self.song_to_vector_map.get(str(song_id))
            if song_vector is None: continue

            encoding = ranker_tokenizer(text, max_length=self.config.generator_t5.max_input_length, padding="max_length", truncation=True, return_tensors="pt")
            song_vector_tensor = torch.tensor(song_vector, dtype=torch.float).unsqueeze(0) # Add batch dimension

            with torch.no_grad():
                logits = self.ranker(
                    input_ids=encoding.input_ids.to(self.device),
                    attention_mask=encoding.attention_mask.to(self.device),
                    song_vector=song_vector_tensor.to(self.device)
                )['logits']
                score = logits.squeeze().item()
            scores.append((song_id, score))

        # === Step 3: Sort and Return Top K ===
        scores.sort(key=lambda x: x[1], reverse=True)
        return [song_id for song_id, score in scores[:top_k]]

    def interactive_demo(self):
        print("\n" + "="*50)
        print("  🎶 生成式排序推荐系统已就绪 🎶")
        print("="*50)
        print("  输入任意文本（如歌单标题、心情、场景），然后按 Enter。")
        print("  输入 'exit' 或 'quit' 即可退出。")
        print("-"*50)

        while True:
            try:
                prompt = input("\n请输入文本 > ")
                if prompt.lower() in ['exit', 'quit']: break
                if not prompt: continue

                print("\n生成中... (此过程包含生成和排序两阶段，可能需要一些时间)")
                final_recommendations = self.predict(prompt)

                if not final_recommendations:
                    print("未能生成有效的歌曲列表，请尝试更换输入文本。")
                    continue
                
                print("\n✨ 为您推荐的最终歌曲列表 (已排序): ✨")
                for i, song_id in enumerate(final_recommendations, 1):
                    info = self.song_info_map.get(str(song_id), {"name": "N/A", "singer": "N/A"})
                    print(f"  {i}. {info['name']} - {info['singer']} (ID: {song_id})")

            except KeyboardInterrupt:
                break
        print("\n感谢使用，再见！")

    # Helper methods
    def _get_semantic_tuples(self, semantic_str: str) -> List[Tuple[int, ...]]:
        numerical_ids = [int(token[4:-1]) for token in semantic_str.split() if token.startswith("<id_")]
        chunk_size = self.config.rqkmeans.levels
        return list(dict.fromkeys([tuple(numerical_ids[i:i+chunk_size]) for i in range(0, len(numerical_ids) - chunk_size + 1, chunk_size)]))
    
    def _create_mappings(self) -> Tuple[Dict, Dict]:
        song_to_sem_id, sem_id_to_songs = {}, defaultdict(list)
        try:
            with open(self.config.data.semantic_ids_file, 'r') as f:
                for line in f: 
                    item = json.loads(line)
                    song_id, sem_id = item['song_id'], tuple(item['semantic_ids'])
                    song_to_sem_id[song_id] = sem_id
                    sem_id_to_songs[sem_id].append(song_id)
        except FileNotFoundError: sys.exit(f"FATAL: {self.config.data.semantic_ids_file} not found.")
        return dict(song_to_sem_id), dict(sem_id_to_songs)

    def _load_song_vectors(self) -> dict:
        logger.info(f"Loading song vectors from {self.config.data.song_vectors_file}...")
        df = pd.read_csv(self.config.data.song_vectors_file, dtype={'mixsongid': str}).set_index('mixsongid')
        return {idx: row.to_numpy(dtype=np.float32) for idx, row in df.iterrows()}

    def _load_song_info(self) -> dict: 
        import csv
        mapping = {}
        try:
            with open(self.config.data.song_info_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if len(row) >= 3: mapping[row[0]] = {"name": row[1], "singer": row[2]}
        except FileNotFoundError: logger.warning("Song info file not found.")
        return mapping

if __name__ == "__main__":
    config = Config()
    setup_logging(log_file=os.path.join(config.log_dir, "f1_demo.log"))
    logger = logging.getLogger(__name__)
    demo = Demo(config)
    demo.interactive_demo()
