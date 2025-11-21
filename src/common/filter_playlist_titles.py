"""
This script prepares prompts for a Large Language Model (LLM) to filter
playlist titles based on their semantic meaning and descriptiveness.

It reads all playlist titles, batches them, and formats them into structured
prompts according to a template. The output is a JSONL file where each line
is a prompt for the LLM to process.
"""
import os
import sys
import pandas as pd
import json
import argparse
from tqdm import tqdm

# Add project root for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config
from src.common.utils import setup_logging

logger = setup_logging()

PROMPT_TEMPLATE = """你是一名专业的音乐策划和数据分析师。你的任务是分类歌单标题，判断它们是具有描述性、语义清晰的标题，还是笼统、私人化或噪音标题。

**分类标准：**

*   **"Good" (描述性标题)：** 保留这些。这类标题描述了特定的主题、风格、情绪、活动、季节、事件或艺术概念。
    *   示例："80年代摇滚金曲", "雨天伤感歌曲", "电子音乐", "健身动力", "秋日清冷", "后摇天花板".

*   **"Bad" (非描述性/噪音标题)：** 丢弃这些。这类标题是笼统的、私人化的、无意义的，或者仅仅是艺术家/歌曲列表。
    *   示例："我的歌单", "最爱", "我的喜爱", "张三的精选", "123", "啊啊啊", "周杰伦", "封面".

**输入：**
你将收到一个 JSON 格式的歌单列表，每个歌单包含一个 "id" 和一个 "title"。

**输出格式：**
你**必须只**返回一个有效的 JSON 格式的对象列表，其中每个对象包含 "id" 和你的 "classification" ('Good' 或 'Bad')。不要包含任何解释、介绍性文字或 Markdown 格式。

**输出示例：**
[
  {{
    "id": "12345",
    "classification": "Good"
  }},
  {{
    "id": "67890",
    "classification": "Bad"
  }}
]

**待分类歌单标题：**

{playlist_batch_json}
"""

def create_llm_prompts(
    playlist_info_file: str, 
    output_file: str, 
    batch_size: int = 20
):
    """
    Loads playlist data, batches it, and generates prompts for LLM-based filtering.

    Args:
        playlist_info_file: Path to the gen_playlist_info.csv file.
        output_file: Path to save the generated .jsonl file.
        batch_size: Number of playlists to include in each prompt.
    """
    logger.info(f"Loading playlists from {playlist_info_file}...")
    try:
        df = pd.read_csv(
            playlist_info_file, 
            sep='\t', 
            usecols=['glid', 'listname'],
            dtype={'glid': str, 'listname': str}
        ).dropna()
        logger.info(f"Loaded {len(df)} playlists.")
    except FileNotFoundError:
        logger.error(f"FATAL: Playlist info file not found at '{playlist_info_file}'")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading or parsing CSV file: {e}")
        sys.exit(1)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    num_batches = (len(df) + batch_size - 1) // batch_size
    logger.info(f"Total playlists: {len(df)}. Batch size: {batch_size}. Generating {num_batches} prompts.")

    with open(output_file, 'w', encoding='utf-8') as f:
        for i in tqdm(range(0, len(df), batch_size), desc="Generating Prompts"):
            batch = df.iloc[i:i+batch_size]
            
            # Format the batch of playlists as a JSON string
            batch_data = [
                {"id": row.glid, "title": row.listname}
                for row in batch.itertuples(index=False)
            ]
            playlist_batch_json = json.dumps(batch_data, indent=2, ensure_ascii=False)

            # Fill the template
            final_prompt = PROMPT_TEMPLATE.format(
                playlist_batch_json=playlist_batch_json
            )
            
            # Write the prompt as a single line in a JSON object
            prompt_object = {"prompt": final_prompt}
            f.write(json.dumps(prompt_object, ensure_ascii=False) + '\n')

    logger.info(f"Successfully generated {num_batches} prompts.")
    logger.info(f"Output file saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate prompts for LLM-based playlist title filtering."
    )
    
    # Use config for default paths
    config = Config()
    
    parser.add_argument(
        "--input_file",
        type=str,
        default=config.data.playlist_info_file,
        help=f"Path to the input playlist info file. Default: {config.data.playlist_info_file}"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=os.path.join(config.output_dir, "llm_filter_prompts.jsonl"),
        help="Path to save the generated JSONL file for prompts."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,
        help="Number of playlists to include in each prompt."
    )

    args = parser.parse_args()

    create_llm_prompts(
        playlist_info_file=args.input_file,
        output_file=args.output_file,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
