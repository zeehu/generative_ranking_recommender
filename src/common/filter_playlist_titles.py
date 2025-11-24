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

*   **"Good" (描述性标题)：** 保留这些。这类标题应具有描述性，能让人联想到音乐内容。标准如下：
    *   描述了特定的主题、风格、情绪、比喻、活动、季节、事件、节目、歌手、艺术家、人物、影视剧或艺术概念。
    *   富有创意、比喻或氛围感，即使它们看起来有些主观 (例如：“古迹爱情风”)。
    *   包含常见音乐标签，即使括号没有闭合 (例如: `(Remix`, `(Live)` )。
    *   允许使用引号、括号等标点符号来强调或组织信息 (例如：“（emo歌曲）”)。
    *   包含热门的网络用语或迷因（meme），只要它们能唤起特定的文化或情感共鸣 (例如：“我是路过的滑稽”)。
    *   使用适度的对话式或个人化语气来吸引听众，只要它们仍然暗示了音乐的内容或情绪 (例如：“你喜欢的英文歌都在这里”, “经典和粤语，你们听听看。”)。
    *   英文、日文、韩文等常见外语标题，包括中外文混合的标题。不应将外语字符（如日语假名 `か`）误判为无意义符号。
    *   标题主体部分有明确含义，即使末尾包含少量无意义或无法识别的字符/拼写错误 (例如：“民谣歌曲inh”)。
    *   **示例：** "80年代摇滚金曲", "雨天伤感歌曲", "电子音乐", "健身动力", "秋日清冷", "踌躇满志", "戴上耳机，准备电疗。", "偶像练习生", "张杰用心歌唱，星星守护梦想", "⭐️ 《杀破狼》影视原声", "纯电3D超震撼 (Remix", "“（emo歌曲）”", "孤独か患者", "民谣歌曲inh".

*   **"Bad" (非描述性/噪音标题)：** 丢弃这些。这类标题信息量极低，无法让人对音乐内容产生任何联想。
    *   极其笼统且缺乏具体信息的私人化标题 (例如：“我的歌单”, “最爱”, “收藏夹”)。
    *   完全由无意义的符号、数字或乱码组成 (例如：“123”, “啊啊啊”, “。。。”, "?????")。
    *   与音乐完全无关的词语 (例如：“封面”, “新建列表”, "未命名")。
    *   **示例：** "我的歌单", "最爱", "我的喜爱", "张三的精选", "123", "啊啊啊", "。。。", "封面".

**输入：**
你将收到一个 JSON 格式的歌单列表，每个歌单包含一个 "id" 和一个 "title"。

**输出格式：**
你**必须只**返回一个有效的 JSON 格式的对象列表，其中每个对象包含 "id"，"title"和你的 "classification" ('Good' 或 'Bad')，"reason"（简要说明）。不要包含任何介绍性文字或 Markdown 格式。

**输出规则 (必须严格遵守)：**

你的整个输出**必须且只能**包含一个被 `<json_output>` 和 `</json_output>` 标签包裹的 JSON 数组。
*   **绝对禁止**在这两个标签之外添加任何介绍性文字、注释或结束语。
*   `<json_output>` 标签内的内容必须是一个**完整且语法有效**的 JSON 数组。
*   不要在最后一个 JSON 对象后面留下悬挂的逗号。

**输出示例：**
[
  {{
    "id": "12345",
    "title": "国风民谣",
    "classification": "Good",
    "reason": "特点风格"
  }},
  {{
    "id": "67890",
    "title": "我的收藏歌曲",
    "classification": "Bad",
    "reason": "私人化的标题"
  }}
]

**待分类歌单：**

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