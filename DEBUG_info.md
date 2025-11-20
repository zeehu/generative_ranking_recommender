python src/semantic_id_generator/debug_collisions.py 
2025-11-20 11:55:32,141 - root - INFO - --- Starting Semantic ID Collision Analysis ---
2025-11-20 11:55:32,144 - root - INFO - Reading semantic IDs from outputs/semantic_id/song_semantic_ids.jsonl...
Building reverse map: 1095986it [00:03, 330274.33it/s]
2025-11-20 11:55:35,469 - root - INFO - Analyzing for collisions...

================================================================================
  Semantic ID Collision Report
================================================================================

Found 107556 unique semantic IDs that have collisions.
A total of 287066 songs are involved in these collisions.

----------------------------------------
Worst Collision Case:
  A single Semantic ID (112, 113, 131) was assigned to 45 different songs.
----------------------------------------

--- Example Collisions (showing first 5) ---

1. Semantic ID (99, 68, 67) is shared by 3 songs:
   ['88156227', '89975939', '32143665']

2. Semantic ID (87, 104, 114) is shared by 5 songs:
   ['32100650', '32042828', '32042821', '32144418', '32029511']

3. Semantic ID (21, 107, 136) is shared by 2 songs:
   ['89488966', '61700953']

4. Semantic ID (6, 93, 193) is shared by 2 songs:
   ['317568292', '353525988']

5. Semantic ID (99, 68, 109) is shared by 2 songs:
   ['32204446', '32044137']


python src/semantic_id_generator/analyze_semantic_ids.py 
总歌曲数: 1095986
唯一语义ID数: 916476
覆盖率: 83.62%

================================================================================
单层分布分析
================================================================================
第1层唯一ID数: 128
第2层唯一ID数: 128
第3层唯一ID数: 256

================================================================================
前两层组合分析
================================================================================
理论组合数: 128 × 128 = 16,384
实际组合数: 16383
组合利用率: 99.99%

================================================================================
第3层分布分析（按前两层组合）
================================================================================
平均每个[layer1,layer2]组合使用的layer3 ID数: 55.94
最少: 1
最多: 223
中位数: 56.00
标准差: 16.42

================================================================================
语义ID重复度分析
================================================================================
平均每个语义ID对应的歌曲数: 1.20

最常见的10个语义ID:
  (112, 113, 131): 45首歌曲
  (112, 2, 30): 45首歌曲
  (112, 29, 69): 43首歌曲
  (11, 0, 50): 41首歌曲
  (112, 54, 53): 40首歌曲
  (36, 25, 16): 38首歌曲
  (112, 2, 144): 37首歌曲
  (16, 86, 170): 36首歌曲
  (99, 62, 94): 35首歌曲
  (66, 0, 98): 34首歌曲

只出现一次的语义ID数: 808920
占比: 88.26%

================================================================================
[Layer1, Layer2]组合的样本数分布
================================================================================
平均样本数: 66.90
最少: 1
最多: 256
中位数: 66.00
标准差: 17.78

样本数最少的10个[layer1, layer2]组合:
  [112, 46]: 1首歌曲, 使用了1个layer3 ID
  [112, 76]: 1首歌曲, 使用了1个layer3 ID
  [71, 95]: 4首歌曲, 使用了4个layer3 ID
  [112, 80]: 7首歌曲, 使用了7个layer3 ID
  [112, 89]: 8首歌曲, 使用了3个layer3 ID
  [112, 87]: 10首歌曲, 使用了3个layer3 ID
  [103, 44]: 10首歌曲, 使用了9个layer3 ID
  [17, 96]: 10首歌曲, 使用了10个layer3 ID
  [37, 17]: 10首歌曲, 使用了10个layer3 ID
  [101, 16]: 12首歌曲, 使用了12个layer3 ID



python src/generator/prepare_corpus.py 
2025-11-20 11:59:10,402 - __main__ - INFO - --- Starting Step G2: Generator Corpus Generation ---
2025-11-20 11:59:10,402 - __main__ - INFO - Loading semantic IDs from outputs/semantic_id/song_semantic_ids.jsonl...
2025-11-20 11:59:13,244 - __main__ - INFO - Loaded 1095986 song-to-semantic-ID mappings from 1095986 lines.
2025-11-20 11:59:13,244 - __main__ - INFO - Loading playlist info from data/gen_playlist_info.csv...
2025-11-20 11:59:27,286 - __main__ - INFO - Loading playlist songs from data/gen_playlist_song.csv.sort...
2025-11-20 12:02:36,651 - __main__ - INFO - Loaded 4081755 playlists
2025-11-20 12:02:36,654 - __main__ - INFO - Building text-to-text corpus...
Processing playlists: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4081755/4081755 [04:10<00:00, 16288.01it/s]
2025-11-20 12:06:47,256 - __main__ - INFO - Successfully built corpus with 4019109 entries.
2025-11-20 12:06:47,258 - __main__ - INFO - Corpus building statistics:
2025-11-20 12:06:47,258 - __main__ - INFO -   Total playlists: 4081755
2025-11-20 12:06:47,258 - __main__ - INFO -   Valid corpus entries: 4019109
2025-11-20 12:06:47,258 - __main__ - INFO -   Playlists without info: 1
2025-11-20 12:06:47,258 - __main__ - INFO -   Playlists without title: 0
2025-11-20 12:06:47,258 - __main__ - INFO -   Playlists with too few songs: 62645
2025-11-20 12:06:47,258 - __main__ - INFO -   Total songs processed: 344384573
2025-11-20 12:06:47,258 - __main__ - INFO -   Songs with semantic IDs: 331070466 (96.13%)
2025-11-20 12:06:47,258 - __main__ - INFO -   Songs without semantic IDs: 13314107 (3.87%)
2025-11-20 12:06:47,382 - __main__ - INFO - --- Target Sequence Length Analysis (Before Truncation) ---
2025-11-20 12:06:47,382 - __main__ - INFO -   Total valid playlists: 4019109
2025-11-20 12:06:47,383 - __main__ - INFO -   Min length: 30
2025-11-20 12:06:47,384 - __main__ - INFO -   Max length: 1497
2025-11-20 12:06:47,386 - __main__ - INFO -   Avg length: 246.90
2025-11-20 12:06:47,422 - __main__ - INFO -   Median length (50th percentile): 162.0
2025-11-20 12:06:47,454 - __main__ - INFO -   90th percentile: 543.00
2025-11-20 12:06:47,483 - __main__ - INFO -   95th percentile: 783.00
2025-11-20 12:06:47,511 - __main__ - INFO -   99th percentile: 1233.00
2025-11-20 12:06:47,511 - __main__ - INFO - --- Truncation Analysis ---
2025-11-20 12:06:47,511 - __main__ - INFO -   Max allowed length (max_target_length - 1): 383
2025-11-20 12:06:47,511 - __main__ - INFO -   Playlists truncated: 684245 (17.02%)
2025-11-20 12:06:47,535 - __main__ - INFO - Splitting data and saving to files...
2025-11-20 12:06:49,357 - __main__ - INFO - Data split: 3938726 train, 40191 validation, 40192 test.
2025-11-20 12:06:49,364 - __main__ - INFO - Saving 3938726 records to outputs/generator/train.tsv...
2025-11-20 12:07:26,544 - __main__ - INFO - Saving 40191 records to outputs/generator/val.tsv...
2025-11-20 12:07:26,904 - __main__ - INFO - Saving 40192 records to outputs/generator/test.tsv...
2025-11-20 12:07:27,270 - __main__ - INFO - --- Step G2 Completed Successfully ---