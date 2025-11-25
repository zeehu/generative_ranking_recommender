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
2025-11-24 14:36:19,509 - __main__ - INFO - --- 开始步骤 G2: 生成器语料库生成 ---
2025-11-24 14:36:19,509 - __main__ - INFO - 随机种子已设置为 42 以确保数据处理的可重复性
2025-11-24 14:36:19,509 - __main__ - INFO - 正在从 outputs/semantic_id/song_semantic_ids.jsonl 加载语义ID...
2025-11-24 14:36:22,427 - __main__ - INFO - 从 1095986 行中加载了 1095986 个歌曲到语义ID的映射。
2025-11-24 14:36:22,427 - __main__ - INFO - 正在从 data/gen_playlist_info.csv 加载歌单信息...
2025-11-24 14:36:36,375 - __main__ - INFO - 正在从 data/gen_playlist_song.csv.sort 加载歌单歌曲...
2025-11-24 14:40:32,704 - __main__ - INFO - 已加载 4081755 个歌单
2025-11-24 14:40:32,705 - __main__ - INFO - 正在从 data/llm_filter_prompts.parse_res 加载歌单过滤数据...
2025-11-24 14:40:32,719 - __main__ - WARNING - 第 1 行: 歌单 id 的标签 'classification' 未知
2025-11-24 14:40:37,501 - __main__ - WARNING - 第 1491291 行: 歌单 id 的标签 'classification' 未知
2025-11-24 14:40:41,938 - __main__ - WARNING - 第 2982194 行: 歌单 id 的标签 'classification' 未知
2025-11-24 14:40:44,369 - __main__ - INFO - 从 4473358 行中加载了 4471355 个歌单的过滤数据。
2025-11-24 14:40:44,369 - __main__ - WARNING - 加载过滤数据时遇到 3 个错误
2025-11-24 14:40:44,738 - __main__ - INFO - 过滤统计: Good=3683006, Bad=788346, N/A=3
2025-11-24 14:40:44,738 - __main__ - INFO - 正在使用分块策略构建文本到文本语料库...
处理歌单中: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4081755/4081755 [07:15<00:00, 9382.87it/s]
2025-11-24 14:47:59,763 - __main__ - INFO - 成功构建了包含 4062289 条记录的语料库 (分块后)。
2025-11-24 14:48:00,339 - __main__ - INFO - 语料库构建统计:
2025-11-24 14:48:00,339 - __main__ - INFO -   原始歌单总数: 4081755
2025-11-24 14:48:00,339 - __main__ - INFO -   有效原始歌单数: 3325196
2025-11-24 14:48:00,339 - __main__ - INFO -   生成的训练样本总数 (分块后): 4062289
2025-11-24 14:48:00,339 - __main__ - INFO -   过滤的歌单 (Bad): 704163
2025-11-24 14:48:00,339 - __main__ - INFO -   过滤的歌单 (N/A): 3
2025-11-24 14:48:00,339 - __main__ - INFO -   无信息的歌单: 1
2025-11-24 14:48:00,339 - __main__ - INFO -   无标题的歌单: 21
2025-11-24 14:48:00,339 - __main__ - INFO -   歌曲数过少的歌单: 52371
2025-11-24 14:48:00,339 - __main__ - INFO -   处理的歌曲总数: 278995438
2025-11-24 14:48:00,340 - __main__ - INFO -   有语义ID的歌曲: 268158934 (96.12%)
2025-11-24 14:48:00,342 - __main__ - INFO -   无语义ID的歌曲: 10836504 (3.88%)
2025-11-24 14:48:00,534 - __main__ - INFO - --- 原始目标序列长度分析 (分块前) ---
2025-11-24 14:48:00,534 - __main__ - INFO -   有效歌单总数: 3325196
2025-11-24 14:48:00,535 - __main__ - INFO -   最小长度: 30
2025-11-24 14:48:00,536 - __main__ - INFO -   最大长度: 1497
2025-11-24 14:48:00,538 - __main__ - INFO -   平均长度: 241.71
2025-11-24 14:48:00,658 - __main__ - INFO -   中位数长度 (第50百分位): 159.0
2025-11-24 14:48:00,716 - __main__ - INFO -   第90百分位: 519.00
2025-11-24 14:48:00,742 - __main__ - INFO -   第95百分位: 750.00
2025-11-24 14:48:00,768 - __main__ - INFO -   第99百分位: 1215.00
2025-11-24 14:48:00,769 - __main__ - INFO - --- 分块影响分析 ---
2025-11-24 14:48:00,769 - __main__ - INFO -   每个分块允许的最大长度: 383
2025-11-24 14:48:00,769 - __main__ - INFO -   需要分块的原始歌单数: 532648 (16.02%)
2025-11-24 14:48:00,791 - __main__ - INFO - 正在拆分数据并保存到文件...
2025-11-24 14:48:02,746 - __main__ - INFO - 数据拆分: 3981043 训练集, 40622 验证集, 40624 测试集。
2025-11-24 14:48:03,349 - __main__ - INFO - 正在保存 3981043 条记录到 outputs/generator/train.tsv...
2025-11-24 14:50:14,193 - __main__ - INFO - 正在保存 40622 条记录到 outputs/generator/val.tsv...
2025-11-24 14:50:28,503 - __main__ - INFO - 正在保存 40624 条记录到 outputs/generator/test.tsv...
2025-11-24 14:50:54,971 - __main__ - INFO - --- 步骤 G2 成功完成 ---



python src/generator/preprocess_tokenize.py 
2025-11-24 14:56:39,403 - __main__ - INFO - ================================================================================
2025-11-24 14:56:39,404 - __main__ - INFO - 🚀 开始预处理tokenization（内存优化版本）
2025-11-24 14:56:39,404 - __main__ - INFO - ================================================================================
2025-11-24 14:56:39,404 - __main__ - INFO - 
📊 Tokenizer配置:
2025-11-24 14:56:39,404 - __main__ - INFO -   模型: /home/search/base-model/mengzi-t5-base
2025-11-24 14:56:39,404 - __main__ - INFO -   Layer 1 词表大小: 128
2025-11-24 14:56:39,404 - __main__ - INFO -   Layer 2 词表大小: 128
2025-11-24 14:56:39,404 - __main__ - INFO -   Layer 3 词表大小: 256
2025-11-24 14:56:39,404 - __main__ - INFO -   总语义ID tokens: 512
2025-11-24 14:56:39,406 - __main__ - INFO - 
================================================================================
2025-11-24 14:56:39,406 - __main__ - INFO - 处理训练集
2025-11-24 14:56:39,406 - __main__ - INFO - ================================================================================
2025-11-24 14:56:39,406 - __main__ - INFO - ================================================================================
2025-11-24 14:56:39,406 - __main__ - INFO - 🚀 多进程并行Tokenization（内存优化版本）
2025-11-24 14:56:39,406 - __main__ - INFO - ================================================================================
2025-11-24 14:56:39,406 - __main__ - INFO - 文件路径: outputs/generator/train.tsv
2025-11-24 14:56:39,406 - __main__ - INFO - Chunk大小: 10,000 样本/chunk (降低以减少内存)
2025-11-24 14:56:39,406 - __main__ - INFO - 并行进程数: 16 (CPU核心数: 256)
2025-11-24 14:56:39,406 - __main__ - INFO - 内存优化: 流式合并 + 及时清理
2025-11-24 14:56:39,409 - __main__ - INFO - 
📖 步骤1: 读取并分割数据...
2025-11-24 14:56:39,410 - __main__ - INFO - 读取文件: outputs/generator/train.tsv
2025-11-24 14:56:39,410 - __main__ - INFO - Chunk大小: 10,000 样本/chunk
2025-11-24 14:57:12,333 - __main__ - INFO - 总样本数: 3,981,043
2025-11-24 14:57:12,336 - __main__ - INFO - 清理特殊符号的样本数: 0 (0.00%)
2025-11-24 14:57:12,336 - __main__ - INFO - 分割成 399 个chunks
2025-11-24 14:57:12,336 - __main__ - INFO - ✅ 读取完成，耗时: 32.9秒
2025-11-24 14:57:12,336 - __main__ - INFO - 
⚡ 步骤2: 启动 16 个进程进行并行tokenization..
====================================================================================================
📋 采样检查 - Chunk 0 的前5条数据
====================================================================================================

────────────────────────────────────────────────────────────────────────────────────────────────────
样本 #1
────────────────────────────────────────────────────────────────────────────────────────────────────

【原始输入】
  文本: 经典好歌，通勤最佳
  长度: 9 字符

【原始输出】
  文本: <id_l2_107> <id_l3_208> <id_l1_87> <id_l2_34> <id_l3_143> <id_l1_87> <id_l2_107> <id_l3_171> <id_l1_127> <id_l2_84> <id_l3_125> <id_l1_87> <id_l2_61> <id_l3_80> <id_l1_87> <id_l2_107> <id_l3_212> <id_...
  字符长度: 4347 字符
  语义ID数量: 383 个

【Tokenize后的输入】
  input_ids: [2521, 67, 1129, 3, 24754, 2379, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]...
  input_ids长度: 128
  有效token数: 7
  padding数: 121

【Tokenize后的输出】
  label_ids: [32363, 32592, 32215, 32290, 32527, 32215, 32363, 32555, 32255, 32340, 32509, 32215, 32317, 32464, 32215, 32363, 32596, 32215, 32268, 32586, 32215, 32363, 32407, 32207, 32304, 32539, 32215, 32378, 32502, 32227, 32356, 32395, 32227, 32302, 32400, 32215, 32305, 32581, 32215, 32332, 32538, 32215, 32268, 32551, 32227, 32313, 32444, 32157, 32274, 32491]...
  label_ids长度: 384
  有效token数: 384
  padding数: 0

【解码验证（前50个token）】
  输入解码: 经典好歌,通勤最佳</s>
  输出解码: <id_l2_107> <id_l3_208> <id_l1_87> <id_l2_34> <id_l3_143> <id_l1_87> <id_l2_107> <id_l3_171> <id_l1_127> <id_l2_84> <id_l3_125> <id_l1_87> <id_l2_61> <id_l3_80> <id_l1_87> <id_l2_107> <id_l3_212> <id_l1_87> <id_l2_12> <id_l3_202> <id_l1_87> <id_l2_107> <id_l3_23> <id_l1_79> <id_l2_48> <id_l3_155> <id_l1_87> <id_l2_122> <id_l3_118> <id_l1_99> <id_l2_100> <id_l3_11> <id_l1_99> <id_l2_46> <id_l3_16> <id_l1_87> <id_l2_49> <id_l3_197> <id_l1_87> <id_l2_76> <id_l3_154> <id_l1_87> <id_l2_12> <id_l3_167> <id_l1_99> <id_l2_57> <id_l3_60> <id_l1_29> <id_l2_18> <id_l3_107>

────────────────────────────────────────────────────────────────────────────────────────────────────
样本 #2
────────────────────────────────────────────────────────────────────────────────────────────────────

【原始输入】
  文本: 怀旧经典老歌曲500首免费：8090后经典老歌
  长度: 23 字符

【原始输出】
  文本: <id_l1_41> <id_l2_68> <id_l3_243> <id_l1_13> <id_l2_53> <id_l3_203> <id_l1_65> <id_l2_105> <id_l3_34> <id_l1_65> <id_l2_50> <id_l3_200> <id_l1_41> <id_l2_28> <id_l3_217> <id_l1_65> <id_l2_108> <id_l3_...
  字符长度: 1655 字符
  语义ID数量: 147 个

【Tokenize后的输入】
  input_ids: [23089, 2521, 170, 4619, 2056, 850, 1375, 13, 1704, 2015, 37, 2521, 170, 1129, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]...
  input_ids长度: 128
  有效token数: 15
  padding数: 113

【Tokenize后的输出】
  label_ids: [32169, 32324, 32627, 32141, 32309, 32587, 32193, 32361, 32418, 32193, 32306, 32584, 32169, 32284, 32601, 32193, 32364, 32528, 32141, 32312, 32502, 32169, 32324, 32638, 32142, 32307, 32636, 32193, 32337, 32405, 32193, 32273, 32453, 32193, 32282, 32535, 32193, 32273, 32435, 32193, 32324, 32464, 32141, 32309, 32555, 32193, 32282, 32402, 32141, 32312]...
  label_ids长度: 384
  有效token数: 149
  padding数: 235

【解码验证（前50个token）】
  输入解码: 怀旧经典老歌曲500首免费:8090后经典老歌</s>
  输出解码: <id_l1_41> <id_l2_68> <id_l3_243> <id_l1_13> <id_l2_53> <id_l3_203> <id_l1_65> <id_l2_105> <id_l3_34> <id_l1_65> <id_l2_50> <id_l3_200> <id_l1_41> <id_l2_28> <id_l3_217> <id_l1_65> <id_l2_108> <id_l3_144> <id_l1_13> <id_l2_56> <id_l3_118> <id_l1_41> <id_l2_68> <id_l3_254> <id_l1_14> <id_l2_51> <id_l3_252> <id_l1_65> <id_l2_81> <id_l3_21> <id_l1_65> <id_l2_17> <id_l3_69> <id_l1_65> <id_l2_26> <id_l3_151> <id_l1_65> <id_l2_17> <id_l3_51> <id_l1_65> <id_l2_68> <id_l3_80> <id_l1_13> <id_l2_53> <id_l3_171> <id_l1_65> <id_l2_26> <id_l3_18> <id_l1_13> <id_l2_56>

────────────────────────────────────────────────────────────────────────────────────────────────────
样本 #3
────────────────────────────────────────────────────────────────────────────────────────────────────

【原始输入】
  文本: 我们不能哭
  长度: 5 字符

【原始输出】
  文本: <id_l1_26> <id_l2_52> <id_l3_111> <id_l1_99> <id_l2_90> <id_l3_109> <id_l1_127> <id_l2_4> <id_l3_164> <id_l1_6> <id_l2_92> <id_l3_204> <id_l1_99> <id_l2_64> <id_l3_111> <id_l1_65> <id_l2_105> <id_l3_1...
  字符长度: 1884 字符
  语义ID数量: 168 个

【Tokenize后的输入】
  input_ids: [15969, 2083, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]...
  input_ids长度: 128
  有效token数: 3
  padding数: 125

【Tokenize后的输出】
  label_ids: [32154, 32308, 32495, 32227, 32346, 32493, 32255, 32260, 32548, 32134, 32348, 32588, 32227, 32320, 32495, 32193, 32361, 32540, 32141, 32309, 32581, 32227, 32346, 32617, 32133, 32279, 32471, 32227, 32340, 32510, 32235, 32271, 32472, 32227, 32319, 32405, 32227, 32324, 32518, 32133, 32335, 32589, 32141, 32296, 32544, 32243, 32359, 32550, 32227, 32369]...
  label_ids长度: 384
  有效token数: 170
  padding数: 214

【解码验证（前50个token）】
  输入解码: 我们不能哭</s>
  输出解码: <id_l1_26> <id_l2_52> <id_l3_111> <id_l1_99> <id_l2_90> <id_l3_109> <id_l1_127> <id_l2_4> <id_l3_164> <id_l1_6> <id_l2_92> <id_l3_204> <id_l1_99> <id_l2_64> <id_l3_111> <id_l1_65> <id_l2_105> <id_l3_156> <id_l1_13> <id_l2_53> <id_l3_197> <id_l1_99> <id_l2_90> <id_l3_233> <id_l1_5> <id_l2_23> <id_l3_87> <id_l1_99> <id_l2_84> <id_l3_126> <id_l1_107> <id_l2_15> <id_l3_88> <id_l1_99> <id_l2_63> <id_l3_21> <id_l1_99> <id_l2_68> <id_l3_134> <id_l1_5> <id_l2_79> <id_l3_205> <id_l1_13> <id_l2_40> <id_l3_160> <id_l1_115> <id_l2_103> <id_l3_166> <id_l1_99> <id_l2_113>

────────────────────────────────────────────────────────────────────────────────────────────────────
样本 #4
────────────────────────────────────────────────────────────────────────────────────────────────────

【原始输入】
  文本: 浓缩是精华
  长度: 5 字符

【原始输出】
  文本: <id_l1_99> <id_l2_40> <id_l3_198> <id_l1_99> <id_l2_56> <id_l3_237> <id_l1_29> <id_l2_69> <id_l3_27> <id_l1_2> <id_l2_89> <id_l3_112> <id_l1_2> <id_l2_89> <id_l3_255> <id_l1_29> <id_l2_35> <id_l3_27> ...
  字符长度: 1043 字符
  语义ID数量: 93 个

【Tokenize后的输入】
  input_ids: [15736, 11, 5032, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]...
  input_ids长度: 128
  有效token数: 4
  padding数: 124

【Tokenize后的输出】
  label_ids: [32227, 32296, 32582, 32227, 32312, 32621, 32157, 32325, 32411, 32130, 32345, 32496, 32130, 32345, 32639, 32157, 32291, 32411, 32134, 32282, 32481, 32149, 32335, 32536, 32134, 32282, 32630, 32130, 32345, 32579, 32130, 32337, 32573, 32157, 32291, 32411, 32149, 32363, 32529, 32134, 32282, 32613, 32227, 32296, 32461, 32157, 32305, 32404, 32134, 32348]...
  label_ids长度: 384
  有效token数: 95
  padding数: 289

【解码验证（前50个token）】
  输入解码: 浓缩是精华</s>
  输出解码: <id_l1_99> <id_l2_40> <id_l3_198> <id_l1_99> <id_l2_56> <id_l3_237> <id_l1_29> <id_l2_69> <id_l3_27> <id_l1_2> <id_l2_89> <id_l3_112> <id_l1_2> <id_l2_89> <id_l3_255> <id_l1_29> <id_l2_35> <id_l3_27> <id_l1_6> <id_l2_26> <id_l3_97> <id_l1_21> <id_l2_79> <id_l3_152> <id_l1_6> <id_l2_26> <id_l3_246> <id_l1_2> <id_l2_89> <id_l3_195> <id_l1_2> <id_l2_81> <id_l3_189> <id_l1_29> <id_l2_35> <id_l3_27> <id_l1_21> <id_l2_107> <id_l3_145> <id_l1_6> <id_l2_26> <id_l3_229> <id_l1_99> <id_l2_40> <id_l3_77> <id_l1_29> <id_l2_49> <id_l3_20> <id_l1_6> <id_l2_92>

────────────────────────────────────────────────────────────────────────────────────────────────────
样本 #5
────────────────────────────────────────────────────────────────────────────────────────────────────

【原始输入】
  文本: 能说会唱！伤感学会了自我介绍。
  长度: 15 字符

【原始输出】
  文本: <id_l1_64> <id_l2_18> <id_l3_129> <id_l1_64> <id_l2_38> <id_l3_167> <id_l1_99> <id_l2_104> <id_l3_49> <id_l1_99> <id_l2_84> <id_l3_147> <id_l1_64> <id_l2_124> <id_l3_57> <id_l1_64> <id_l2_93> <id_l3_1...
  字符长度: 544 字符
  语义ID数量: 48 个

【Tokenize后的输入】
  input_ids: [76, 58, 56, 1328, 30, 15983, 9367, 20575, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]...
  input_ids长度: 128
  有效token数: 10
  padding数: 118

【Tokenize后的输出】
  label_ids: [32192, 32274, 32513, 32192, 32294, 32551, 32227, 32360, 32433, 32227, 32340, 32531, 32192, 32380, 32441, 32192, 32349, 32533, 32192, 32296, 32466, 32227, 32324, 32418, 32192, 32276, 32455, 32192, 32326, 32452, 32227, 32262, 32623, 32192, 32371, 32607, 32227, 32331, 32554, 32192, 32297, 32440, 32192, 32276, 32561, 32227, 32331, 32509, 32641, 1]...
  label_ids长度: 384
  有效token数: 50
  padding数: 334

【解码验证（前50个token）】
  输入解码: 能说会唱!伤感学会了自我介绍。</s>
  输出解码: <id_l1_64> <id_l2_18> <id_l3_129> <id_l1_64> <id_l2_38> <id_l3_167> <id_l1_99> <id_l2_104> <id_l3_49> <id_l1_99> <id_l2_84> <id_l3_147> <id_l1_64> <id_l2_124> <id_l3_57> <id_l1_64> <id_l2_93> <id_l3_149> <id_l1_64> <id_l2_40> <id_l3_82> <id_l1_99> <id_l2_68> <id_l3_34> <id_l1_64> <id_l2_20> <id_l3_71> <id_l1_64> <id_l2_70> <id_l3_68> <id_l1_99> <id_l2_6> <id_l3_239> <id_l1_64> <id_l2_115> <id_l3_223> <id_l1_99> <id_l2_75> <id_l3_170> <id_l1_64> <id_l2_41> <id_l3_56> <id_l1_64> <id_l2_20> <id_l3_177> <id_l1_99> <id_l2_75> <id_l3_125> <eos> </s>

====================================================================================================
✅ 采样检查完成
====================================================================================================


train_t5_optimized.py 训练日志如下：
W1124 15:07:50.954000 19714 torch/distributed/run.py:774] *****************************************
2025-11-24 15:09:37,343 - __main__ - INFO - Using optimized configuration (config_optimized.py)
2025-11-24 15:09:37,343 - __main__ - INFO - 
################################################################################
2025-11-24 15:09:37,343 - __main__ - INFO - #                                                                              #
2025-11-24 15:09:37,343 - __main__ - INFO - #                    T5 GENERATOR TRAINING (OPTIMIZED)                         #
2025-11-24 15:09:37,343 - __main__ - INFO - #                                                                              #
2025-11-24 15:09:37,343 - __main__ - INFO - ################################################################################
2025-11-24 15:09:37,343 - __main__ - INFO - 
2025-11-24 15:09:37,343 - __main__ - INFO - Hardware: 4×L20 GPU (48GB) + 20 CPU + 50GB RAM
2025-11-24 15:09:37,343 - __main__ - INFO - Dataset: ~98% training samples
2025-11-24 15:09:37,344 - __main__ - INFO - Epochs: 3
2025-11-24 15:09:37,344 - __main__ - INFO - Expected speedup: ~10% vs batch=96 vs original config
2025-11-24 15:09:37,344 - __main__ - INFO - 
2025-11-24 15:09:37,344 - __main__ - INFO - ################################################################################

2025-11-24 15:09:38,366 - __main__ - INFO - Detected 4 GPU(s):
2025-11-24 15:09:38,370 - __main__ - INFO -   GPU 0: NVIDIA L20 (47.8 GB)
2025-11-24 15:09:38,370 - __main__ - INFO -   GPU 1: NVIDIA L20 (47.8 GB)
2025-11-24 15:09:38,370 - __main__ - INFO -   GPU 2: NVIDIA L20 (47.8 GB)
2025-11-24 15:09:38,370 - __main__ - INFO -   GPU 3: NVIDIA L20 (47.8 GB)
2025-11-24 15:09:38,370 - __main__ - INFO - 
2025-11-24 15:09:38,447 - __main__ - INFO - ================================================================================
2025-11-24 15:09:38,447 - __main__ - INFO - Step G3: T5 Generator Model Training (OPTIMIZED)
2025-11-24 15:09:38,447 - __main__ - INFO - ================================================================================
2025-11-24 15:09:38,447 - __main__ - INFO - Layer vocab sizes: {'l1': 128, 'l2': 128, 'l3': 256}
2025-11-24 15:09:38,447 - __main__ - INFO - Total semantic ID tokens: 512 + 2 special tokens
2025-11-24 15:09:38,447 - __main__ - INFO - Initializing TIGER model...
2025-11-24 15:11:48,153 - __main__ - INFO - Tokenizer vocab size: 32644
2025-11-24 15:11:48,153 - __main__ - INFO - Loading memory-mapped dataset from outputs/generator/train_tokenized...
2025-11-24 15:11:48,337 - __main__ - INFO - Loaded 3,981,043 samples (memory-mapped)
2025-11-24 15:11:48,338 - __main__ - INFO - Dataset features: {'input_ids': List(Value('int64')), 'attention_mask': List(Value('int64')), 'labels': List(Value('int64'))}
2025-11-24 15:11:48,338 - __main__ - INFO - Loading memory-mapped dataset from outputs/generator/val_tokenized...
2025-11-24 15:11:48,356 - __main__ - INFO - Loaded 40,622 samples (memory-mapped)
2025-11-24 15:11:48,356 - __main__ - INFO - Dataset features: {'input_ids': List(Value('int64')), 'attention_mask': List(Value('int64')), 'labels': List(Value('int64'))}
2025-11-24 15:11:50,493 - accelerate.utils.other - WARNING - Detected kernel version 4.15.0, which is below the recommended minimum of 5.5.0; this can cause the process to hang. It is recommended to upgrade the kernel to the minimum version or higher.
2025-11-24 15:11:50,719 - __main__ - INFO - ================================================================================
2025-11-24 15:11:50,719 - __main__ - INFO - Enabling torch.compile for model optimization
2025-11-24 15:11:50,719 - __main__ - INFO - Backend: inductor
2025-11-24 15:11:50,719 - __main__ - INFO - Mode: reduce-overhead
2025-11-24 15:11:50,719 - __main__ - INFO - Note: First few iterations will be slower due to compilation
2025-11-24 15:11:50,719 - __main__ - INFO - Expected speedup: 5-15% after compilation
2025-11-24 15:11:50,719 - __main__ - INFO - ================================================================================
2025-11-24 15:12:32,022 - __main__ - INFO - torch.compile enabled successfully
2025-11-24 15:12:32,022 - __main__ - INFO - 
================================================================================
2025-11-24 15:12:32,022 - __main__ - INFO - TRAINING CONFIGURATION SUMMARY (4×L20 GPU OPTIMIZED)
2025-11-24 15:12:32,022 - __main__ - INFO - ================================================================================
2025-11-24 15:12:32,022 - __main__ - INFO - 
[Dataset Information]
2025-11-24 15:12:32,022 - __main__ - INFO -   Training samples:   3,981,043
2025-11-24 15:12:32,022 - __main__ - INFO -   Validation samples: 40,622
2025-11-24 15:12:32,022 - __main__ - INFO -   Train/Val ratio:    98.0:1
2025-11-24 15:12:32,022 - __main__ - INFO - 
[Hardware Configuration]
2025-11-24 15:12:32,022 - __main__ - INFO -   Number of GPUs:     4
2025-11-24 15:12:32,022 - __main__ - INFO -   GPU Model:          L20 (48GB GDDR6)
2025-11-24 15:12:32,022 - __main__ - INFO -   CPU cores:          20
2025-11-24 15:12:32,022 - __main__ - INFO -   System memory:      50GB
2025-11-24 15:12:32,022 - __main__ - INFO - 
[Batch Configuration]
2025-11-24 15:12:32,022 - __main__ - INFO -   Per-device batch:   128
2025-11-24 15:12:32,022 - __main__ - INFO -   Gradient accum:     4
2025-11-24 15:12:32,022 - __main__ - INFO -   Effective batch:    2048 = 128 × 4 × 4
2025-11-24 15:12:32,022 - __main__ - INFO - 
[Training Schedule]
2025-11-24 15:12:32,022 - __main__ - INFO -   Epochs:             3
2025-11-24 15:12:32,022 - __main__ - INFO -   Steps per epoch:    1,943
2025-11-24 15:12:32,022 - __main__ - INFO -   Total steps:        5,829
2025-11-24 15:12:32,022 - __main__ - INFO -   Warmup steps:       600 (10.3%)
2025-11-24 15:12:32,022 - __main__ - INFO -   Eval frequency:     every 500 steps
2025-11-24 15:12:32,022 - __main__ - INFO -   Save frequency:     every 500 steps
2025-11-24 15:12:32,022 - __main__ - INFO - 
[Optimization Settings]
2025-11-24 15:12:32,022 - __main__ - INFO -   Learning rate:      0.00042
2025-11-24 15:12:32,022 - __main__ - INFO -   LR scheduler:       cosine
2025-11-24 15:12:32,022 - __main__ - INFO -   Weight decay:       0.01
2025-11-24 15:12:32,022 - __main__ - INFO -   Max grad norm:      1.0
2025-11-24 15:12:32,022 - __main__ - INFO -   Mixed precision:    FP16 (O2 level)
2025-11-24 15:12:32,022 - __main__ - INFO -   Gradient ckpt:      True
2025-11-24 15:12:32,022 - __main__ - INFO - 
[DataLoader Settings]
2025-11-24 15:12:32,022 - __main__ - INFO -   Num workers:        16
2025-11-24 15:12:32,022 - __main__ - INFO -   Pin memory:         True
2025-11-24 15:12:32,022 - __main__ - INFO -   Prefetch factor:    4
2025-11-24 15:12:32,022 - __main__ - INFO -   Persistent workers: True
2025-11-24 15:12:32,022 - __main__ - INFO - 
[Performance Estimation]
2025-11-24 15:12:32,022 - __main__ - INFO -   Est. time per step: ~3.3s
2025-11-24 15:12:32,022 - __main__ - INFO -   Est. total time:    ~5.3 hours
2025-11-24 15:12:32,022 - __main__ - INFO -   Expected GPU util:  85-90%
2025-11-24 15:12:32,022 - __main__ - INFO -   Speedup vs orig:    ~10% vs batch=96
2025-11-24 15:12:32,022 - __main__ - INFO - 
================================================================================

2025-11-24 15:12:32,022 - __main__ - INFO - 
================================================================================
2025-11-24 15:12:32,022 - __main__ - INFO - STARTING TRAINING
2025-11-24 15:12:32,022 - __main__ - INFO - ================================================================================
2025-11-24 15:12:32,023 - __main__ - INFO - Monitor GPU usage: watch -n 1 nvidia-smi
2025-11-24 15:12:32,023 - __main__ - INFO - Monitor training log: tail -f logs/g3_train_t5_optimized.log
2025-11-24 15:12:32,023 - __main__ - INFO - ================================================================================

{'loss': 18.2196, 'grad_norm': inf, 'learning_rate': 0.0, 'epoch': 0.0}                                                                                                                                                                     
{'loss': 6.9392, 'grad_norm': 0.7139238119125366, 'learning_rate': 6.93e-05, 'epoch': 0.05}                                                                                                                                                 
{'loss': 3.0593, 'grad_norm': 0.1482037454843521, 'learning_rate': 0.0001393, 'epoch': 0.1}                                                                                                                                                 
{'loss': 3.0567, 'grad_norm': 0.16590699553489685, 'learning_rate': 0.00020930000000000002, 'epoch': 0.15}                                                                                                                                  
{'loss': 2.9673, 'grad_norm': 0.2092275768518448, 'learning_rate': 0.0002793, 'epoch': 0.21}                                                                                                                                                
{'loss': 2.7695, 'grad_norm': 0.3825385272502899, 'learning_rate': 0.00034930000000000003, 'epoch': 0.26}                                                                                                                                   
{'eval_loss': 2.3665096759796143, 'eval_runtime': 60.4082, 'eval_samples_per_second': 672.459, 'eval_steps_per_second': 1.059, 'epoch': 0.26}                                                                                               
  9%|████████████████▍                                                                                                                                                                              | 500/5832 [1:42:21<17:55:06, 12.10s/it]/mnt/jezeehu/gen_playlist/gen-venv/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py:4807: UserWarning: No device id is provided via `init_process_group` or `barrier `. Using the current device set by the user.          
  warnings.warn(  # warn only once
/mnt/jezeehu/gen_playlist/gen-venv/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py:4807: UserWarning: No device id is provided via `init_process_group` or `barrier `. Using the current device set by the user. 
  warnings.warn(  # warn only once
/mnt/jezeehu/gen_playlist/gen-venv/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py:4807: UserWarning: No device id is provided via `init_process_group` or `barrier `. Using the current device set by the user. 
  warnings.warn(  # warn only once
/mnt/jezeehu/gen_playlist/gen-venv/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py:4807: UserWarning: No device id is provided via `init_process_group` or `barrier `. Using the current device set by the user. 
  warnings.warn(  # warn only once
{'loss': 2.1458, 'grad_norm': 1.1142971515655518, 'learning_rate': 0.0004193, 'epoch': 0.31}                                                                                                                                                
{'loss': 1.9841, 'grad_norm': 0.5199762582778931, 'learning_rate': 0.0004196290664901623, 'epoch': 0.36}                                                                                                                                    
{'loss': 1.8604, 'grad_norm': 0.21664495766162872, 'learning_rate': 0.00041850258243401276, 'epoch': 0.41}                                                                                                                                  
{'loss': 1.7432, 'grad_norm': 0.22095535695552826, 'learning_rate': 0.0004166245702805468, 'epoch': 0.46}                                                                                                                                   
{'loss': 1.695, 'grad_norm': 0.09925086051225662, 'learning_rate': 0.0004140017991492828, 'epoch': 0.51}                                                                                                                                    
{'eval_loss': 1.6024506092071533, 'eval_runtime': 60.1479, 'eval_samples_per_second': 675.368, 'eval_steps_per_second': 1.064, 'epoch': 0.51}                                                                                               
 17%|████████████████████████████████▌                                                                                                                                                             | 1000/5832 [3:24:24<16:14:12, 12.10s/it]/mnt/jezeehu/gen_playlist/gen-venv/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py:4807: UserWarning: No device id is provided via `init_process_group` or `barrier `. Using the current device set by the user.          
  warnings.warn(  # warn only once
{'loss': 1.6123, 'grad_norm': 0.09582892060279846, 'learning_rate': 0.0004106437225739613, 'epoch': 0.57}                                                                                                                                   
{'loss': 1.5687, 'grad_norm': 0.10555332899093628, 'learning_rate': 0.0004065624444281642, 'epoch': 0.62}                                                                                                                                   
{'loss': 1.5379, 'grad_norm': 0.09707406908273697, 'learning_rate': 0.00040177267529803256, 'epoch': 0.67}                                                                                                                                  
{'loss': 1.5193, 'grad_norm': 0.13226012885570526, 'learning_rate': 0.00039629167945933174, 'epoch': 0.72}                                                                                                                                  
{'loss': 1.5019, 'grad_norm': 0.1011730283498764, 'learning_rate': 0.0003901392126499811, 'epoch': 0.77}                                                                                                                                    
{'eval_loss': 1.4454649686813354, 'eval_runtime': 60.1764, 'eval_samples_per_second': 675.049, 'eval_steps_per_second': 1.064, 'epoch': 0.77}                                                                                               
 26%|████████████████████████████████████████████████▊                                                                                                                                             | 1500/5832 [5:06:32<14:34:01, 12.11s/it]/mnt/jezeehu/gen_playlist/gen-venv/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py:4807: UserWarning: No device id is provided via `init_process_group` or `barrier `. Using the current device set by the user.          
  warnings.warn(  # warn only once
{'loss': 1.4872, 'grad_norm': 0.08239787071943283, 'learning_rate': 0.00038333745086234033, 'epoch': 0.82}                                                                                                                                  
{'loss': 1.5126, 'grad_norm': 7.773032188415527, 'learning_rate': 0.0003759109104119149, 'epoch': 0.87}                                                                                                                                     
{'loss': 1.5096, 'grad_norm': 0.07301688939332962, 'learning_rate': 0.00036788635957058527, 'epoch': 0.93}                                                                                                                                  
{'loss': 1.4661, 'grad_norm': 0.1110166683793068, 'learning_rate': 0.000359292722082869, 'epoch': 0.98}                                                                                                                                     
{'loss': 1.4629, 'grad_norm': 0.08957090973854065, 'learning_rate': 0.0003501609729129839, 'epoch': 1.03}                                                                                                                                   
{'eval_loss': 1.4099483489990234, 'eval_runtime': 60.2293, 'eval_samples_per_second': 674.456, 'eval_steps_per_second': 1.063, 'epoch': 1.03}                                                                                               
 34%|█████████████████████████████████████████████████████████████████▏                                                                                                                            | 2000/5832 [6:48:44<12:52:48, 12.10s/it]/mnt/jezeehu/gen_playlist/gen-venv/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py:4807: UserWarning: No device id is provided via `init_process_group` or `barrier `. Using the current device set by the user.          
  warnings.warn(  # warn only once
{'loss': 1.4524, 'grad_norm': 0.05971243977546692, 'learning_rate': 0.00034052402659847996, 'epoch': 1.08}                                                                                                                                  
{'loss': 1.4488, 'grad_norm': 0.0642375648021698, 'learning_rate': 0.00033041661861286123, 'epoch': 1.13}                                                                                                                                   
{'loss': 1.4382, 'grad_norm': 0.08360655605792999, 'learning_rate': 0.00031987518016481385, 'epoch': 1.18}                                                                                                                                  
{'loss': 1.438, 'grad_norm': 0.059480853378772736, 'learning_rate': 0.00030893770688531446, 'epoch': 1.23}                                                                                                                                  
{'loss': 1.4319, 'grad_norm': 0.07792435586452484, 'learning_rate': 0.0002976436218759248, 'epoch': 1.29}                                                                                                                                   
{'eval_loss': 1.3854539394378662, 'eval_runtime': 60.1631, 'eval_samples_per_second': 675.198, 'eval_steps_per_second': 1.064, 'epoch': 1.29}                                                                                               
 43%|█████████████████████████████████████████████████████████████████████████████████▍                                                                                                            | 2500/5832 [8:30:54<11:13:26, 12.13s/it]/mnt/jezeehu/gen_playlist/gen-venv/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py:4807: UserWarning: No device id is provided via `init_process_group` or `barrier `. Using the current device set by the user.          
  warnings.warn(  # warn only once
{'loss': 1.4296, 'grad_norm': 0.0643736720085144, 'learning_rate': 0.00028603363361190187, 'epoch': 1.34}                                                                                                                                   
{'loss': 1.4322, 'grad_norm': 0.06530717760324478, 'learning_rate': 0.00027414958921229853, 'epoch': 1.39}                                                                                                                                  
{'loss': 1.4231, 'grad_norm': 0.06650301814079285, 'learning_rate': 0.0002620343236059285, 'epoch': 1.44}                                                                                                                                   
{'loss': 1.4184, 'grad_norm': 0.6447392106056213, 'learning_rate': 0.0002497315051368639, 'epoch': 1.49}                                                                                                                                    
{'loss': 1.4143, 'grad_norm': 0.06335574388504028, 'learning_rate': 0.00023728547816596478, 'epoch': 1.54}                                                                                                                                  
{'eval_loss': 1.3708056211471558, 'eval_runtime': 60.147, 'eval_samples_per_second': 675.379, 'eval_steps_per_second': 1.064, 'epoch': 1.54}                                                                                                
 51%|█████████████████████████████████████████████████████████████████████████████████████████████████▋                                                                                            | 3000/5832 [10:13:07<9:31:50, 12.12s/it]/mnt/jezeehu/gen_playlist/gen-venv/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py:4807: UserWarning: No device id is provided via `init_process_group` or `barrier `. Using the current device set by the user.          
  warnings.warn(  # warn only once
{'loss': 1.4157, 'grad_norm': 0.07428540289402008, 'learning_rate': 0.00022474110323576898, 'epoch': 1.59}                                                                                                                                  
{'loss': 1.4126, 'grad_norm': 0.06722905486822128, 'learning_rate': 0.00021214359537485467, 'epoch': 1.65}                                                                                                                                  
{'loss': 1.4115, 'grad_norm': 0.08636120706796646, 'learning_rate': 0.00019953836112448906, 'epoch': 1.7}                                                                                                                                   
{'loss': 1.4089, 'grad_norm': 0.05577811971306801, 'learning_rate': 0.0001869708348749891, 'epoch': 1.75}                                                                                                                                   
{'loss': 1.4031, 'grad_norm': 0.11574093997478485, 'learning_rate': 0.00017448631510170234, 'epoch': 1.8}                                                                                                                                   
{'eval_loss': 1.3608745336532593, 'eval_runtime': 60.2411, 'eval_samples_per_second': 674.324, 'eval_steps_per_second': 1.062, 'epoch': 1.8}                                                                                                
 60%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                                            | 3500/5832 [11:55:16<7:50:21, 12.10s/it]/mnt/jezeehu/gen_playlist/gen-venv/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py:4807: UserWarning: No device id is provided via `init_process_group` or `barrier `. Using the current device set by the user.          
  warnings.warn(  # warn only once
{'loss': 1.406, 'grad_norm': 0.05993891879916191, 'learning_rate': 0.00016212980109088024, 'epoch': 1.85}                                                                                                                                   
{'loss': 1.4052, 'grad_norm': 0.18539205193519592, 'learning_rate': 0.00014994583074395118, 'epoch': 1.9}                                                                                                                                   
{'loss': 1.4004, 'grad_norm': 0.10395051538944244, 'learning_rate': 0.0001379783200448119, 'epoch': 1.95}                                                                                                                                   
{'loss': 1.4003, 'grad_norm': 0.05507673695683479, 'learning_rate': 0.00012627040476876226, 'epoch': 2.01}                                                                                                                                  
{'loss': 1.3921, 'grad_norm': 0.07021324336528778, 'learning_rate': 0.00011486428500362986, 'epoch': 2.06}                                                                                                                                  
{'eval_loss': 1.3541339635849, 'eval_runtime': 60.181, 'eval_samples_per_second': 674.997, 'eval_steps_per_second': 1.063, 'epoch': 2.06}                                                                                                   
 69%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                           | 4000/5832 [13:37:25<6:09:36, 12.11s/it]/mnt/jezeehu/gen_playlist/gen-venv/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py:4807: UserWarning: No device id is provided via `init_process_group` or `barrier `. Using the current device set by the user.          
  warnings.warn(  # warn only once
{'loss': 1.395, 'grad_norm': 0.06551236659288406, 'learning_rate': 0.00010380107304349155, 'epoch': 2.11}                                                                                                                                   
{'loss': 1.3983, 'grad_norm': 0.07631615549325943, 'learning_rate': 9.312064520324756e-05, 'epoch': 2.16}                                                                                                                                   
{'loss': 1.3994, 'grad_norm': 0.053269848227500916, 'learning_rate': 8.286149808816561e-05, 'epoch': 2.21}                                                                                                                                  
{'loss': 1.3905, 'grad_norm': 0.12374285608530045, 'learning_rate': 7.306060983646108e-05, 'epoch': 2.26}                                                                                                                                   
{'loss': 1.393, 'grad_norm': 0.19865798950195312, 'learning_rate': 6.375330683504886e-05, 'epoch': 2.31}                                                                                                                                    
{'eval_loss': 1.3496205806732178, 'eval_runtime': 60.0912, 'eval_samples_per_second': 676.006, 'eval_steps_per_second': 1.065, 'epoch': 2.31}                                                                                               
 77%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                           | 4500/5832 [15:19:30<4:29:21, 12.13s/it]/mnt/jezeehu/gen_playlist/gen-venv/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py:4807: UserWarning: No device id is provided via `init_process_group` or `barrier `. Using the current device set by the user.          
  warnings.warn(  # warn only once
{'loss': 1.3938, 'grad_norm': 0.3210863471031189, 'learning_rate': 5.497313638887913e-05, 'epoch': 2.37}                                                                                                                                    
{'loss': 1.3933, 'grad_norm': 0.3070966303348541, 'learning_rate': 4.6751745802807614e-05, 'epoch': 2.42}                                                                                                                                   
{'loss': 1.3891, 'grad_norm': 0.10366880148649216, 'learning_rate': 3.911876831183955e-05, 'epoch': 2.47}                                                                                                                                   
{'loss': 1.3915, 'grad_norm': 0.055540747940540314, 'learning_rate': 3.2101716270899577e-05, 'epoch': 2.52}                                                                                                                                 
{'loss': 1.3911, 'grad_norm': 0.0814662054181099, 'learning_rate': 2.5725881989116675e-05, 'epoch': 2.57}                                                                                                                                   
{'eval_loss': 1.3469421863555908, 'eval_runtime': 60.0737, 'eval_samples_per_second': 676.203, 'eval_steps_per_second': 1.065, 'epoch': 2.57}                                                                                               
 86%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                           | 5000/5832 [17:01:38<2:47:51, 12.10s/it]/mnt/jezeehu/gen_playlist/gen-venv/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py:4807: UserWarning: No device id is provided via `init_process_group` or `barrier `. Using the current device set by the user.          
  warnings.warn(  # warn only once
{'loss': 1.3896, 'grad_norm': 0.1887950748205185, 'learning_rate': 2.001424656605644e-05, 'epoch': 2.62}                                                                                                                                    
{'loss': 1.3922, 'grad_norm': 0.06595566868782043, 'learning_rate': 1.4987397058492607e-05, 'epoch': 2.67}                                                                                                                                  
{'loss': 1.391, 'grad_norm': 0.08022186905145645, 'learning_rate': 1.0663452276282738e-05, 'epoch': 2.73}                                                                                                                                   
{'loss': 1.3902, 'grad_norm': 0.04883366823196411, 'learning_rate': 7.057997474808929e-06, 'epoch': 2.78}                                                                                                                                   
{'loss': 1.3877, 'grad_norm': 0.054620612412691116, 'learning_rate': 4.184028179379719e-06, 'epoch': 2.83}                                                                                                                                  
{'eval_loss': 1.346081256866455, 'eval_runtime': 60.1573, 'eval_samples_per_second': 675.263, 'eval_steps_per_second': 1.064, 'epoch': 2.83}                                                                                                
 94%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏          | 5500/5832 [18:43:46<1:06:55, 12.10s/it]/mnt/jezeehu/gen_playlist/gen-venv/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py:4807: UserWarning: No device id is provided via `init_process_group` or `barrier `. Using the current device set by the user.          
  warnings.warn(  # warn only once
{'loss': 1.3871, 'grad_norm': 0.0498526394367218, 'learning_rate': 2.0519033440720127e-06, 'epoch': 2.88}                                                                                                                                   
{'loss': 1.3871, 'grad_norm': 0.0707998052239418, 'learning_rate': 6.693080138478636e-07, 'epoch': 2.93}                                                                                                                                    
{'loss': 1.3884, 'grad_norm': 0.07536571472883224, 'learning_rate': 4.1225624527380104e-08, 'epoch': 2.98}                                                                                                                                  
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5832/5832 [19:50:58<00:00, 12.10s/it]/mnt/jezeehu/gen_playlist/gen-venv/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py:4807: UserWarning: No device id is provided via `init_process_group` or `barrier `. Using the current device set by the user. 
  warnings.warn(  # warn only once
There were missing keys in the checkpoint model loaded: ['encoder.embed_tokens.weight', 'decoder.embed_tokens.weight'].
There were missing keys in the checkpoint model loaded: ['encoder.embed_tokens.weight', 'decoder.embed_tokens.weight'].
There were missing keys in the checkpoint model loaded: ['encoder.embed_tokens.weight', 'decoder.embed_tokens.weight'].
There were missing keys in the checkpoint model loaded: ['encoder.embed_tokens.weight', 'decoder.embed_tokens.weight'].
2025-11-25 11:03:46,203 - transformers.trainer - WARNING - There were missing keys in the checkpoint model loaded: ['encoder.embed_tokens.weight', 'decoder.embed_tokens.weight'].
{'train_runtime': 71469.2888, 'train_samples_per_second': 167.109, 'train_steps_per_second': 0.082, 'train_loss': 1.669920513822872, 'epoch': 3.0}                                                                                          
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5832/5832 [19:51:09<00:00, 12.25s/it]
2025-11-25 11:03:46,229 - __main__ - INFO - 
================================================================================
2025-11-25 11:03:46,229 - __main__ - INFO - TRAINING COMPLETED SUCCESSFULLY
2025-11-25 11:03:46,229 - __main__ - INFO - ================================================================================
2025-11-25 11:03:46,229 - __main__ - INFO - Total training time: 19.85 hours
2025-11-25 11:03:46,229 - __main__ - INFO - Average time per step: 12.26s
2025-11-25 11:03:46,229 - __main__ - INFO - ================================================================================

2025-11-25 11:03:50,162 - __main__ - INFO - 
================================================================================
2025-11-25 11:03:50,163 - __main__ - INFO - MODEL SAVED
2025-11-25 11:03:50,163 - __main__ - INFO - ================================================================================
2025-11-25 11:03:50,163 - __main__ - INFO - Final model path: models/generator/final_model
2025-11-25 11:03:50,163 - __main__ - INFO - Model size: ~0.00 GB
2025-11-25 11:03:50,163 - __main__ - INFO - ================================================================================


analyze_voting_results.py部分输出日志如下：
2025-11-25 12:04:01,099 - __main__ - INFO - 4. 语义ID维度对比分析
2025-11-25 12:04:01,099 - __main__ - INFO - ================================================================================
2025-11-25 12:04:01,190 - __main__ - INFO - 有效对比数量: 500
2025-11-25 12:04:01,190 - __main__ - INFO - 平均Jaccard相似度（3层完全匹配）: 0.0645
2025-11-25 12:04:01,190 - __main__ - INFO - 平均Jaccard相似度（前2层匹配）: 0.1055
2025-11-25 12:04:01,190 - __main__ - INFO - 平均Jaccard相似度（第1层匹配）: 0.2030
2025-11-25 12:04:01,190 - __main__ - INFO - 
--------------------------------------------------------------------------------
2025-11-25 12:04:01,190 - __main__ - INFO - 指标说明:
2025-11-25 12:04:01,190 - __main__ - INFO - --------------------------------------------------------------------------------
2025-11-25 12:04:01,190 - __main__ - INFO - 语义ID Jaccard相似度 = |投票语义IDs ∩ 生成语义IDs| / |投票语义IDs ∪ 生成语义IDs|
2025-11-25 12:04:01,190 - __main__ - INFO -   - 3层完全匹配: 比较完整的语义ID (L1, L2, L3)
2025-11-25 12:04:01,190 - __main__ - INFO -   - 前2层匹配: 只比较 (L1, L2)，忽略L3
2025-11-25 12:04:01,190 - __main__ - INFO -   - 第1层匹配: 只比较 L1，忽略L2和L3
2025-11-25 12:04:01,190 - __main__ - INFO -   - 范围: [0, 1]，越高表示语义理解越准确
2025-11-25 12:04:01,190 - __main__ - INFO - 
--------------------------------------------------------------------------------
2025-11-25 12:04:01,190 - __main__ - INFO - 语义ID最好的10条结果:
2025-11-25 12:04:01,190 - __main__ - INFO - --------------------------------------------------------------------------------
2025-11-25 12:04:01,190 - __main__ - INFO - 
[1] Query: 2008年qq飞车
2025-11-25 12:04:01,190 - __main__ - INFO -     Jaccard(3层): 0.3429, Jaccard(L1+L2): 0.4000, Jaccard(L1): 0.7000
2025-11-25 12:04:01,190 - __main__ - INFO -     投票语义ID数: 46, 生成语义ID数: 48
2025-11-25 12:04:01,190 - __main__ - INFO - 
    投票Top10语义ID:
2025-11-25 12:04:01,190 - __main__ - INFO -        1. (95, 89, 115) - 28388641 - 불꽃 (火花) - 高耀太 (vote=-1)
2025-11-25 12:04:01,190 - __main__ - INFO -        2. (29, 82, 150) - 60576270 - 죽을 만큼 아파서 (死一样的痛苦)(Feat. 멜로우) - MC 몽、Mellow (vote=-1)
2025-11-25 12:04:01,190 - __main__ - INFO -        3. (29, 18, 252) - 88976381 - Right Now (Na Na Na) - Akon (vote=-1)
2025-11-25 12:04:01,190 - __main__ - INFO -        4. (29, 82, 253) - 38226427 - Insomnia (불면증) - 辉星 (vote=-1)
2025-11-25 12:04:01,190 - __main__ - INFO -        5. (13, 56, 102) - 62029801 - 无限速 - 本兮、阿悄 (vote=-1)
2025-11-25 12:04:01,190 - __main__ - INFO -        6. (29, 96, 254) - 32021618 - She Is My Sin - Nightwish (vote=-1)
2025-11-25 12:04:01,190 - __main__ - INFO -        7. (29, 82, 253) - 40291179 - Tik Tok - 2PM、尹恩惠 (vote=-1)
2025-11-25 12:04:01,190 - __main__ - INFO -        8. (29, 18, 252) - 90812099 - Loves Me Not - t.A.T.u. (vote=-1)
2025-11-25 12:04:01,190 - __main__ - INFO -        9. (55, 10, 254) - 64419569 - 하루도 (Not A Single Day) (Not A Single Day) - Rain (vote=-1)
2025-11-25 12:04:01,190 - __main__ - INFO -       10. (107, 83, 252) - 68881270 - 超速度 - 韦琪、本兮 (vote=-1)
2025-11-25 12:04:01,190 - __main__ - INFO - 
    生成Top10语义ID:
2025-11-25 12:04:01,190 - __main__ - INFO -        1. (29, 82, 150) - 60576270 - 죽을 만큼 아파서 (死一样的痛苦)(Feat. 멜로우) - MC 몽、Mellow
2025-11-25 12:04:01,190 - __main__ - INFO -        2. (79, 103, 252) - 196552952 - Don't push me - Sweetbox
2025-11-25 12:04:01,190 - __main__ - INFO -        3. (29, 18, 252) - 54108355 - Insomnia - Craig David
2025-11-25 12:04:01,190 - __main__ - INFO -        4. (79, 103, 229) - 33220625 - Your Love Is My Drug - Kesha
2025-11-25 12:04:01,190 - __main__ - INFO -        5. (29, 82, 253) - 32253934 - 죽을만큼 아파서 Part.2 (死一样的痛过 Part.2) - MC 몽、Sweden Laundry
2025-11-25 12:04:01,190 - __main__ - INFO -        6. (29, 96, 254) - 32021618 - She Is My Sin - Nightwish
2025-11-25 12:04:01,190 - __main__ - INFO -        7. (29, 37, 106) - 27524324 - Crazy Kids - Kesha
2025-11-25 12:04:01,190 - __main__ - INFO -        8. (29, 106, 253) - 28106150 - That's Not My Name - The Ting Tings
2025-11-25 12:04:01,190 - __main__ - INFO -        9. (2, 63, 253) - 104663098 - 极速梦想 - 丹戈尔
2025-11-25 12:04:01,190 - __main__ - INFO -       10. (29, 106, 51) - 28374059 - What I Believe - Skillet