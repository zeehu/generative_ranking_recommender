# Generative Ranking Recommender (生成式排序推荐器)

## 📖 项目简介

本项目是一个**先进的、端到端的歌曲推荐系统**，采用创新的"**生成-排序 (Generate-and-Rank)**"架构，突破了传统推荐系统的局限。

### 🎯 核心创新

与传统的"召回-排序"架构不同，本系统将推荐问题转化为**序列生成问题**，通过以下三个阶段实现高质量推荐：

#### 1. 语义ID生成 (Semantic ID Generation)
- 使用**层次化残差量化K-Means (Hierarchical RQ-KMeans)** 算法
- 将每首歌曲的连续向量（512维）量化为离散的、层次化的语义ID
- 三层结构设计：`[128, 128, 256]`，每首歌曲对应一个三元组ID，如 `(45, 67, 123)`
- 支持训练中断恢复、动态batch size优化、内存高效处理

#### 2. 候选生成 (Candidate Generation)
- 基于**T5生成模型 (TIGER Model)** 的序列到序列学习
- 输入：用户文本（歌单标题、心情描述等）
- 输出：语义ID序列，如 `<id_l1_45> <id_l2_67> <id_l3_123> <sep> <id_l1_12> ...`
- 通过语义ID映射回具体歌曲，生成100首候选歌曲
- 利用生成模型的创造力，召回的候选集相关性极高

#### 3. 精准排序 (Precise Ranking)
- 使用**混合Cross-Encoder模型**进行深度交互
- 融合文本语义和歌曲向量特征
- 对候选歌曲逐一计算与输入文本的精准相关性分数
- 输出最终的Top-K推荐列表

---

## 📁 项目结构

```
generative_ranking_recommender/
├── data/                              # 原始数据文件 (git ignored)
│   ├── gen_playlist_song.csv.sort     # 歌单-歌曲关系数据
│   ├── gen_playlist_info.csv          # 歌单信息（标题、描述等）
│   └── gen_song_info.csv              # 歌曲信息（名称、歌手等）
│
├── outputs/                           # 中间输出文件 (git ignored)
│   ├── playlists_corpus.txt           # 预处理后的歌单语料
│   ├── sg1_vs512_w50_ep20_song_vectors.csv  # 歌曲向量 (512维)
│   ├── semantic_id/                   # 语义ID生成模块输出
│   │   └── song_semantic_ids.jsonl    # 歌曲语义ID映射
│   ├── generator/                     # 生成器训练数据
│   │   ├── train.tsv                  # 训练集
│   │   └── val.tsv                    # 验证集
│   └── ranker/                        # 排序器训练数据
│       └── ranking_train_data.tsv     # 排序训练数据
│
├── models/                            # 训练好的模型 (git ignored)
│   ├── generator/                     # T5生成模型
│   │   └── final_model/               # 最终模型
│   └── ranker/                        # Cross-Encoder排序模型
│       └── final_model/               # 最终模型
│
├── src/                               # 源代码
│   ├── common/                        # 通用工具模块
│   │   ├── utils.py                   # 工具函数（随机种子、日志等）
│   │   ├── preprocess_playlists.py    # 歌单预处理
│   │   ├── train_word2vec.py          # Word2Vec训练
│   │   └── evaluate_vectors.py        # 向量质量评估
│   │
│   ├── semantic_id_generator/         # 语义ID生成模块
│   │   ├── hierarchical_rq_kmeans.py  # 核心算法实现
│   │   ├── simplified_semantic_id_generator.py  # 简化版生成器
│   │   ├── train_semantic_ids.py      # 完整版训练脚本
│   │   ├── evaluate_semantic_ids.py   # 交互式评估工具
│   │   ├── calculate_metrics.py       # 批量指标计算
│   │   ├── debug_collisions.py        # 冲突检测工具
│   │   └── balancekmeans/             # 平衡K-Means实现
│   │       ├── main.py                # K-Means核心算法
│   │       └── soft_dtw_cuda.py       # CUDA加速距离计算
│   │
│   ├── generator/                     # 生成器模块
│   │   ├── tiger_model.py             # TIGER模型定义
│   │   ├── prepare_corpus.py          # 语料准备
│   │   ├── train_t5.py                # T5训练脚本
│   │   └── inference_t5.py            # T5推理脚本
│   │
│   ├── ranker/                        # 排序器模块
│   │   ├── ranker_model.py            # Cross-Encoder模型定义
│   │   ├── prepare_data.py            # 数据准备
│   │   └── train_ranker.py            # 排序器训练脚本
│   │
│   └── demo.py                        # 完整推荐系统演示
│
├── logs/                              # 日志文件 (git ignored)
├── .gitignore                         # Git忽略配置
├── config.py                          # 全局配置文件
├── requirements.txt                   # Python依赖
└── README.md                          # 项目说明文档
```

---

## ⚙️ 环境要求与安装

### 系统要求
- **Python**: 3.8+
- **CUDA**: 11.0+ (推荐用于GPU加速)
- **内存**: 建议32GB+ (处理大规模数据集)
- **GPU**: 建议NVIDIA GPU with 16GB+ VRAM (用于模型训练)

### 安装步骤

1. **克隆项目**
   ```bash
   git clone [您的仓库地址]
   cd generative_ranking_recommender
   ```

2. **创建虚拟环境** (推荐)
   ```bash
   conda create -n genrank python=3.8
   conda activate genrank
   ```

3. **安装PyTorch** (根据您的CUDA版本)
   ```bash
   # CUDA 11.8示例
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```

4. **安装FAISS** (向量检索库)
   ```bash
   # GPU版本 (推荐)
   conda install -c pytorch faiss-gpu
   
   # 或CPU版本
   conda install -c pytorch faiss-cpu
   ```

5. **安装其他依赖**
   ```bash
   pip install -r requirements.txt
   ```

### 验证安装
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import faiss; print(f'FAISS: {faiss.__version__}')"
```

---

## 🚀 完整实施路线图

本项目采用模块化设计，按照以下步骤逐步构建完整的推荐系统。

### **阶段0：基础向量学习 (G0)**

**目标**: 从原始歌单数据中学习高质量的歌曲向量表示

#### G0a - 预处理歌单数据
```bash
python src/common/preprocess_playlists.py
```
- **输入**: `data/gen_playlist_song.csv.sort`
- **处理**: 筛选长度在10-300之间的歌单，生成Word2Vec训练语料
- **输出**: `outputs/playlists_corpus.txt` (每行一个歌单的歌曲ID序列)
- **配置**: `config.py` → `Word2VecConfig`

#### G0b - 训练歌曲向量
```bash
python src/common/train_word2vec.py
```
- **输入**: `outputs/playlists_corpus.txt`
- **算法**: FastText (CBOW模式)
- **参数**: 
  - 向量维度: 512
  - 窗口大小: 50
  - 训练轮数: 20
  - 最小词频: 10
- **输出**: `outputs/sg1_vs512_w50_ep20_song_vectors.csv`
- **说明**: 将歌曲视为"词"，歌单视为"句子"，学习歌曲的分布式表示

#### G0c - 评估向量质量 (可选)
```bash
python src/common/evaluate_vectors.py
```
- **功能**: 交互式工具，输入歌曲ID查找最相似的歌曲
- **用途**: 定性评估向量学习效果，验证相似歌曲是否语义相关

---

### **阶段1：语义ID生成与评估 (G1)**

**目标**: 将连续向量量化为离散的层次化语义ID，实现高效的序列生成

#### G1a - 生成语义ID
```bash
python src/semantic_id_generator/simplified_semantic_id_generator.py
```
- **输入**: `outputs/sg1_vs512_w50_ep20_song_vectors.csv`
- **算法**: 层次化残差量化K-Means (Hierarchical RQ-KMeans)
- **核心特性**:
  - **三层结构**: `[128, 128, 256]` 聚类中心
  - **残差学习**: 每层学习上一层的残差，逐步细化表示
  - **平衡聚类**: 确保每个簇的样本数量相对均衡
  - **中断恢复**: 支持训练中断后从检查点继续
  - **动态优化**: 自适应batch size和迭代次数
- **输出**: `outputs/semantic_id/song_semantic_ids.jsonl`
  ```json
  {"song_id": "123456", "semantic_ids": [45, 67, 123]}
  ```
- **配置**: `config.py` → `HierarchicalRQKMeansConfig`

#### G1b - 交互式评估
```bash
python src/semantic_id_generator/evaluate_semantic_ids.py
```
- **功能1**: 输入歌曲ID → 查找相同语义ID的歌曲
- **功能2**: 输入语义ID → 查找该簇下的所有歌曲
- **用途**: 定性评估聚类质量，验证语义相似性

#### G1c - 冲突检测
```bash
python src/semantic_id_generator/debug_collisions.py
```
- **检查**: 是否存在不同歌曲被分配到相同语义ID的冲突
- **输出**: 冲突统计报告和详细列表

#### G1d - 定量评估
```bash
python src/semantic_id_generator/calculate_metrics.py
```
- **指标**:
  - **轮廓系数 (Silhouette Score)**: 衡量簇内紧密性和簇间分离度
  - **CH指数 (Calinski-Harabasz)**: 簇间离散度与簇内离散度的比值
  - **DB指数 (Davies-Bouldin)**: 簇内散度与簇间距离的比率
- **优化**: 采用数据抽样策略，高效处理大规模数据集

---

### **阶段2：生成器训练 (G2) - TIGER模型**

**目标**: 训练T5模型将文本转换为语义ID序列

#### G2a - 准备训练语料
```bash
python src/generator/prepare_corpus.py
```
- **输入**: 
  - `data/gen_playlist_info.csv` (歌单标题、描述)
  - `data/gen_playlist_song.csv.sort` (歌单-歌曲关系)
  - `outputs/semantic_id/song_semantic_ids.jsonl` (语义ID映射)
- **处理**: 构建 `(文本, 语义ID序列)` 训练对
- **输出**: 
  - `outputs/generator/train.tsv`
  - `outputs/generator/val.tsv`
- **格式示例**:
  ```
  playlist_id\t文本描述\t<id_l1_45> <id_l2_67> <id_l3_123> <sep> <id_l1_12> ...
  ```

#### G2b - 训练TIGER模型
```bash
python src/generator/train_t5.py
```
- **基础模型**: T5 (mengzi-t5-base)
- **自定义Tokenizer**: 
  - 添加层级语义ID tokens: `<id_l1_0>` ~ `<id_l1_127>`, `<id_l2_0>` ~ `<id_l2_127>`, `<id_l3_0>` ~ `<id_l3_255>`
  - 特殊tokens: `<eos>`, `<sep>`
- **训练参数**:
  - Batch size: 160
  - Learning rate: 2e-4
  - Epochs: 5
  - FP16: 启用
  - Gradient checkpointing: 启用
- **输出**: `models/generator/final_model/`
- **核心创新**: 将推荐问题转化为序列到序列生成任务

---

### **阶段3：排序器训练 (R) - Cross-Encoder**

**目标**: 训练混合Cross-Encoder模型进行精准相关性评分

#### R1 - 准备训练数据
```bash
python src/ranker/prepare_data.py
```
- **输入**: 
  - `data/gen_playlist_info.csv`
  - `data/gen_playlist_song.csv.sort`
  - `outputs/semantic_id/song_semantic_ids.jsonl`
- **策略**: 
  - **正样本**: 歌单中的真实歌曲 (label=1)
  - **负样本**: 随机采样的不相关歌曲 (label=0)
- **输出**: `outputs/ranker/ranking_train_data.tsv`
- **格式**: `text\tsong_id\tlabel`

#### R2 - 模型架构
**文件**: `src/ranker/ranker_model.py`

**混合Cross-Encoder架构**:
```
输入: (文本, 歌曲向量)
  ↓
文本编码 (T5 Encoder) + 歌曲向量投影
  ↓
特征融合 (Concatenation)
  ↓
T5 Encoder深度交互
  ↓
池化 + 分类层
  ↓
输出: 相关性分数 (0-1)
```

**关键特性**:
- **文本-向量融合**: 将512维歌曲向量投影到T5隐藏空间
- **深度交互**: 利用T5 Encoder的自注意力机制
- **端到端训练**: 联合优化文本理解和歌曲匹配

#### R3 - 训练排序器
```bash
python src/ranker/train_ranker.py
```
- **任务**: 二分类 (相关/不相关)
- **损失函数**: BCEWithLogitsLoss
- **训练参数**:
  - Batch size: 64
  - Learning rate: 1e-5
  - Epochs: 3
  - 评估指标: F1, Precision, Recall, Accuracy
- **输出**: `models/ranker/final_model/`

---

### **阶段4：完整推荐系统演示 (F)**

#### F1 - 运行推荐系统
```bash
python src/demo.py
```

**完整推理流程**:

1. **加载模型**
   - TIGER生成模型
   - Cross-Encoder排序模型
   - 语义ID映射表
   - 歌曲向量和信息

2. **候选生成阶段**
   ```python
   输入: "适合深夜听的抒情歌曲"
     ↓
   TIGER模型生成语义ID序列
     ↓
   <id_l1_45> <id_l2_67> <id_l3_123> <sep> <id_l1_12> <id_l2_89> <id_l3_201> ...
     ↓
   通过映射表转换为歌曲ID
     ↓
   候选集: [song_123, song_456, song_789, ...] (100首)
   ```

3. **精准排序阶段**
   ```python
   对每首候选歌曲:
     输入: ("适合深夜听的抒情歌曲", song_vector)
       ↓
     Cross-Encoder计算相关性分数
       ↓
     score = 0.87
   
   按分数降序排序
     ↓
   Top-10推荐列表
   ```

4. **输出结果**
   ```
   ✨ 为您推荐的最终歌曲列表 (已排序): ✨
     1. 夜空中最亮的星 - 逃跑计划 (ID: 123456)
     2. 安静 - 周杰伦 (ID: 234567)
     3. 演员 - 薛之谦 (ID: 345678)
     ...
   ```

**交互式演示**:
- 支持连续输入多个查询
- 实时生成和排序
- 输入 `exit` 或 `quit` 退出

---

## 📊 技术细节

### 核心算法：层次化残差量化K-Means

**算法原理**:
```
第1层: 直接聚类
  输入: 原始向量 X
  输出: 聚类中心 C1, 聚类ID L1
  残差: R1 = X - C1[L1]

第2层: 递归聚类
  输入: 归一化残差 R1
  策略: 在第1层每个簇内分别聚类
  输出: 聚类中心 C2, 聚类ID L2
  残差: R2 = R1 - C2[L2]

第3层: 双模型+匹配矩阵
  输入: 归一化残差 R2
  策略: 训练2个K-Means，通过匹配矩阵筛选最优中心
  输出: 聚类中心 C3, 聚类ID L3

最终语义ID: (L1, L2, L3)
```

**关键优化**:
- **动态Batch Size**: 根据GPU内存自动调整，防止OOM
- **自适应迭代次数**: 根据数据规模和聚类数量动态调整
- **检查点机制**: 支持训练中断恢复，节省计算资源
- **内存优化**: 使用原地操作、及时释放中间变量

### TIGER模型架构

**自定义Tokenizer**:
- 基于T5 Tokenizer扩展
- 添加512个层级语义ID tokens (128+128+256)
- 支持序列化保存和加载

**训练策略**:
- Seq2Seq学习: 文本 → 语义ID序列
- Beam Search解码: 生成多样化候选
- Gradient Checkpointing: 节省显存

### Cross-Encoder排序器

**混合架构优势**:
1. **文本理解**: T5 Encoder捕捉查询语义
2. **歌曲表示**: 512维向量编码歌曲特征
3. **深度交互**: 自注意力机制建模文本-歌曲关联
4. **端到端优化**: 联合训练，避免特征工程

---

## 💻 核心模块源码深度解析

### 1. 语义ID生成模块 (`src/semantic_id_generator/train_semantic_ids.py`)

该脚本是语义ID生成的总控程序，负责协调数据加载、模型训练、ID生成和统计分析。

*   **训练流程管理 (`SemanticIDTrainer`)**:
    *   **多模式支持**: 支持 `PROD` (全量数据) 和 `TEST` (10w条采样) 两种模式，便于快速调试。
    *   **断点续训**: 内置检查点 (Checkpoint) 机制，支持从中断处恢复训练，这对于处理大规模数据至关重要。
    *   **统计分析**: 训练完成后自动计算每层聚类的分布统计（如簇的利用率、样本分布的均值/方差），帮助评估量化质量。

*   **数据处理优化**:
    *   使用 `tqdm` 监控数据加载进度。
    *   支持流式读取 CSV 文件，避免一次性加载过多数据导致 OOM。

*   **核心方法**:
    *   `train()`: 标准化的训练流水线：加载数据 -> 初始化模型 -> 训练 -> 保存模型 -> 生成ID -> 计算统计。
    *   `_generate_semantic_ids()`: 将模型输出的聚类索引转换为层次化的语义ID元组 `(l1, l2, l3)`。

### 2. 生成器训练模块 (`src/generator/train_t5_optimized.py`)

这是针对大规模数据训练 T5 模型的高度优化版本，集成了多项深度学习工程优化技术。

*   **极致的性能优化**:
    *   **内存映射 (Memory Mapping)**: 使用 `datasets.load_from_disk` 实现零内存拷贝的数据加载，极大降低内存占用。
    *   **预分词 (Pre-tokenization)**: 移除了训练循环中的分词操作，直接读取预处理好的 Token ID，显著提升数据吞吐量。
    *   **混合精度训练 (FP16)**: 启用 FP16 自动混合精度，减少显存占用并利用 Tensor Cores 加速计算。
    *   **梯度检查点 (Gradient Checkpointing)**: 牺牲少量计算换取大量显存，支持更大的 Batch Size。
    *   **PyTorch 2.0 编译**: 集成 `torch.compile`，通过图优化进一步提升训练速度 (5-15%)。

*   **工程化设计**:
    *   **硬件感知**: 自动检测 GPU 数量和显存，打印详细的硬件配置报告。
    *   **配置解耦**: 优先加载 `config_optimized.py`，实现针对特定硬件环境的参数调优。
    *   **监控友好**: 详细计算并打印“预期训练时间”、“显存利用率”等关键指标，便于运维监控。

*   **自定义组件**:
    *   `MemoryMappedDataset`: 专为超大数据集设计的 Dataset 类。
    *   `PrePaddedDataCollator`: 简化的 Collator，因为数据已在预处理阶段完成 Padding，进一步减少 CPU 负载。

---

## 📈 评估指标

### 语义ID质量评估

| 指标 | 说明 | 取值范围 | 优化目标 |
|------|------|----------|----------|
| **轮廓系数** (Silhouette Score) | 簇内紧密性 vs 簇间分离度 | [-1, 1] | 越高越好 |
| **CH指数** (Calinski-Harabasz) | 簇间离散度 / 簇内离散度 | [0, +∞) | 越高越好 |
| **DB指数** (Davies-Bouldin) | 簇内散度 / 簇间距离 | [0, +∞) | 越低越好 |

**计算策略**: 采用分层抽样，在保证统计显著性的同时优化计算效率

### 推荐系统评估

- **生成质量**: BLEU, ROUGE (语义ID序列与真实序列的匹配度)
- **排序质量**: NDCG, MRR, Precision@K, Recall@K
- **端到端**: 用户满意度、点击率、播放完成率

---

## 🔧 配置说明

### 核心配置文件: `config.py`

```python
# 数据路径配置
class DataConfig:
    playlist_songs_file: str  # 歌单-歌曲关系
    playlist_info_file: str   # 歌单信息
    song_info_file: str       # 歌曲信息
    song_vectors_file: str    # 歌曲向量
    semantic_ids_file: str    # 语义ID映射

# Word2Vec配置
class Word2VecConfig:
    vector_size: int = 512    # 向量维度
    window: int = 50          # 上下文窗口
    epochs: int = 20          # 训练轮数

# 语义ID生成配置
class HierarchicalRQKMeansConfig:
    layer_clusters: [128, 1280, 1280]  # 每层聚类中心数
    need_clusters: [128, 128, 256]     # 实际使用的聚类数
    embedding_dim: int = 512           # 向量维度
    iter_limit: int = 100              # 最大迭代次数

# T5生成器配置
class PlaylistTIGERConfig:
    model_name: str           # 基础T5模型路径
    max_input_length: int     # 最大输入长度
    max_target_length: int    # 最大输出长度
    learning_rate: float      # 学习率
    num_train_epochs: int     # 训练轮数
```

---

## 🎯 使用场景

### 1. 歌单续写
```
输入: "周末放松的轻音乐"
输出: [歌曲1, 歌曲2, ..., 歌曲10]
```

### 2. 情感推荐
```
输入: "失恋后适合听的歌"
输出: 情感匹配的歌曲列表
```

### 3. 场景推荐
```
输入: "健身房跑步BGM"
输出: 节奏感强的歌曲列表
```

### 4. 风格探索
```
输入: "类似周杰伦的中国风歌曲"
输出: 风格相似的歌曲推荐
```

---

## 🚨 常见问题

### Q1: 训练过程中出现OOM错误
**解决方案**:
- 减小batch size
- 启用gradient checkpointing
- 使用FP16混合精度训练
- 调整`_calculate_safe_batch_size`中的内存使用率

### Q2: 语义ID生成速度慢
**解决方案**:
- 使用GPU加速 (CUDA)
- 调整`iter_limit`参数
- 使用测试配置进行快速验证
- 启用检查点机制，分批训练

### Q3: 生成的推荐结果不理想
**解决方案**:
- 检查歌曲向量质量 (G0c评估)
- 评估语义ID聚类效果 (G1b, G1d)
- 增加训练数据量
- 调整生成器的beam size
- 微调排序器的训练参数

### Q4: 如何处理冷启动问题
**策略**:
- 新歌曲: 使用歌曲元信息 (歌手、风格) 预测初始向量
- 新用户: 基于流行度和多样性的混合推荐
- 增量更新: 定期重新训练向量和语义ID

---

## 📚 参考文献

1. **TIGER**: "TIGER: Text-to-Image Grounding for Image Caption Evaluation and Retrieval"
2. **RQ-VAE**: "Residual Quantization for Vector Compression"
3. **Cross-Encoder**: "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
4. **T5**: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"

---

### **附录**
#### **语义 id 评估指标说明**
1. 轮廓系数 (Silhouette Score)(Higher is better)
   * 衡量标准：同时考虑了簇内紧密性和簇间分离度。对于每个样本，它计算该样本与其自身簇的平均距离，以及与最近邻簇的平均距离。
   * 取值范围：-1 到 +1。
   * 解释：
       * +1：表示样本与其自身簇非常匹配，且与相邻簇分离良好。
       * 0：表示样本位于两个簇的边界上。
       * -1：表示样本可能被分配到了错误的簇。
   * 整体评估：所有样本的平均轮廓系数。值越高越好。
   * 优点：直观，综合考虑了内聚和分离。
   * 缺点：计算成本较高，对于180万条数据，直接计算可能会非常慢。通常需要对数据进行抽样来计算。

2. Calinski-Harabasz 指数 (CH Index / 方差比准则)(Higher is better)
   * 衡量标准：簇内离散度与簇间离散度的比值。
   * 取值范围：无上限，通常为正数。
   * 解释：值越高越好，表示簇的定义越清晰，分离度越高。
   * 优点：计算速度相对较快。
   * 缺点：倾向于选择密度高、球形的簇。

3. Davies-Bouldin 指数 (DB Index)(Lower is better)
   * 衡量标准：衡量簇内散度与簇间距离的比率。
   * 取值范围：通常为正数。
   * 解释：值越低越好，表示簇内紧密性高，簇间分离度高。
   * 优点：计算速度相对较快。
   * 缺点：对噪声和异常值敏感。

---

## 📄 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

---

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

---

## 📧 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至: [zee_h@qq.com]

---

**⭐ 如果这个项目对您有帮助，请给个Star支持一下！**
