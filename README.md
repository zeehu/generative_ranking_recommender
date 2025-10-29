# Generative Ranking Recommender (生成式排序推荐器)

本项目是一个先进的、端到端的歌曲推荐系统。它没有采用传统的“召回-排序”架构，而是使用了一个更前沿、更统一的“**生成-排序 (Generate-and-Rank)**”架构。

## 核心架构

1.  **候选生成 (Candidate Generation)**: 我们首先利用一个强大的 **T5生成模型**，根据用户输入的任意文本（如歌单标题、心情描述），创造性地生成一个高质量、多样化的候选歌曲池（例如100首歌）。这一步利用了生成模型的创造力，召回的候选集相关性非常高。

2.  **精准排序 (Precise Ranking)**: 接着，我们使用一个**Cross-Encoder相关性模型**，对第一阶段生成的100个候选歌曲，逐一计算它们与输入文本的“精准相关性”分数。

3.  **最终输出**: 最后，我们根据这个分数对候选歌曲进行排序，得到最终的、高质量的Top-K推荐列表。

## 📁 项目结构

```
generative_ranking_recommender/
├── data/                     # 存放您的原始数据文件 (git ignored)
├── models/                   # 存放所有训练好的模型 (git ignored)
│   ├── generator/            # T5生成模型
│   └── ranker/               # Cross-Encoder排序模型
├── outputs/                  # 存放所有中间文件 (git ignored)
│   ├── semantic_id/          # 语义ID生成模块的输出
├── src/                      # 源代码
│   ├── common/               # 通用工具 (如日志、随机种子)
│   ├── semantic_id_generator/# 【新增】语义ID生成与评估模块
│   ├── generator/            # 【引擎一】生成模型的全部代码
│   └── ranker/               # 【引擎二】排序模型的代码
├── .gitignore                # Git忽略文件配置
├── config.py                 # 项目全局配置 (新增语义ID生成参数)
├── requirements.txt          # 项目依赖
└── README.md                 # 项目总说明
```

## ⚙️ 环境与安装

1.  **克隆项目**: `git clone [您的仓库地址]`
2.  **进入目录**: `cd generative_ranking_recommender`
3.  **安装依赖**: 
    ```bash
    # 强烈建议先根据您的CUDA环境手动安装torch和faiss
    # 例如: conda install -c pytorch torch torchvision faiss-gpu
    
    # 然后通过 requirements.txt 安装其余的库
    pip install -r requirements.txt
    ```

## 🚀 最终实施路线图

我们将严格按照此路线图，一步一步完成整个工程。

### **第零部分：基础向量学习 (G0)**

这一部分旨在从原始数据高效地生成高质量的歌曲向量。

1.  **G0a - 预处理歌单数据**:
    *   **命令**: `python src/common/preprocess_playlists.py`
    *   **作用**: 读取原始的歌单-歌曲文件，筛选长度在10到300之间的歌单，并生成一个内存优化的、用于`word2vec`训练的语料文件。
    *   **输出**: `outputs/playlists_corpus.txt`

2.  **G0b - 训练歌曲向量**:
    *   **命令**: `python src/common/train_word2vec.py`
    *   **作用**: 高效地读取上一步生成的语料文件，使用`FastText(CBOW)`模型为每首歌曲学习一个基础向量。
    *   **输出**: `outputs/song_vectors.csv`

3.  **G0c - (可选) 评估向量质量**:
    *   **命令**: `python src/common/evaluate_vectors.py`
    *   **作用**: 启动一个交互式界面，输入歌曲ID，查找最相似的歌曲，从而定性地判断向量学习的效果。

### **第一部分：语义ID生成与评估 (G1)**

这一部分将歌曲的连续向量量化为离散的、层次化的语义ID，并提供工具进行评估。

1.  **G1a - 生成语义ID**:
    *   **命令**: `python src/semantic_id_generator/simplified_semantic_id_generator.py`
    *   **作用**: 读取 `config.py` 中配置的歌曲向量文件 (`config.data.song_vectors_file`)，使用层次化残差平衡K-Means算法，为每首歌曲生成一个三层的语义ID。
    *   **输出**: `outputs/semantic_id/song_semantic_ids.jsonl` (JSONL格式)。
    *   **配置**: 训练参数（如聚类层数、每层聚类数、迭代次数等）在 `config.py` 的 `HierarchicalRQKMeansConfig` 中定义，并根据 `TEST_DATA_LIMIT` 自动切换生产/测试配置。

2.  **G1b - 交互式评估语义ID质量**:
    *   **命令**: `python src/semantic_id_generator/evaluate_semantic_ids.py`
    *   **作用**: 提供一个交互式工具，用于定性评估生成的语义ID质量。您可以输入歌曲ID查找相似歌曲，或输入语义ID查找该簇下的歌曲。

3.  **G1c - 检查语义ID冲突**:
    *   **命令**: `python src/semantic_id_generator/debug_collisions.py`
    *   **作用**: 检查 `song_semantic_ids.jsonl` 文件中是否存在不同歌曲被分配到相同语义ID的冲突情况，并生成报告。

4.  **G1d - 批量计算语义ID评估指标**:
    *   **命令**: `python src/semantic_id_generator/calculate_metrics.py`
    *   **作用**: 专门用于批量计算生成的语义ID的定量评估指标（如轮廓系数、CH指数、DB指数）。该脚本会进行数据抽样以优化内存和计算效率。

### **第二部分：【引擎一】T5生成模型 (用于候选生成)**

1.  **G2a - 语料生成**: `python src/generator/prepare_corpus.py`，生成 `(文本 -> 概念簇ID序列)` 格式的训练数据。
2.  **G2b - 模型训练**: `python src/generator/train_t5.py`，训练T5模型学会根据文本生成概念序列。

### **第三部分：【引擎二】Cross-Encoder相关性模型 (用于精准排序)**

1.  **R1 - 数据生成**: `python src/ranker/prepare_data.py`，生成 `(文本, 歌曲ID) -> 相关/不相关` 格式的标签数据。
2.  **R2 - 模型定义**: `src/ranker/ranker_model.py`，定义将拼接“文本”和“歌曲语义ID”作为输入的Cross-Encoder模型。
3.  **R3 - 模型训练**: `src/ranker/train_ranker.py`，在一个二分类任务上训练Cross-Encoder模型。

### **第四部分：最终流水线与演示**

1.  **F1 - 推理演示**: `python src/demo.py`，该脚本将串联两个引擎：
    *   调用**生成模型**，为输入文本产出候选集。
    *   调用**排序模型**，对候选集进行打分和排序。
    *   输出最终推荐列表。

### **附录**
#### **语义 id 评估指标说明**
1. 轮廓系数 (Silhouette Score)
   * 衡量标准：同时考虑了簇内紧密性和簇间分离度。对于每个样本，它计算该样本与其自身簇的平均距离，以及与最近邻簇的平均距离。
   * 取值范围：-1 到 +1。
   * 解释：
       * +1：表示样本与其自身簇非常匹配，且与相邻簇分离良好。
       * 0：表示样本位于两个簇的边界上。
       * -1：表示样本可能被分配到了错误的簇。
   * 整体评估：所有样本的平均轮廓系数。值越高越好。
   * 优点：直观，综合考虑了内聚和分离。
   * 缺点：计算成本较高，对于180万条数据，直接计算可能会非常慢。通常需要对数据进行抽样来计算。

2. Calinski-Harabasz 指数 (CH Index / 方差比准则)
   * 衡量标准：簇内离散度与簇间离散度的比值。
   * 取值范围：无上限，通常为正数。
   * 解释：值越高越好，表示簇的定义越清晰，分离度越高。
   * 优点：计算速度相对较快。
   * 缺点：倾向于选择密度高、球形的簇。

3. Davies-Bouldin 指数 (DB Index)
   * 衡量标准：衡量簇内散度与簇间距离的比率。
   * 取值范围：通常为正数。
   * 解释：值越低越好，表示簇内紧密性高，簇间分离度高。
   * 优点：计算速度相对较快。
   * 缺点：对噪声和异常值敏感。
