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
├── src/                      # 源代码
│   ├── common/               # 通用工具 (如日志、随机种子)
│   ├── generator/            # 【引擎一】生成模型的全部代码
│   └── ranker/               # 【引擎二】排序模型的代码
├── .gitignore                # Git忽略文件配置
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

### **第一部分：【引擎一】T5生成模型 (用于候选生成)**

1.  **G1 - 向量量化**: 在 `src/generator/` 中创建 `train_rq_kmeans.py`，将歌曲向量量化为“概念簇ID”。
2.  **G2 - 语料生成**: 在 `src/generator/` 中创建 `prepare_corpus.py`，生成 `(文本 -> 概念簇ID序列)` 格式的训练数据。
3.  **G3 - 模型训练**: 在 `src/generator/` 中创建 `train_t5.py`，训练T5模型学会根据文本生成概念序列。

### **第二部分：【引擎二】Cross-Encoder相关性模型 (用于精准排序)**

1.  **R1 - 数据生成**: 在 `src/ranker/` 中创建 `prepare_data.py`，生成 `(文本, 歌曲ID) -> 相关/不相关` 格式的标签数据。
2.  **R2 - 模型训练**: 在 `src/ranker/` 中创建 `train_ranker.py`，在一个二分类任务上训练Cross-Encoder模型。

### **第三部分：最终流水线与演示**

1.  **F1 - 推理演示**: 在 `src/` 目录下创建 `demo.py`，该脚本将串联两个引擎：
    *   调用**生成模型**，为输入文本产出候选集。
    *   调用**排序模型**，对候选集进行打分和排序。
    *   输出最终推荐列表。
