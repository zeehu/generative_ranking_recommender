** 生成模型训练记录 **
- 歌曲向量
    - 当前可用版本：outputs/sg1_vs512_w50_ep20_song_vectors.csv
        - 使用skip-gram模型，512维向量，上下文窗口50，epoch 20，最低次数10
        - 对比256维向量、cbow模型效果更好

- 语义id
    - 当前可用版本：outputs/semantic_id_128_128_128
        - 使用simplified_semantic_id_generator.py 训练，三层[128, 128, 128]，70w唯一id，最多121首个分配到同一id序列，采样case效果可以，还有优化空间
        - CH：2.6393，DB：0.6041，Silhuette：0.0005
    - 在训版本：new_generative_ranking_recommender/outputs/semantic_id
        - 使用simplified_semantic_id_generator.py 训练，三层[128, 128, 256]
    - 测试版本：new_v2_generative_ranking_recommender/src/semantic_id_generator/train_semantic_ids.py
        - 代码重构，训练中断层恢复，训练和预测功能，动态计算batch_size和iter_limit

- T5模型
    - 语料生成
       - prepare_corpus.py当前版本修复语义id未分层问题，为可用版本
    - 模型训练
       - train_t5.py 当前版本修复语义id未分层问题，版本待测试