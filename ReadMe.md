对于输入，要统一格式： 有annotation的，存放在obsm['spatial']， 清洗数据，过滤掉标签为NA的点， 确保表达矩阵X存储的是原始技术Raw Counts. 标准化的.h5ad文件存放在data/raw_h5ad/目录下。


进行数据预处理：
./run_preprocess.sh E1S1 DLPFC


现在预处理部分：/home/yawen/project/GreS/run_preprocess.sh

1. 数据预处理
运行预处理文件，生成对应的可输入的格式：/home/yawen/project/GreS/preprocess/DLPFC_generate_data.py

2. 生成语义embedding
筛选语义embedding和GRNS. 对语义embeddings进行更新： /home/yawen/project/GreS/build_programST_assets.py

3. 生成spot embedding
python grn_generate_spot_embedding.py --dataset_id 151507 --gene_emb_source grn --topk 30

4. 构建基于基因的特征邻接图（fadj——spotemb）
python build_fadj_from_geneemb.py --dataset_id 151507 --source grn --k 14

5. 模型训练
python train.py \
    --dataset_id 151672 \
    --config_name DLPFC \
    --llm_emb_dir data/npys_grn/ \
    --run_name test