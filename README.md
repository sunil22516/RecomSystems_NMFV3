# recommendation-systems

This project implements a recommendation system using both **Matrix Factorization (SVD/ALS)** and a deep learning approach, **Neural Collaborative Filtering (NeuMF)**, to predict user-item interactions. The dataset is preprocessed by binarizing ratings, and the models are trained to learn user and item representations for ranking tasks. Additionally, an LLM-based method is included to generate ratings as part of the project requirements.

The performance of the models is evaluated using **Hit Rate** and **NDCG**, focusing on ranking quality rather than raw prediction accuracy. The pipeline includes data preprocessing, model training, prediction generation, and ranking-based evaluation, with results presented in tabular form for comparison across different approaches.
