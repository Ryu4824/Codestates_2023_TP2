from models.KNN import KNN
from models.MF import MF
from utils.Dataloader import load_ratings,load_movies

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from gensim.models import Word2Vec
import joblib
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

# 각 모델별 출력 값에 근거, 순서와 모델의 가중에 따라 점수를 다시 매기고 높은 점수대로 출력
class Base_Hybrid:
    def __init__(self, knn_model, mf_model, alpha):
        self.knn_model = knn_model
        self.mf_model = mf_model
        self.alpha = alpha  # 가중치 조절 파라미터

    def predict(self, USER_ID, TOP_NUM):
        # base 모델 결과 가져오기
        knn_r = self.knn_model.predict(USER_ID, TOP_NUM)
        mf_r = self.mf_model.predict(USER_ID, TOP_NUM)

        # 각 모델의 결과에 대해 순위를 매김
        knn_rank = {idx: 1 / (i + 1) for i, idx in enumerate(knn_r)}
        mf_rank = {idx: 1 / (i + 1) for i, idx in enumerate(mf_r)}
        
        # 가중치(alpha)를 부여하여 두 결과 병합
        combined_scores = {}
        for idx, rank in knn_rank.items():
            combined_scores[idx] = combined_scores.get(idx, 0) + (self.alpha * rank)
        for idx, rank in mf_rank.items():
            combined_scores[idx] = combined_scores.get(idx, 0) + ((1 - self.alpha) * rank)

        # 결과 정렬
        sorted_combined = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        # 상위 TOP_NUM 아이템 추출
        top_indices = [idx for idx, _ in sorted_combined[:TOP_NUM]]
        return top_indices, knn_r, mf_r
