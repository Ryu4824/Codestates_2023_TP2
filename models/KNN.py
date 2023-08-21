import joblib
from utils.Dataloader import load_ratings,load_movies
import sys
from pathlib import Path
import os
import argparse

#콘텐츠 기반 필터링용 패키지
from gensim.models import Word2Vec
from utils.Preprocessing import tokenizer, vectorizer
import gensim.downloader as api
from sklearn.neighbors import NearestNeighbors
import numpy as np
import joblib

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--user', type=int, default=1)
    parser.add_argument('-n', '--num', type=int, default=5)

    opt = parser.parse_args()
    return opt

class KNN():
    def __init__(self, path):
        self.model = joblib.load(path)
        
    def predict(self, userid, n):
        """
        콘텐츠 정보를 기반으로 영화를 추천합니다.
        
        Args:
            userid (int) : 추천 대상 유저의 id.
            n (int) : 출력하는 추천 영화의 수. n을 입력하면 유저가 지금까지 평점을 남긴 데이터를 기반으로 비슷한 상위 n개의 영화를 추천합니다.
        """
        #data load
        cbf_data = joblib.load('models/cbf_data.joblib')
        ratings_df = load_ratings('datasets/')
        
        #유저가 시청했던 영화 목록 호출
        movie_list = ratings_df[ratings_df['userId']==userid]['movieId'].tolist()
        
        #입력 벡터 생성
        #입력 벡터는 유저가 본 영화의 모든 벡터의 평균을 사용
        m_vector = 0
        for m in movie_list:
            m_vector += cbf_data[m]
        
        #예측    
        return self.model.kneighbors(m_vector.reshape((1,-1)), n_neighbors=n)[1][0]

def train(movies_df=None, vector_size=100, pretrained = 'glove-twitter-100'):
    if movies_df is None:
        movies_df = load_movies('datasets/')
    
    print("---Tokenizing...---")
    tokens = movies_df['title'].apply(tokenizer)
    print("Tokenizing Complete.")

    print("---w2v Training...---")
    w2v = Word2Vec(sentences=tokens, vector_size = vector_size, window = 2, min_count = 1, workers = 4, sg= 0)
    w2v.save("./models/word2vec.model")
    print(w2v.wv.vectors.shape)
    print("w2v Training Complete.")

    wv = w2v.wv

    vectors = tokens.apply(vectorizer)

    print("---pre-trained w2v loading...---")
    #사전 훈련된 w2v 가중치 호출
    wv2 = api.load(f"{pretrained}")
    print("loading Complete.")

    def gen2vec(sentence):
        vector = 0
        for g in sentence.split('|'):
            if g.lower() == "children's":
                g = "children"
            elif g.lower() == "film-noir":
                g = "noir"

            vector += wv2[g.lower()]
        return vector

    g_vector = movies_df['genres'].apply(gen2vec)
    
    #훈련 데이터 생성
    cbf_vectors = ((vectors.to_numpy() + g_vector.to_numpy()) / 2).tolist()
    cbf_data = np.zeros((movies_df['movieId'].max()+1, 100))
    
    for idx, vec in zip(movies_df['movieId'], cbf_vectors):
        cbf_data[idx] = vec
    
    joblib.dump(cbf_data, "./models/cbf_data.joblib")

    print("---knn Training...---")
    knn = NearestNeighbors()
    knn.fit(cbf_data)
    joblib.dump(knn, "./models/knn.joblib")
    print("---knn Done.---")
    
if __name__ == '__main__':
    opt = parse_opt()
    knn = KNN()
    print(knn.predict(opt.user, opt.num))