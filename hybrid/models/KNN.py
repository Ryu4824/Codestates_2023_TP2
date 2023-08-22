import joblib
from utils.Dataloader import load_ratings,load_movies
import sys
sys.path.append('/Users/jain3/OneDrive/Desktop/AI/Team_project_2/KNN')  # 프로젝트 루트 경로를 추가해주세요.
from pathlib import Path
import os
import argparse
from sklearn.metrics.pairwise import cosine_similarity

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
        model_file_path = '/Users/jain3/OneDrive/Desktop/AI/Team_project_2/hybrid/models/knn.joblib'  # 실제 파일 경로로 수정해야 합니다.
        self.model = joblib.load(model_file_path)

    #내용 추가
    def calculate_movie_similarity(self, user_id):
        """
        사용자의 시청한 영화들의 벡터 간 유사도를 계산합니다.
        
        Args:
            user_id (int): 사용자의 ID
        
        Returns:
            dict: 영화 ID를 키로 하고 유사도를 값으로 하는 딕셔너리
        """
        #data load
        cbf_data = joblib.load('models/cbf_data.joblib')
        ratings_df = load_ratings('datasets/')

        # 사용자가 시청한 영화 목록 호출
        movie_list = ratings_df[ratings_df['userId'] == user_id]['movieId'].tolist()

        # 유저가 시청한 영화들의 벡터 가져오기
        movie_vectors = [cbf_data[m] for m in movie_list]

        # 유저의 시청 영화 벡터들 간의 코사인 유사도 계산
        user_movie_similarity = {}
        for i, movie_id in enumerate(movie_list):
            similarities = cosine_similarity([movie_vectors[i]], cbf_data)
            user_movie_similarity[movie_id] = similarities[0]

        return user_movie_similarity  
    
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
        
        #예측(성능 향상시 n_neighbor 수정)
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
    knn = NearestNeighbors(metric='cosine') #metric 수정 -> default 값은 유클리드/맨하탄 거리
    knn.fit(cbf_data)
    joblib.dump(knn, "./models/knn.joblib")
    print("---knn Done.---")
    
if __name__ == '__main__':
    opt = parse_opt()
    knn = KNN(path='/Users/jain3/OneDrive/Desktop/AI/Team_project_2/hybrid/')  # 실제 경로를 정확하게 지정해야 합니다.
    print(knn.predict(opt.user, opt.num))