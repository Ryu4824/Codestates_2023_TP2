from tensorflow.keras.models import load_model
import tensorflow as tf
from utils.Dataloader import load_ratings

#협업 필터링용 패키지
from utils import Dataloader
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Embedding, Flatten
from sklearn.model_selection import train_test_split

class MF():
    """
    Matrix Factorization를 활용한 협업 필터링 기반 추천 모델입니다.
    """
    def __init__(self, path):
        self.model = load_model(path)
        
    def predict(self, userid, top = 10):
        """
        유저 정보를 기반으로 영화를 추천합니다.
        
        Args:
            userid (int) : 추천 대상 유저의 id.
            top (int) : 출력하는 추천 영화의 수. n을 입력하면 유저가 만족할 것으로 생각되는 상위 n개의 영화를 추천합니다.
        """
        user = self.model.get_layer('u_emb')(userid)[tf.newaxis,:]
        items = tf.transpose(self.model.get_layer('i_emb').weights[0], [1,0])
        mm = tf.matmul(user,items)
        
        return tf.argsort(mm, direction='DESCENDING').numpy().tolist()[0][:top]
    
#모델 기반 협업 필터링(Matrix Factorization)
#모델 파이프라인 생성
def mf_model(user_dim, item_dim, K):
    user = Input((1,))
    item = Input((1,))
    u_emb = Embedding(user_dim, K, name='u_emb')(user)
    i_emb = Embedding(item_dim, K, name='i_emb')(item)

    R = tf.keras.layers.dot([u_emb, i_emb], axes=2)
    R = Flatten()(R)
    return Model(inputs=[user, item], outputs=R)

def train(users_df=None, movies_df=None, ratings_df=None, K=200, epochs=1, batch_size = 512, validation_split=0.2):
    
    if users_df is None:
        users_df = Dataloader.load_users('datasets')
    if movies_df is None:
        movies_df = Dataloader.load_movies('datasets')
    if ratings_df is None:
        ratings_df = Dataloader.load_ratings('datasets')
        
    USER_DIM = users_df['userId'].max()+1
    ITEM_DIM = movies_df['movieId'].max()+1

    #data split
    train, val = train_test_split(ratings_df, test_size=0.2)
    x_train = [train['userId'], train['movieId']]
    y_train = train['rating']

    #모델 선언
    mf = mf_model(USER_DIM, ITEM_DIM, K)
    mf.compile(loss="mse",
               optimizer="adam"
              )

    #모델 훈련
    mf.fit(x_train, y_train, 
           epochs=epochs,
           batch_size = batch_size,
           validation_split=validation_split)

    #모델 저장
    print("---Saving model---")
    mf.save('./models/mf.h5')
    print('Save Complete.')