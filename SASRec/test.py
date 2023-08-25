import tensorflow as tf
from model import Model
import argparse
from util import *
from main import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=5, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--time_span', default=256, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.00005, type=float)

args = parser.parse_args()

# 예시 데이터를 생성하거나 불러옵니다.
ratings_df = load_ratings('datasets/')  # 사용자별 영화 평점 데이터
movies_df = load_movies('datasets/')

# Placeholders (assuming they are defined in your model. Adjust if different)
dataset = data_partition(args.dataset)
[_, _, _, usernum, itemnum, timenum] = dataset

model = Model(usernum, itemnum, timenum, args)

if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    
    with tf.Session(config=config) as sess:
        # Create a saver object
        saver = tf.train.Saver()

        # Load the checkpoint
        try:
            saver.restore(sess, "./SASRec/checkpoints/model.ckpt")
            print("Model restored from the checkpoint.")

            # Example data loading and prediction
            userId = 4
            # 사용자의 영화 시청 기록을 가져옵니다.
            user_movie_history = ratings_df[ratings_df['userId'] == userId]

            # 영화 시청 시간 순으로 정렬합니다.
            user_movie_history = user_movie_history.sort_values(by='timestamp', ascending=True)  # 시간순으로 정렬

            # 최근 50개의 영화를 선택하고, 뒤쪽에 0을 패딩합니다.
            if len(user_movie_history) < args.maxlen:
                padding = [0] * (args.maxlen - len(user_movie_history))
                seq = padding + user_movie_history['movieId'].tolist()
                watch_times = padding + user_movie_history['timestamp'].tolist()
            else:
                recent_movie_history = user_movie_history.tail(args.maxlen)
                seq = recent_movie_history['movieId'].tolist()
                watch_times = recent_movie_history['timestamp'].tolist()

            # 시퀀스와 시청 시간을 numpy 배열로 변환합니다.
            seq = np.array(seq)
            watch_times = np.array(watch_times)

            # 시청 시간을 이용하여 time_matrix 생성
            time_matrix = computeRePos(watch_times, args.time_span)
            time_matrix = time_matrix[-args.maxlen:]

            # time_matrix에 대한 패딩 처리
            if len(time_matrix) < args.maxlen:
                padding = np.zeros((args.maxlen - len(time_matrix), args.maxlen))
                time_matrix = np.concatenate((padding, time_matrix), axis=0)
            
            all_movie_ids = ratings_df['movieId'].unique()  # 중복 없는 영화 아이디만 선택
            #ratings데이터에 movieId 값은 중복을 포함해서 1000209개가 있는데
            #어떻게 데이터를 제외했는지 모름(단순히 중복을 제외는 아닌듯함 중복만 제외면 3706개이기때문)
            #현재 movieId가 3416(itemnum)까지만 가능
            #이러면 다른 영화를 추천에 제한됨
            
            item_idx = all_movie_ids
            print(all_movie_ids)
            print([userId])
            print("--------------------------------")
            print([seq],seq.shape)
            print("--------------------------------")
            print([time_matrix],time_matrix.shape)
            print("--------------------------------")
            print(item_idx,len(item_idx))

            predictions = model.predict(sess, [userId], [seq], [time_matrix],item_idx)
            # predictions = sess.run()
            print(f"예측값은?{predictions}")
            # 예측값으로부터 상위 10개 영화 인덱스 추출
            top_10_indices = np.argsort(predictions)[0][-10:][::-1]

            # 상위 10개 영화 인덱스에 해당하는 예측값
            top_10_predictions = predictions[0][top_10_indices]
            
            # 영화 제목으로 매핑하여 결과 출력
            for idx in top_10_indices:
                movie_title = movies_df[movies_df['movieId'] == idx]['title'].item()
                prediction = predictions[0][idx]
                print(f"영화 제목: {movie_title}, 예측 평점: {prediction}")

        except Exception as e:
            print(f"Error restoring from the checkpoint. Error: {e}")