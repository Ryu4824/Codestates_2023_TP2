from utils.Dataloader import load_ratings
import pandas as pd
import numpy as np

ratings_df = load_ratings('datasets')

def popular(top=5):
    # 아이템별 리뷰 수 계산
    item_review_counts = ratings_df['movieId'].value_counts()

    # 아이템별 평균 평점 계산
    item_avg_ratings = ratings_df.groupby('movieId')['rating'].mean()

    # 스코어 계산 함수 정의
    def calculate_score(average_rating, review_count):
        return average_rating - (average_rating - 0.5) * 2 ** (-np.log10(review_count))

    # 아이템 스코어 계산
    item_scores = {}
    for movie_id, avg_rating in item_avg_ratings.items():
        review_count = item_review_counts.get(movie_id, 1)  # 1 to avoid division by zero
        score = calculate_score(avg_rating, review_count)
        item_scores[movie_id] = score

    # 스코어에 따라 아이템 정렬
    sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)

    top_items = sorted_items[:top]
    top_item_ids = [item[0] for item in top_items]

    return top_item_ids