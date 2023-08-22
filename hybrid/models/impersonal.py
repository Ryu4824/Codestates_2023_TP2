from utils.Dataloader import load_ratings

ratings_df = load_ratings('datasets')

def popular(top=5):
    return ratings_df.groupby('movieId').count()['userId'].sort_values(ascending=False).index.tolist()[:top]