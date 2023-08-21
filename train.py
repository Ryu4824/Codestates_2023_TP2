import sys
from pathlib import Path
import os

#경로 설정
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils import Dataloader
from models import MF, KNN
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--vector_size', type=int, default=100)
    parser.add_argument('--pretrained', type=str, default='glove-twitter-100')
    parser.add_argument('--knn', action=argparse.BooleanOptionalAction)
    parser.set_defaults(knn=True)
    parser.add_argument('--mf', action=argparse.BooleanOptionalAction)
    parser.set_defaults(mf=True)
    parser.add_argument('-k', type=int, default=200)
    parser.add_argument('-e', '--epochs',type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=512)
    args = parser.parse_args()
    return args

args = parse_args()

#데이터 폴더 경로
DIR_PATH = "./datasets/"

#데이터 호출
users_df = Dataloader.load_users(DIR_PATH)
ratings_df = Dataloader.load_ratings(DIR_PATH)
movies_df = Dataloader.load_movies(DIR_PATH)

if args.mf:
    MF.train(users_df=users_df,
             movies_df=movies_df,
             ratings_df=ratings_df,
             K=args.k,
             epochs=args.epochs,
             batch_size=args.batch_size
            )

if args.knn:
    KNN.train(movies_df=movies_df,
              vector_size = args.vector_size,
              pretrained= args.pretrained
             )