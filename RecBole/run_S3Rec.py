from recbole.data import create_dataset, data_preparation
from recbole.config import Config
from recbole.model.sequential_recommender.s3rec import S3Rec
from recbole.trainer import Trainer
from recbole.utils.enum_type import ModelType
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cuda')

# Define config for S3Rec
config_dict = {
    'data_path': './datasets/',
    'checkpoint_dir': './checkpoints',
    'MODEL_TYPE': ModelType.SEQUENTIAL,
    'model': 'S3Rec',  # Specify the model as S3Rec
    'learning_rate': 0.001,
    'epochs': 20,
    'eval_step': 1, 
    'stopping_step': 5,
    'batch_size': 1024,
    'state': 'INFO',
    'load_col': {
        'inter': ['user_id', 'item_id', 'rating', 'timestamp']
    },
    'train_batch_size': 1024,
    'eval_batch_size': 2048,
    'USER_ID_FIELD': 'user_id',
    'ITEM_ID_FIELD': 'item_id',
    'RATING_FIELD': 'rating',
    'TIME_FIELD': 'timestamp',
    'NEG_PREFIX': 'neg_',
    'eval_setting': 'RO_RS,full',
    'topk': 10,
    'loss_type': 'BPR',
    'valid_metric': 'Recall@10',
    # 'hidden_size': 64,
    # 'inner_size': 256,
    # 'n_layers': 2,
    # 'n_heads': 2,
    # 'hidden_dropout_prob': 0.5,
    # 'attn_dropout_prob': 0.5,
    # 'hidden_act': 'gelu', #['gelu', 'relu', 'swish', 'tanh', 'sigmoid']
    # 'layer_norm_eps': 1e-12,
    # 'initializer_range': 0.02,
    'metric' : ['Recall@10', 'MRR@10', 'NDCG@10', 'Hit@10', 'Precision@10'],
    'dataset': 'movies',
    
}

config = Config(config_dict=config_dict)

# Prepare dataset
dataset = create_dataset(config)
train_data, valid_data, test_data = data_preparation(config, dataset)

# Initialize the model
model = S3Rec(config, dataset) 

# Move the model to the desired device
model = model.to(device)

# Train the model
trainer = Trainer(config, model)
trainer.fit(train_data, valid_data, saved=True, show_progress=True)