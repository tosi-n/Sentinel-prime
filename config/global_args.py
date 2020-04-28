import torch


global_args = {
    "model_name": "aen_bert",
    "dataset": "logically",
    "trainset": "./data/train.csv",
    "testset": "./data/dev.csv",
    "state_dict_path": "state_dict/aen_bert_logically_val_temp",
    "optimizer": torch.optim.Adam,
    "initializer": torch.nn.init.xavier_uniform_,
    "learning_rate": 2e-5,
    "dropout": 0.1,
    "l2reg": 0.01,
    "num_epoch": 10,
    "batch_size": 32,
    "max_seq_length": 80,
    "pretrained_bert_name": "bert-base-uncased",
    "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    "seed": 3333,
    "cross_val_fold": 10,
    "log_step": 10,
    "embed_dim": 300,
    "hidden_dim": 300,
    "bert_dim": 768,
    "polarities_dim": 3,
}


