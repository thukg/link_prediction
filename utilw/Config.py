import os

class Config:
    dropout = 0.2
    batchSize = 32
    learning_rate = 0.1
    embedding_dim = 100
    p_norm = 1
    margin = 1.0
    use_bias = True
    training_epoch = 200
    hits_at_k =  10
    data_path = "./benchmarks/FB15K/"
    model_type = 'DistMult'
    cuda = True
    fastmode = True
    loss_function = "marginranking"
    gamma = 6.0
    negativeSize = 1
    threads = 1
    fast_size = 2000
    adversial_sampling = False
    adversial_alpha = 0.5
    weight_decay = 0.0001
    entTotal = 0
    relTotal = 0