from utilw import Config
from train import Model
import torch
import torch.optim as optim
import argparse
import os
import torch.multiprocessing as mp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_params", action="store_true", default=False, help="set hyperparameters on your own")
    parser.add_argument("--no-cuda", action="store_true", default=False,
                        help="Disable CUDA trainning")
    parser.add_argument("--seed", type=int, default=42, help="Random Seed.")
    parser.add_argument("--epoches", type=int, default=200, help="Number of epoches to train")
    parser.add_argument("--dataset", type=str, default="WN18", help="Select Training dataset.")
    parser.add_argument("--model", type=str, default='rotatE', help="Select Model.")
    parser.add_argument("--learning_rate",type=float, default=0.001, help="set learning rate")
    parser.add_argument("--load_model", type=str, default=None, help="load exited model")
    parser.add_argument("--loss_function",type=str, default="marginranking",
                        help="use \"marginranking\" or \"softplus\" as loss function")
    parser.add_argument("--threads", type=int, default=1,
                        help="threads to use")
    parser.add_argument("--adversial_sampling", action="store_true", default=False,
                        help="adopt negative adversial sampling")
    parser.add_argument("--adversial_alpha", type=float, default=1.0,
                        help="negative adversial-sampling temperature")
    parser.add_argument("--weight_decay",type=float, default=0,
                        help="regularization_rate")
    parser.add_argument("--negative_size",type=int, default=1,
                        help="negative samples for each entity")
    parser.add_argument("--batchsize", type=int, default=256,
                        help="set batchsize")
    parser.add_argument("--margin", type=float, default=1.0,
                        help="set margin")
    parser.add_argument("--embedding_dim",type=int,default=200,
                        help="embedding vector dimension")

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    config = Config()
    config.cuda = args.cuda
    config.training_epoch = args.epoches
    config.data_path = os.path.join("./benchmarks/",args.dataset)
    config.model_type = args.model
    config.learning_rate = args.learning_rate
    config.load_model = args.load_model
    config.loss_function = args.loss_function
    config.threads = args.threads
    config.adversial_sampling = args.adversial_sampling
    config.adversial_alpha = args.adversial_alpha
    config.weight_decay = args.weight_decay
    config.negativeSize = args.negative_size
    config.embedding_dim = args.embedding_dim
    config.batchsize = args.batchsize

    if not args.set_params:
        if config.model_type == "DistMult":
            config.batchSize = 256
            config.weight_decay = 0.00001
            config.embedding_dim = 200
            config.learning_rate = 0.001
            config.training_epoch = 200
        elif config.model_type == "simplE":
            config.batchSize = 100
            # config.weight_decay = 0.001
            config.embedding_dim = 200
            config.learning_rate = 0.001
            config.training_epoch = 30
        elif config.model_type == "convE":
            config.batchSize = 128
            config.learning_rate = 0.001
            config.embedding_dim = 200
            config.training_epoch = 100
        elif config.model_type == "transE":
            config.batchSize = 256
            config.embedding_dim = 100
            config.learning_rate = 0.001
            config.margin = 1
            config.training_epoch = 100
        elif config.model_type == "rotatE":
            config.batchSize = 256
            config.gamma = 3.0
            config.adversial_alpha = 0.5
            config.embedding_dim = 250
            config.learning_rate = 0.001
            config.training_epoch = 100


    md = Model(config)
    md.train(50, 0.2)
