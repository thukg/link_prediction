import time

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from models import *
from utilw import data_generator


class Model:
    def __init__(self,config):
        self.model = None
        self.config = config
        self.sample = data_generator(self.config)
        self.entTotal = 0
        self.relTotal = 0
        self.init()

    # @jit
    def init(self):
        self.entTotal = self.sample.get_entTotal()
        self.config.entTotal = self.entTotal
        self.relTotal = self.sample.get_relTotal()
        self.config.relTotal = self.relTotal
        # self.sample.generate_train_batch()

        if self.config.model_type == 'transE':
            self.model = transE(self.config)
        elif self.config.model_type == 'convE':
            self.model = convE(self.config)
        elif self.config.model_type == 'DistMult':
            self.model = DistMult(self.config)
        elif self.config.model_type == "simplE":
            self.model = simplE(self.config)
        elif self.config.model_type == "rotatE":
            self.model = rotatE(self.config)
        else:
            self.model = DistMult(self.config)
        
    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        
    def my_collate(self, batch):
        pos = np.array([x[0] for x in batch])
        neg = np.array([x[1] for x in batch])
        neg = np.concatenate(neg)
        return torch.LongTensor(np.concatenate([pos, neg]))
    
    def save_embeddings(self, model_path, prefix):
        self.model.save_embedding(model_path, prefix)
    
    def train(self, cycle, gamma):
        np.random.seed(42)
        torch.manual_seed(42)
        optimizer = optim.Adam(self.model.parameters(),lr=self.config.learning_rate, weight_decay=self.config.weight_decay)  
    
        if self.config.cuda:
            print("cuda is available")
            torch.cuda.manual_seed(42)
            self.model.cuda()

        schedular = StepLR(optimizer=optimizer, step_size=cycle, gamma=gamma)

        print("before training")
        cnt = 1
        loss_cnt = 0
        for epoch in range(self.config.training_epoch):
            print("In epoch {}, training model...".format(epoch))
            data_iter = DataLoader(dataset=self.sample, batch_size=self.config.batchSize, collate_fn=self.my_collate,drop_last=False, shuffle=True)
            start_time = time.time()

            loss_mean = 0
            self.model.train()
            for batch in data_iter:
                optimizer.zero_grad()
                if self.config.cuda:
                    batch = batch.cuda()
                score_pos, score_neg = self.model(batch)
                loss = self.model.loss(score_pos, score_neg)

                loss_mean += loss.item()
                loss_cnt += loss.item()
                if cnt % 100 == 0:
                    print("loss:", loss_cnt / 100)
                    loss_cnt = 0
                cnt += 1
                loss.backward()
                optimizer.step()
            print("loss_sum:",loss_mean)

            end_time = time.time()
            print("epoch{} consumes: {}".format(epoch, end_time - start_time))

            if self.config.model_type == "simplE":
                print("saving embeddings...")
                self.save_embeddings("./embeddings/", str(epoch))
                if self.config.cuda:
                    self.model.cuda()
            elif self.config.model_type == "convE" or self.config.model_type == "rotatE":
                print("saving model...")
                torch.save(self.model.cpu().state_dict(), "./train_model/{}_step_{}_epoch_{}".format(self.config.model_type, cnt, epoch))
                self.save_embeddings("./embeddings/", str(epoch))
                if self.config.cuda:
                    self.model.cuda()
            elif epoch > 100 or epoch % 20 == 0:
                self.save_embeddings("./embeddings/", str(epoch))
                if self.config.cuda:
                    self.model.cuda()
            schedular.step(epoch)


# if __name__ == "__main__":
#     config = Config()
#     config.batchSize = 256
#     config.gamma = 3.0
#     config.adversial_alpha = 0.5
#     config.embedding_dim = 250
#     config.learning_rate = 0.001
#
#     config.data_path = "./benchmarks/WN18/"
#
#     config.model_type = "rotatE"
#     md = Model(config)
#     # md.train()
#
#     md.load_model("./train_model/rotatE_step_7743_epoch_13")
#     md.save_embeddings("./embeddings/","rotatE_test")