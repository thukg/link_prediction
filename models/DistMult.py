import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .initModel import initModel
from torch.autograd import Variable



class DistMult(initModel):
    def __init__(self, config):
        super(DistMult, self).__init__(config)
        self.entEmbedding = nn.Embedding(self.config.entTotal, self.config.embedding_dim)
        self.relEmbedding = nn.Embedding(self.config.relTotal, self.config.embedding_dim)
        self.criterion = nn.MarginRankingLoss(self.config.margin, reduction="sum")
        self.criterion = nn.Softplus()
        self.init()

    def init(self):
        nn.init.xavier_uniform_(self.entEmbedding.weight.data)
        nn.init.xavier_uniform_(self.relEmbedding.weight.data)

    def loss(self, score_pos, score_neg):
        y = torch.Tensor([-1])
        if self.config.cuda:
            y = y.cuda()

        loss1 = torch.mean(self.criterion(y * score_pos) + self.criterion(score_neg))
        # loss1 = self.criterion(score_neg, score_pos, y)
        return loss1

    def pos_neg_score(self,score):
        pos_score = score[:self.batchSize].view(self.batchSize)

        neg_score = score[self.batchSize:].view(self.batchSize, -1)
        neg_score = neg_score.mean(dim=1)
        return pos_score, neg_score

    def forward(self, batch):
        self.batchSize = batch.shape[0]//(1 + self.config.negativeSize * 2)
        h = batch[:,0]
        t = batch[:,1]
        r = batch[:,2]
        emb_h = self.entEmbedding(h)
        emb_r = self.relEmbedding(r)
        emb_t = self.entEmbedding(t)

        score = torch.sum(emb_h * emb_r * emb_t, -1)
        score_pos, score_neg = self.pos_neg_score(score)

        return score_pos, score_neg

    def predict(self, h, r, t):
        emb_h = self.entEmbedding(h)
        emb_r = self.relEmbedding(r)
        emb_t = self.entEmbedding(t)

        score = torch.sum(emb_h * emb_r * emb_t, -1)
        return score

    def renormalize(self):
        self.entEmbedding.weight.data = F.normalize(self.entEmbedding.weight.data, dim=1)
        self.relEmbedding.weight.data = F.normalize(self.relEmbedding.weight.data, dim=1)
    

    