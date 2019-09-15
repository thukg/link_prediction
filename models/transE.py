import torch.nn as nn
# import Config
import torch
from .initModel import initModel
import torch.nn.functional as F
from torch.autograd import Variable

class transE(initModel):
    def __init__(self, config):
        super(transE,self).__init__(config)
        self.entEmbedding = nn.Embedding(self.config.entTotal, self.config.embedding_dim)
        self.relEmbedding = nn.Embedding(self.config.relTotal, self.config.embedding_dim)
        # self.criterion = nn.MarginRankingLoss(self.config.margin, reduction="sum")
        self.criterion = nn.Softplus()
        self.batchSize = self.config.batchSize
        self.init()

    def init(self):
        nn.init.xavier_uniform_(self.entEmbedding.weight.data)
        nn.init.xavier_uniform_(self.relEmbedding.weight.data)

    def pos_neg_score(self,score):
        pos_score = score[:self.batchSize]
        neg_score = score[self.batchSize:].view(self.batchSize, -1)
        neg_score = neg_score.mean(dim=1)
        return pos_score, neg_score

    def loss(self, score_pos, score_neg):
        y = Variable(torch.Tensor([-1]))
        if self.config.cuda:
            y = y.cuda()
        # loss1 = self.criterion(score_neg, score_pos, y)
        loss1 = torch.sum(self.criterion(y * score_pos) + self.criterion(score_neg))
        return loss1

    def forward(self, batch):
        self.batchSize = batch.shape[0]//(1 + self.config.negativeSize * 2)
        h = batch[:,0]
        t = batch[:,1]
        r = batch[:,2]

        emb_h = self.entEmbedding(h)
        emb_t = self.entEmbedding(t)
        emb_r = self.relEmbedding(r)

        score = torch.norm(emb_h + emb_r - emb_t, dim=1, p=1)
        score_pos, score_neg = self.pos_neg_score(score)
        return score_pos, score_neg
    
    def predict(self, h, r, t):
        emb_h = F.normalize(self.entEmbedding(h), 1)
        emb_r = F.normalize(self.relEmbedding(r), 1)
        emb_t = F.normalize(self.entEmbedding(t), 1)

        score = torch.abs(emb_h + emb_r - emb_t)
        score = torch.sum(score, -1)
        return score
