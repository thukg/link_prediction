import torch.nn as nn
import torch 
from .initModel import initModel
import torch.nn.functional as F
from torch.autograd import Variable
import codecs
import os
import json

class simplE(initModel):
    def __init__(self, config):
        super(simplE, self).__init__(config)
        self.entHeadEmbedding = nn.Embedding(self.config.entTotal, self.config.embedding_dim)
        self.entTailEmbedding = nn.Embedding(self.config.entTotal, self.config.embedding_dim)
        self.relEmbedding = nn.Embedding(self.config.relTotal, self.config.embedding_dim)
        self.relInverseEmbedding = nn.Embedding(self.config.relTotal, self.config.embedding_dim)
        self.criterion = nn.Softplus()
        self.batchSize = self.config.batchSize
        self.init()

    def init(self):
        nn.init.xavier_uniform_(self.entHeadEmbedding.weight.data)
        nn.init.xavier_uniform_(self.entTailEmbedding.weight.data)
        nn.init.xavier_uniform_(self.relEmbedding.weight.data)
        nn.init.xavier_uniform_(self.relInverseEmbedding.weight.data)

    def loss(self, score_pos, score_neg):
        y = Variable(torch.Tensor([-1]))
        if self.config.cuda:
            y = y.cuda()
        #softplus
        loss1 = torch.sum(self.criterion(-score_pos) + self.criterion(score_neg))
        return loss1

    def pos_neg_score(self,score):
        pos_score = score[:self.batchSize]
        neg_score = score[self.batchSize:].view(self.batchSize, -1)
        neg_score = torch.mean(neg_score,dim=1)

        pos_score = torch.clamp(pos_score, min=-20, max=20)
        neg_score = torch.clamp(neg_score, min=-20, max=20)
        return pos_score, neg_score    

    def forward(self, batch):
        self.batchSize = batch.shape[0]//(1 + self.config.negativeSize * 2)

        h = batch[:, 0]
        t = batch[:, 1]
        r = batch[:, 2]
        emb_h_as_h = self.entHeadEmbedding(h)
        emb_t_as_t = self.entTailEmbedding(t)
        emb_r = self.relEmbedding(r)
        emb_h_as_t = self.entTailEmbedding(h)
        emb_t_as_h = self.entHeadEmbedding(t)
        emb_r_inv = self.relInverseEmbedding(r)

        score = torch.sum((emb_h_as_h * emb_r * emb_t_as_t + emb_h_as_t * emb_r_inv * emb_t_as_h)/2, -1)
        score = self.pos_neg_score(score)
        return score
    
    def predict(self, h, r, t):
        emb_h_as_h = self.entHeadEmbedding(h)
        emb_t_as_t = self.entHeadEmbedding(t)
        emb_r = self.relEmbedding(r)
        emb_h_as_t = self.entTailEmbedding(h)
        emb_t_as_h = self.entHeadEmbedding(t)
        emb_r_inv = self.relInverseEmbedding(r)

        score = torch.sum(1/2 * (emb_h_as_h * emb_r * emb_t_as_t + emb_h_as_t * emb_r_inv * emb_t_as_h), -1)
        score = torch.clamp(score, min=-20, max=20)
        return score
    
    def save_embedding(self, emb_path, prefix):
        ent_head_path = os.path.join(emb_path, "simplE_head_entity{}.embedding".format(prefix))
        ent_tail_path = os.path.join(emb_path, "simplE_tail_entity{}.embedding".format(prefix))
        rel_path = os.path.join(emb_path, "simplE_rel{}.embedding".format(prefix))
        rel_rev_path = os.path.join(emb_path, "simplE_rel_rev{}.embedding".format(prefix))
        with codecs.open(ent_head_path, "w") as f:
            json.dump(self.entHeadEmbedding.cpu().weight.data.numpy().tolist(), f)
        with codecs.open(ent_tail_path, "w") as f:
            json.dump(self.entTailEmbedding.cpu().weight.data.numpy().tolist(), f)
        with codecs.open(rel_path, "w") as f:
            json.dump(self.relEmbedding.cpu().weight.data.numpy().tolist(), f)
        with codecs.open(rel_rev_path, "w") as f:
            json.dump(self.relInverseEmbedding.cpu().weight.data.numpy().tolist(), f)