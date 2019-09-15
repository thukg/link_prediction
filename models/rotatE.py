import torch
import torch.nn as nn
import torch.nn.functional as F

from .initModel import initModel


class rotatE(initModel):
    def __init__(self, config):
        super(rotatE, self).__init__(config)
        self.entEmbedding = nn.Embedding(self.config.entTotal, self.config.embedding_dim*2)
        self.relEmbedding = nn.Embedding(self.config.entTotal, self.config.embedding_dim)
        self.gamma = self.config.gamma
        self.batchSize = self.config.batchSize
        self.epsilon = 2.0
        self.embedding_range = nn.Parameter(torch.Tensor([(self.gamma + self.epsilon) / self.config.embedding_dim]))

    def init(self):
        nn.init.xavier_uniform_(self.entEmbedding.weight.data)
        nn.init.xavier_uniform_(self.entEmbedding.weight.data)
    

    def pos_neg_score(self,score):
        pos_score = score[:self.batchSize]
        neg_score = score[self.batchSize:].reshape(self.batchSize,-1)
        return pos_score, neg_score


    def loss(self, score_pos, score_neg):
        score_neg = score_neg.view(self.batchSize, -1)
        if self.config.adversial_sampling:
            score_neg = F.softmax(score_neg * self.config.adversial_alpha, dim=1).detach() * F.logsigmoid(score_neg - self.gamma)
        else:
            score_neg = F.logsigmoid(score_neg - self.gamma)
        score_neg = score_neg.mean(1)
        score_pos = F.logsigmoid(self.gamma - score_pos)
        loss1 = -score_pos.mean() - score_neg.mean()
        return loss1


    def forward(self, batch):
        self.batchSize = batch.shape[0]//(1 + self.config.negativeSize * 2)
        h = batch[:, 0]
        t = batch[:, 1]
        r = batch[:, 2]
        emb_h = self.entEmbedding(h)
        emb_r = self.relEmbedding(r)
        emb_t = self.entEmbedding(t)

        score_pos, score_neg = self.rotate(emb_h ,emb_r, emb_t)

        return score_pos, score_neg
    
    def forward_x(self, batch):
        h = batch[:, 0]
        t = batch[:, 1]
        r = batch[:, 2]

        emb_h = self.entEmbedding(h)
        emb_r = self.relEmbedding(r)
        emb_t = self.entEmbedding(t)

        pi = 3.14150265359
        re_head, im_head = torch.chunk(emb_h, 2, dim=1)

        phase_relation = emb_r / (self.embedding_range.item()/pi)
        
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_score = re_head * re_relation - im_relation * im_head
        im_score = re_head * im_relation + re_relation * im_head

        score = torch.cat([re_score, im_score], dim=1)
        return score


    def rotate(self, emb_h, emb_r, emb_t):
        pi = 3.14150265359
        re_head, im_head = torch.chunk(emb_h, 2, dim=1)
        # re_relation, im_relation = torch.chunk(emb_r, 2, dim=2)
        re_tail, im_tail = torch.chunk(emb_t, 2, dim=1)

        phase_relation = emb_r / (self.embedding_range.item()/pi)
        
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)
        
        # print(re_head.shape, re_relation.shape, im_relation.shape, im_head.shape)

        re_score = re_head * re_relation - im_relation * im_head
        im_score = re_head * im_relation + re_relation * im_head
        re_score = re_score - re_tail
        im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = torch.norm(score, dim=0, p=2)
        score = torch.norm(score, dim=-1, p=2)

        score_pos = score[:self.batchSize]
        score_neg =score[self.batchSize:]
        return score_pos, score_neg

    def protate(self, emb_h, emb_r, emb_t):
        pi = 3.14159265359

        phase_head = emb_h / (self.embedding_range.item()/ pi)
        phase_relation = emb_r / (self.embedding_range.item() / pi)
        phase_tail = emb_t / (self.embedding_range.item() / pi)

        score = phase_head + phase_relation - phase_tail
        score = torch.sin(score)
        score = torch.abs(score)
        score = self.gamma.item() - score.sum(dim = 2) * self.modulus
        return score

    def predict(self, h, r, t):
        emb_h = self.entEmbedding(h)
        emb_r = self.relEmbedding(r)
        emb_t = self.entEmbedding(t)
        emb_h_pos = emb_h.view(-1, 1, self.config.embedding_dim*2)
        emb_r_pos = emb_r.view(-1, 1, self.config.embedding_dim)
        emb_t_pos = emb_t.view(-1, 1, self.config.embedding_dim*2)

        score = self.rotate(emb_h_pos, emb_r_pos, emb_t_pos)
        score = F.logsigmoid(score).squeeze(dim=1)
        return score
