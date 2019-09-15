import torch
import torch.nn as nn
import numpy as np
from .initModel import initModel
import torch.nn.functional as F

class convE(initModel):
    def __init__(self, config):
        super(convE, self).__init__(config)
        self.entEmbedding = nn.Embedding(self.config.entTotal, self.config.embedding_dim)
        self.relEmbedding = nn.Embedding(self.config.relTotal, self.config.embedding_dim)
        self.con = nn.Conv2d(1, 32, kernel_size=(3,3), stride=1, padding=0, bias=self.config.use_bias)
        self.fc = nn.Linear(18*18*32, self.config.embedding_dim)

        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm1d(config.embedding_dim)

        self.input_drop = nn.Dropout(0.2)
        self.conv_drop = nn.Dropout(0.2)
        self.hidden_drop = nn.Dropout(0.3)

        self.epsilon = 0.1

    def init(self):
        nn.init.xavier_uniform_(self.entEmbedding.weight.data)
        nn.init.xavier_uniform_(self.relEmbedding.weight.data)
        nn.init.xavier_uniform_(self.con.weight.data)
    
    
    def loss(self, score_pos, score_neg):
        num = score_neg.shape[0] + score_pos.shape[0]
        loss1 = -torch.mean(F.logsigmoid(score_pos) * (1-self.epsilon) + F.logsigmoid(-score_neg) * self.epsilon)

        return loss1

    def forward(self,batch):
        self.batchSize = batch.shape[0]//(1 + self.config.negativeSize * 2)
        h = batch[:,0]
        t = batch[:,1]
        r = batch[:,2]
        emb_h = self.entEmbedding(h).view(self.batchSize * 3, 1, 10, -1)
        emb_t = self.entEmbedding(t)
        emb_r = self.relEmbedding(r).view(self.batchSize * 3, 1, 10, -1)#(1,10,20) for dim 200

        conv_inputs = torch.cat([emb_h, emb_r], 2)

        # x = self.bn0(conv_inputs)
        x = conv_inputs
        x = self.input_drop(x)
        x = self.con(x)
        # x = self.bn1(x)
        x = F.relu(x)
        x = self.conv_drop(x)

        x = x.view(self.batchSize * 3, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        # x = self.bn2(x)
        x = F.relu(x)

        x = torch.sum(x*emb_t,-1)
        score_pos, score_neg = self.pos_neg_score(x)
        return score_pos, score_neg 

    def forward_x(self, batch):
        mbatch_size = len(batch)
        
        h = batch[:,0]
        r = batch[:,1]
        emb_h = self.entEmbedding(h).view(mbatch_size, 1, 10, -1)
        emb_r = self.relEmbedding(r).view(mbatch_size, 1, 10, -1)#(1,10,20) for dim 200

        conv_inputs = torch.cat([emb_h, emb_r], 2)
        # x = self.bn0(conv_inputs)
        x = self.con(conv_inputs)
        # x = self.bn1(x)
        x = F.relu(x)

        x = x.view(mbatch_size, -1)
        x = self.fc(x)
        # x = self.bn2(x)
        x = F.relu(x)
        return x


    def predict(self, h, r, t):
        emb_h = self.entEmbedding(h)
        emb_r = self.relEmbedding(r)
        emb_t = self.entEmbedding(t)

        score = torch.norm(emb_h + emb_r - emb_t, dim=1)
        return score

