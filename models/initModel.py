import torch.nn as nn
import codecs
import os
import json

class initModel(nn.Module):
    def __init__(self, config):
        super(initModel, self).__init__()
        self.batchSize = config.batchSize
        self.config = config
    
    def pos_neg_score(self,score):
        pos_score = score[:self.batchSize]
        neg_score = score[self.batchSize:].reshape(self.batchSize,-1).mean(1)
        return pos_score, neg_score


    def forward(self):
        raise NotImplementedError

    def save_embedding(self, emb_path, prefix):
        print("saving embeddings...")
        ent_path = os.path.join(emb_path, self.config.model_type+prefix+"_entity.embedding")
        rel_path = os.path.join(emb_path, self.config.model_type+prefix+"_rel.embedding")
        with codecs.open(ent_path, "w") as f:
            json.dump(self.entEmbedding.cpu().weight.data.numpy().tolist(), f)
        with codecs.open(rel_path, "w") as f:
            json.dump(self.relEmbedding.cpu().weight.data.numpy().tolist(), f)