import json
# from utils import load_train_id, load_test_id, load_evaluation_id, load_entity2id
from utilw.utils import load_train_id, load_test_id, load_evaluation_id, load_entity2id
from torch.utils.data import Dataset, DataLoader
import os
# from Config import Config
from utilw.Config import Config
import numpy as np
import torch
import codecs

class data_generator(Dataset):
    def __init__(self, config):
        self.entTotal = 0
        self.relTotal = 0
        self.batchSize = config.batchSize
        self.neg_cnt = config.negativeSize
        self.train_set = []
        self.valid_set = []
        self.test_set = []
        self.entity2id_dict = dict()
        self.rel2id_dict = dict()
        self.entity_set = list()
        self.rel_set = list()
        self.config = config
        # config.data_path = "./benchmarks/FB13/"
        self.triplet_set = set()

        self.load_train_data(config.data_path)
        self.load_test_data(config.data_path)
        self.load_valid_data(config.data_path)
        self.load_entity_dict(config.data_path)
        self.load_relation_dict(config.data_path)
        self.save_triple_set("./")

    def save_triple_set(self, save_path):
        with codecs.open(os.path.join(save_path, "triplet_set.txt"), "w") as f:
            f.write(str(self.triplet_set)) 

    def load_train_data(self, data_path):
        print("Loading train files...")
        self.train_set = load_train_id(os.path.join(data_path,"train2id.txt"))
        for  each in self.train_set:
            self.triplet_set.add("{} {} {}".format(each[0], each[1], each[2])) 
        return self.train_set
    
    def load_valid_data(self, data_path):
        print("Loading validation files...")
        self.valid_set = load_evaluation_id(os.path.join(data_path,"valid2id.txt"))
        for  each in self.valid_set:
            self.triplet_set.add("{} {} {}".format(each[0], each[1], each[2])) 
        return self.valid_set
    
    def load_test_data(self, data_path):
        print("Loading test files...")
        self.test_set = load_test_id(os.path.join(data_path, "test2id.txt"))
        for  each in self.test_set:
            self.triplet_set.add("{} {} {}".format(each[0], each[1], each[2])) 
        return self.test_set

    def load_entity_dict(self, data_path):
        print("Loading entity2dict...")
        self.entity2id_dict = load_entity2id(os.path.join(data_path,"entity2id.txt"))
        self.entTotal = len(self.entity2id_dict)
        self.entity_set = list(self.entity2id_dict.values())
        return self.entity_set

    def load_relation_dict(self, data_path):
        print("Loading relation2dict...")
        self.rel2id_dict = load_entity2id(os.path.join(data_path,"relation2id.txt"))
        self.relTotal = len(self.rel2id_dict)
        self.rel_set = list(self.rel2id_dict.values())
        return self.rel_set
        
    def get_relTotal(self):
        return self.relTotal
    
    def get_entTotal(self):
        return self.entTotal
    
    def head_corruput(self, head, relation, tail):
        chosen = -1
        while True:
            item = np.random.choice(self.entTotal)
            head_s = self.entity_set[item]
            if head_s == head:
                continue 
            chosen = "{} {} {}".format(head_s, tail, relation)
            chosen_rev = "{} {} {}".format(tail, head_s, relation)
            if chosen not in self.triplet_set and chosen_rev not in self.triplet_set:
                return [head_s, tail, relation]
        
    def rel_corrupt(self, head, relation, tail):
        while True:
            item = np.random.choice(self.relTotal)
            relation_s = self.rel_set[item]
            if relation_s == relation:
                continue
            chosen = "{} {} {}".format(head, tail, relation_s)
            chosen_rev = "{} {} {}".format(tail, head, relation_s)
            if chosen not in self.triplet_set and chosen_rev not in self.triplet_set:
                return [head, tail, relation_s]
    
    def tail_corrupt(self, head, relation, tail):
        while True:
            item = np.random.choice(self.entTotal)
            tail_s = self.entity_set[item]
            if tail_s == tail:
                continue
            chosen = "{} {} {}".format(head, tail_s, relation)
            chosen_rev = "{} {} {}".format(tail_s, head, relation)
            if chosen not in self.triplet_set and chosen_rev not in self.triplet_set:
                return [head, tail_s, relation]
    
    def __getitem__(self, index):
        triplet = self.train_set[index]
        neg_tri = []
        mhead = triplet[0]
        mrel = triplet[2]
        mtail = triplet[1]

        for i in range(self.neg_cnt):
            neg_tri.append(self.head_corruput(mhead, mrel, mtail))
            neg_tri.append(self.tail_corrupt(mhead, mrel, mtail))
        return triplet, neg_tri
    
    def __len__(self):
        return len(self.train_set)
    
    def get_validation_set(self):
        return self.valid_set
    
    def get_test_set(self):
        return self.test_set


def my_collate(batch):
    pos = np.array([x[0] for x in batch])
    neg = np.array([x[1] for x in batch])
    neg = np.concatenate(neg)
    print("pos:",pos.shape)
    print("neg:",neg.shape)

    return torch.LongTensor(np.concatenate([pos, neg]))

if __name__ == "__main__":
    config = Config()
    config.data_path = "../benchmarks/FB13/"

    dg = data_generator(config)
    dg.save_triple_set("./")
