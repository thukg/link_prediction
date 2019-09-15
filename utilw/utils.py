import codecs
import json
import torch
import numpy as np
import re


def load_train_id(filename):
    print(filename)
    with codecs.open(filename, "r") as f:
        data = f.readlines()
    data = [re.split(r"[\t|\s]+", i.strip()) for i in data[1:-1]]
    data = [[int(x[0]), int(x[1]), int(x[2])] for x in data]
    return data


def load_test_id(filename):
    with codecs.open(filename, "r") as f:
        data = f.readlines()
    # data = [i.strip().split("\t") for i in data[1:-1]]
    data = [re.split(r"[\t|\s]+", i.strip()) for i in data[1:-1]]
    data = [[int(x[0]), int(x[1]), int(x[2])] for x in data]
    return data

    
def load_evaluation_id(filename):
    with codecs.open(filename, "r") as f:
        data = f.readlines()
    # data = [i.strip().split("\t") for i in data[1:-1]]
    data = [re.split(r"[\t|\s]+", i.strip()) for i in data[1:-1]]
    data = [[int(x[0]), int(x[1]), int(x[2])] for x in data]
    return data


def load_entity2id(filename):
    with codecs.open(filename, "r") as f:
        data = f.readlines()
    data = [i.strip().split("\t") for i in data[1:]]
    data_dict = {index:int(val) for index, val in data}
    return data_dict


def transE_evaluate_data(ent_embedding, rel_embedding, valid_set):
    triplets = []
    for each in valid_set:
        triplets.append(ent_embedding[each[0]] + rel_embedding[each[2]])
    return np.array(triplets).astype("float32")


def DistMult_evaluate_data(ent_embedding,rel_embedding, valid_set):
    triplets = []
    for each in valid_set:
        # print(each)
        triplets.append(ent_embedding[each[0]] * rel_embedding[each[2]])
    return np.array(triplets).astype("float32")

def ConvE_evaluate_data(valid_set, model):
    triplets = torch.from_numpy(valid_set)
    triplets = model.forward_x(triplets)
    return triplets.numpy()


def ConvE_eval_all_data(head_rel, model):
    triplets = torch.from_numpy(head_rel)
    triplets = model.forward_x(triplets)
    return triplets.numpy()


def rotatE_evaluate_data(valid_set, model):
    valid_set = np.array(valid_set)
    triplets = torch.LongTensor(valid_set)
    triplets = model.forward_x(triplets)
    return triplets.detach().numpy()


def setup_validation(file_path):
    print("loading validation set")
    valid_set = load_evaluation_id(file_path)
    valid_str = ["{} {} {}".format(x[0], x[1], x[2]) for x in valid_set]
    return valid_set, valid_str


def load_embedding(file_path):
    print("loading embeddings")
    with codecs.open(file_path, "r") as f:
        return np.array(json.load(f))


