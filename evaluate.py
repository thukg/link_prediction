import codecs
import json
from utilw.utils import load_train_id, load_test_id, load_evaluation_id, load_entity2id
from utilw.utils import transE_evaluate_data, ConvE_evaluate_data, DistMult_evaluate_data, rotatE_evaluate_data
from utilw.utils import load_embedding, setup_validation
import torch
import random as faiss
from utilw.preprocess import data_generator
from utilw import Config
import numpy as np
from models import *
import argparse
import os

emb_path = "./embeddings/"
k = 100

def get_ent_rel_total(data_path):
    with codecs.open(os.path.join(data_path,"entity2id.txt"),"r") as f:
        ent_total = int(f.readline().strip())
    with codecs.open(os.path.join(data_path,"relation2id.txt"), "r") as f:
        rel_total = int(f.readline().strip())
    return ent_total, rel_total

def load_model(model_path, model_type, config):
    print("loading models")

    if model_type == "transE":
        model = transE(config)
    elif model_type == "DistMult":
        model = DistMult(config)
    elif model_type == "ConvE":
        model = convE(config)
    elif model_type == "simplE":
        model = simplE(config)
    elif model_type == "rotatE":
        model = rotatE(config)
    else:
        print("unsupported model type")
        exit(0)
    model.load_state_dict(torch.load(model_path))
    if not torch.cuda.is_available():
        model.cpu()
    else:
        model.cuda()
    model.cpu()
    return model

def load_triplet_set(file_path):
    print("loading triplets")
    with codecs.open(file_path, "r") as f:
        return eval(f.read())

def indicator_(valid_set, existed_triplets, source_emb, ent_embeddings, style):
    print("computing similarity and neighbors")
    d = 100
    global k
    if style == "L2":
        D, I = L2_faiss(source_emb, ent_embeddings, d, "model_L2")
    else:
        D, I = inner_faiss(source_emb, ent_embeddings, d, "model_inner")

    MRR = []
    hits = 0
    hits_bound = 10

    print("computing hits and mrr")
    for idx, item in enumerate(valid_set):
        tail =item[1]
        matched = I[idx]
        
        rank = 1
        flag = False
        for each in matched:
            if each == tail:
                hits += 1 if rank <= hits_bound else 0
                MRR.append(1/rank)
                flag = True
                break
            
            if "{} {} {}".format(item[0], each, item[2]) in existed_triplets:
                continue
            rank += 1
        if flag == False:
            # MRR += 1/(2*k)
            MRR.append(1/(10*k))
    return hits / len(valid_set), sum(MRR)/len(MRR)


def tail_evaluate_embedding(valid_set_path, ent_emb_path, rel_emb_path, model_type):
    print("in head evaluate embedding")
    valid_set, valid_str = setup_validation(valid_set_path)
    ent_embeddings = load_embedding(ent_emb_path)
    rel_embeddings = load_embedding(rel_emb_path)
    existed_triplets = load_triplet_set("./triplet_set.txt")
    if model_type == "transE":
        source_emb = transE_evaluate_data(ent_embeddings, rel_embeddings, valid_set)
        hits_rate, mrr = indicator_(valid_set, existed_triplets, source_emb, ent_embeddings, "L2")
    elif model_type == "DistMult":
        source_emb = DistMult_evaluate_data(ent_embeddings, rel_embeddings, valid_set)
        hits_rate, mrr = indicator_(valid_set, existed_triplets, source_emb, ent_embeddings, "inner")
    else:
        print("wrong model type")
        exit(0)

    print("hit@K: {}, mrr: {}".format(hits_rate, mrr))


def tail_evaluate_simplE(valid_set_path, head_ent_path, tail_ent_path, rel_path, rel_inv_path):
    valid_set, valid_str = setup_validation(valid_set_path)
    head_ent_emb = load_embedding(head_ent_path)
    tail_ent_emb = load_embedding(tail_ent_path)
    rel_emb = load_embedding(rel_path)
    rel_inv_emb = load_embedding(rel_inv_path)
    existed_triplets = load_triplet_set("./triplet_set.txt")

    print("setup validation set")
    valid_set = np.array(valid_set)
    valid_head = valid_set[:, 0]
    valid_rel = valid_set[:, 2]

    valid_head_emb = head_ent_emb[valid_head]
    valid_rel_emb = rel_emb[valid_rel]
    assert valid_head_emb.shape == valid_rel_emb.shape
    valid_start = valid_head_emb * valid_rel_emb

    valid_head_inv_emb = tail_ent_emb[valid_head]
    valid_rel_inv_emb = rel_inv_emb[valid_rel]
    assert valid_head_inv_emb.shape == valid_rel_inv_emb.shape
    valid_inv_start = valid_head_inv_emb * valid_rel_inv_emb

    start_score = np.dot(valid_start, tail_ent_emb.transpose())
    inv_score = np.dot(valid_inv_start, head_ent_emb.transpose())
    score = 0.5 * (start_score + inv_score)*(-1)
    I = np.argsort(score, axis=1)

    MRR = []
    hits = 0
    hits_bound = 10
    global k

    print("computing hits and mrr")
    for idx, item in enumerate(valid_set):
        tail = item[1]
        matched = I[idx][:k]

        rank = 1
        flag = False
        for each in matched:
            if each == tail:
                hits += 1 if rank <= hits_bound else 0
                MRR.append(1 / rank)
                flag = True
                break

            if "{} {} {}".format(item[0], each, item[2]) in existed_triplets:
                continue
            rank += 1
        if flag == False:
            # MRR += 1/(2*k)
            MRR.append(1 / (10 * k))
    print("MRR:{}, hit@K:{}".format(sum(MRR)/len(MRR), hits/len(MRR)))


def tail_evaluate_model_embedding(valid_set_path, ent_emb_path, rel_emb_path, model_path, model_type, config):
    valid_set, valid_str = setup_validation(valid_set_path)
    ent_embeddings = load_embedding(ent_emb_path)
    model = load_model(model_path, model_type, config)
    existed_triplets = load_triplet_set("./triplet_set.txt")
    # rel_embeddings = load_embedding(rel_emb_path)
    if model_type == "convE":
        source_emb = ConvE_evaluate_data(valid_set, model)
        hits_rate, mrr = indicator_(valid_set, existed_triplets, source_emb, ent_embeddings, "inner")
    elif model_type == "rotatE":
        source_emb = rotatE_evaluate_data(valid_set, model)
        hits_rate, mrr = indicator_(valid_set, existed_triplets, source_emb, ent_embeddings, "L2")
    else:
        print("wrong model_type")
        exit(0)

    print("hit@K: {}, mrr: {}".format(hits_rate, mrr))


def L2_faiss(source_emb, obj_emb, dimension, prefix):
    global k
    index = faiss.IndexFlatL2(dimension)
    index.add(obj_emb.astype("float32"))
    D, I = index.search(source_emb, k)
    np.save(prefix + "_index.npy", I)
    np.save(prefix + "_score.npy", D)
    return D, I


def inner_faiss(source_emb, obj_emb, dimension, prefix):
    global k
    index = faiss.IndexFlatIP(dimension)
    index.add(obj_emb.astype("float32"))
    D, I = index.search(source_emb, k)
    np.save(prefix + "_index.npy", I)
    np.save(prefix + "_score.npy", D)
    return D, I


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--dataset", default="WN18", help="evaluation dataset")
    parse.add_argument("--model", default="simplE", help="evaluation model")
    parse.add_argument("--type", default="tail", help="evaluation type: head/relation/tail prediction")
    parse.add_argument("--embedding_dim", default=200, help="embedding dimension the same as training")
    args = parse.parse_args()

    dataset = args.dataset
    model_type = args.model
    eval_type = args.type
    emb_dim = args.embedding_dim
    if eval_type == "tail":
        if model_type == "DistMult" or model_type == "transE":
            ent_path = "./embeddings/{}_entity.embedding".format(model_type)
            rel_path = "./embeddings/{}_rel.embedding".format(model_type)
            tail_evaluate_embedding(dataset, ent_path, rel_path, model_type)
        elif model_type == "simplE":
            head_ent_path = "./embeddings/{}_head_entity.embedding".format(model_type)
            head_rel_path = "./embeddings/{}_rel.embedding".format(model_type)
            tail_ent_path = "./embeddings/{}_tail_entity.embedding".format(model_type)
            tail_rel_path = "./embeddings/{}_rel_rev.embedding".format(model_type)
            tail_evaluate_simplE(dataset, head_ent_path, tail_ent_path, head_rel_path,tail_rel_path)
        elif model_type == "rotatE" or model_type == "convE":
            ent_path = "./embeddings/{}_entity.embedding".format(model_type)
            rel_path = "./embeddings/{}_rel.embedding".format(model_type)
            model_path = "./train_model/{}.model".format(model_type)

            config = Config()
            config.embedding_dim = emb_dim
            config.data_path = "./benchmarks/{}/".format(dataset)
            config.entTotal, config.relTotal = get_ent_rel_total(config.data_path)
            # ent_path = "./embeddings/{}_entity.embedding".format("rotatE_test")
            # rel_path = "./embeddings/{}_rel.embedding".format("rotatE_test")
            # model_path = "./train_model/{}".format("rotatE_step_7743_epoch_13")
            tail_evaluate_model_embedding(os.path.join(config.data_path,"valid2id.txt"), ent_path, rel_path, model_path, model_type, config)
        else:
            print("unexisted model_type")
            exit(0)
    elif eval_type == "head":
        if model_type == "DistMult" or model_type == "transE":
            ent_path = "./embeddings/{}_entity.embedding".format(model_type)
            rel_path = "./embeddings/{}_rel.embedding".format(model_type)
            tail_evaluate_embedding(dataset, ent_path, rel_path, model_type)
        elif model_type == "simplE":
            head_ent_path = "./embeddings/{}_head_entity.embedding".format(model_type)
            head_rel_path = "./embeddings/{}_rel.embedding".format(model_type)
            tail_ent_path = "./embeddings/{}_tail_entity.embedding".format(model_type)
            tail_rel_path = "./embeddings/{}_rel_rev.embedding".format(model_type)
            tail_evaluate_simplE(dataset, head_ent_path, tail_ent_path, head_rel_path, tail_rel_path)
        elif model_type == "rotatE" or model_type == "convE":
            ent_path = "./embeddings/{}_entity.embedding".format(model_type)
            rel_path = "./embeddings/{}_rel.embedding".format(model_type)
            model_path = "./train_model/{}.model".format(model_type)

            config = Config()
            config.embedding_dim = emb_dim
            config.data_path = "./benchmarks/{}/".format(dataset)
            config.entTotal, config.relTotal = get_ent_rel_total(config.data_path)
            # ent_path = "./embeddings/{}_entity.embedding".format("rotatE_test")
            # rel_path = "./embeddings/{}_rel.embedding".format("rotatE_test")
            # model_path = "./train_model/{}".format("rotatE_step_7743_epoch_13")
            tail_evaluate_model_embedding(os.path.join(config.data_path, "valid2id.txt"), ent_path, rel_path,
                                          model_path, model_type, config)
        else:
            print("unexisted model_type")
            exit(0)

