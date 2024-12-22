import sys
import copy
import argparse
import torch
import numpy as np
import random
from tqdm import tqdm
from models.text_colab_rec import *
from pre_train.sasrec.utils import data_partition, evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # GPU train options
    parser.add_argument("--multi_gpu", action='store_true')
    parser.add_argument('--gpu_num', type=int, default=0)

    # Model setting
    parser.add_argument("--llm", type=str, default='opt', help='flan_t5, opt, vicuna')
    parser.add_argument("--recsys", type=str, default='sasrec')

    # Dataset setting
    parser.add_argument("--rec_pre_trained_data", type=str, default='Movies_and_TV')

    # Train phase setting
    parser.add_argument("--pretrain_stage1", action='store_true')
    parser.add_argument("--inference", action='store_true')

    # Hyperparameters
    parser.add_argument('--batch_size1', default=32, type=int)
    parser.add_argument('--batch_size_infer', default=2, type=int)
    parser.add_argument('--maxlen', default=50, type=int)
    parser.add_argument("--stage1_lr", type=float, default=0.0001)
    parser.add_argument("--unfrozen_recsys", action='store_true')
    parser.add_argument("--model_stage", type=str, default='stage1')
    parser.add_argument("--k", type=int, default=10)
    
    args = parser.parse_args()

    # Thiết lập thiết bị
    args.device = torch.device(f'cuda:{args.gpu_num}' if torch.cuda.is_available() else 'cpu')

    print("Loading model...")
    model = TextColabRec(args).to(args.device)

    # model_path = "/home4/khanhnd/hieupt/test2/A-LLMRec/models/saved_models"
    model.load_model(args, phase1_epoch=1, unfrozen_recsys=args.unfrozen_recsys)  

    print("Loading dataset...")
    dataset_path = f'./data/amazon/{args.rec_pre_trained_data}.txt'
    dataset = data_partition(args.rec_pre_trained_data, path=dataset_path)
    [train, valid, test, usernum, itemnum] = dataset

    print("Evaluating model...")
    k = args.k
    ndcg, hit_rate = evaluate(model, [train, valid, test, usernum, itemnum], args, k)
    print(f"NDCG@{k}: {ndcg:.4f}, Hit Rate@{k}: {hit_rate:.4f}")