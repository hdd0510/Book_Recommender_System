import random
import pickle

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import numpy as np

from models.recsys_model import *
from sentence_transformers import SentenceTransformer
class two_layer_mlp(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.fc1 = nn.Linear(dims, 128)
        self.fc2 = nn.Linear(128, dims)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x1 = self.fc2(x)
        return x, x1

class TextCollabRec(nn.Module):
    def __init__(self, args):
        super().__init__()
        rec_pre_trained_data = args.rec_pre_trained_data
        self.args = args
        self.device = args.device
        
        # Load dữ liệu tên và mô tả sản phẩm
        with open(f'./data/amazon/{args.rec_pre_trained_data}_text_name_dict.json.gz', 'rb') as ft:
            self.text_name_dict = pickle.load(ft)

        self.recsys = RecSys(args.recsys, rec_pre_trained_data, self.device)
        self.recsys.to(self.device)
        self.item_num = self.recsys.item_num
        self.rec_sys_dim = self.recsys.hidden_units
        self.sbert_dim = 768  # SBERT embedding size
        
        # MLP để ánh xạ embedding của RecSys
        self.mlp = two_layer_mlp(self.rec_sys_dim)
        
        # Nếu huấn luyện Stage 1
        if args.pretrain_stage1:
            self.sbert = SentenceTransformer('nq-distilbert-base-v1')  # Text encoder (SBERT)
            self.mlp2 = two_layer_mlp(self.sbert_dim)  # MLP để ánh xạ embedding từ SBERT
        
        # Loss functions
        self.mse = nn.MSELoss()
        self.bce_criterion = torch.nn.BCEWithLogitsLoss()
        
        # Siêu tham số và metric
        self.maxlen = args.maxlen
        self.NDCG = 0
        self.HIT = 0
        self.rec_NDCG = 0
        self.rec_HIT = 0
        self.lan_NDCG = 0
        self.lan_HIT = 0
        self.num_user = 0
        self.yes = 0
        
    def save_model(self, args, epoch1=None):
        # Lưu các thành phần của Stage 1
        out_dir = f'./models/saved_models/'
        create_dir(out_dir)
        out_dir += f'{args.rec_pre_trained_data}_{args.recsys}_{epoch1}_'
        
        if args.pretrain_stage1:
            torch.save(self.sbert.state_dict(), out_dir + 'sbert.pt')
            torch.save(self.mlp.state_dict(), out_dir + 'mlp.pt')
            torch.save(self.mlp2.state_dict(), out_dir + 'mlp2.pt') 
            torch.save([self.recsys.model.kwargs, self.recsys.model.state_dict()], out_dir + 'recsys.pth')
    
    def load_model(self, args, phase1_epoch=None, unfrozen_recsys=False):
        # Load các thành phần của Stage 1
        out_dir = f'./models/saved_models/{args.rec_pre_trained_data}_{args.recsys}_{phase1_epoch}_'
        mlp = torch.load(out_dir + 'mlp.pt', map_location=args.device)
        self.mlp.load_state_dict(mlp)
        del mlp
        for name, param in self.mlp.named_parameters():
            param.requires_grad = False

        if args.pretrain_stage1:
            mlp2 = torch.load(out_dir + 'mlp2.pt', map_location=args.device)
            self.mlp2.load_state_dict(mlp2)
            del mlp2
            for name, param in self.mlp2.named_parameters():
                param.requires_grad = False
            
            sbert_dict = torch.load(out_dir + 'sbert.pt', map_location=args.device)
            self.sbert.load_state_dict(sbert_dict)
            del sbert_dict
        if unfrozen_recsys:
            recsys = torch.load(out_dir + 'recsys.pth', map_location=args.device)
            self.recsys.model = SASRec(**recsys[0])
            self.recsys.model.load_state_dict(recsys[1])
            self.recsys.to(args.device)
            del recsys
        self.recsys.to(args.device)
    
    def find_item_text(self, item, title_flag=True, description_flag=True):
        t = 'title'
        d = 'description'
        t_ = 'No Title'
        d_ = 'No Description'
        if title_flag and description_flag:
            return [f'"{self.text_name_dict[t].get(i, t_)}, {self.text_name_dict[d].get(i, d_)}"' for i in item]
        elif title_flag and not description_flag:
            return [f'"{self.text_name_dict[t].get(i, t_)}"' for i in item]
        elif not title_flag and description_flag:
            return [f'"{self.text_name_dict[d].get(i, d_)}"' for i in item]
    
    def find_item_text_single(self, item, title_flag=True, description_flag=True):
        t = 'title'
        d = 'description'
        t_ = 'No Title'
        d_ = 'No Description'
        if title_flag and description_flag:
            return f'"{self.text_name_dict[t].get(item, t_)}, {self.text_name_dict[d].get(item, d_)}"'
        elif title_flag and not description_flag:
            return f'"{self.text_name_dict[t].get(item, t_)}"'
        elif not title_flag and description_flag:
            return f'"{self.text_name_dict[d].get(item, d_)}"'
        
    def get_item_emb(self, item_ids):
        # Lấy embedding của sản phẩm từ RecSys và MLP
        with torch.no_grad():
            item_embs = self.recsys.model.item_emb(torch.LongTensor(item_ids).to(self.device))
            item_embs, item_re = self.mlp(item_embs)
        return item_embs, item_re
    
    def forward(self, data, optimizer=None, batch_iter=None, mode='phase1'):
        # Huấn luyện Stage 1
        if mode == 'phase1':
            self.pre_train_phase1(data, optimizer, batch_iter)

    def pre_train_phase1(self, data, optimizer, batch_iter):
        epoch, total_epoch, step, total_step = batch_iter
        
        for param in self.recsys.model.parameters():
            param.requires_grad = True

        self.sbert.train()
        self.recsys.model.train()  # Đảm bảo recsys.model cũng ở chế độ huấn luyện
        optimizer.zero_grad()

        u, seq, pos, neg = data
        indices = [self.maxlen * (i + 1) - 1 for i in range(u.shape[0])]

        # Forward pass qua recsys.model
        log_emb, pos_emb, neg_emb = self.recsys.model(u, seq, pos, neg, mode='item')
        
        log_emb_ = log_emb[indices]
        pos_emb_ = pos_emb[indices]
        neg_emb_ = neg_emb[indices]
        pos_ = pos.reshape(pos.size)[indices]
        neg_ = neg.reshape(neg.size)[indices]

        start_inx = 0
        end_inx = 60
        iterss = 0
        mean_loss = 0
        bpr_loss = 0
        gt_loss = 0
        rc_loss = 0
        text_rc_loss = 0

        while start_inx < len(log_emb_):
            log_emb = log_emb_[start_inx:end_inx]
            pos_emb = pos_emb_[start_inx:end_inx]
            neg_emb = neg_emb_[start_inx:end_inx]
            
            pos__ = pos_[start_inx:end_inx]
            neg__ = neg_[start_inx:end_inx]
            
            start_inx = end_inx
            end_inx += 60
            iterss += 1
            
            pos_text = self.find_item_text(pos__)
            neg_text = self.find_item_text(neg__)
            
            pos_token = self.sbert.tokenize(pos_text)
            pos_text_embedding = self.sbert({
                'input_ids': pos_token['input_ids'].to(self.device),
                'attention_mask': pos_token['attention_mask'].to(self.device)
            })['sentence_embedding']
            
            neg_token = self.sbert.tokenize(neg_text)
            neg_text_embedding = self.sbert({
                'input_ids': neg_token['input_ids'].to(self.device),
                'attention_mask': neg_token['attention_mask'].to(self.device)
            })['sentence_embedding']
            
            pos_text_matching, pos_proj = self.mlp(pos_emb)
            neg_text_matching, neg_proj = self.mlp(neg_emb)
            
            pos_text_matching_text, pos_text_proj = self.mlp2(pos_text_embedding)
            neg_text_matching_text, neg_text_proj = self.mlp2(neg_text_embedding)
            
            pos_logits, neg_logits = (log_emb * pos_proj).mean(axis=1), (log_emb * neg_proj).mean(axis=1)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=pos_logits.device), torch.zeros(neg_logits.shape, device=neg_logits.device)

            loss = self.bce_criterion(pos_logits, pos_labels)
            loss += self.bce_criterion(neg_logits, neg_labels)
            
            matching_loss = self.mse(pos_text_matching, pos_text_matching_text) + self.mse(neg_text_matching, neg_text_matching_text)
            reconstruction_loss = self.mse(pos_proj, pos_emb) + self.mse(neg_proj, neg_emb)
            text_reconstruction_loss = self.mse(pos_text_proj, pos_text_embedding.data) + self.mse(neg_text_proj, neg_text_embedding.data)
            
            total_loss = 2*loss + matching_loss + 0.5 * reconstruction_loss + 0.2 * text_reconstruction_loss
            total_loss.backward()
            # Sau khi loss.backward()
            # for name, param in self.recsys.model.named_parameters():
            #     if param.grad is not None:
            #         print(f"Parameter: {name}, Gradient Norm: {param.grad.norm().item()}")
            #     else:
            #         print(f"Parameter: {name} has no gradient!")

            optimizer.step()
            
            mean_loss += total_loss.item()
            bpr_loss += loss.item()
            gt_loss += matching_loss.item()
            rc_loss += reconstruction_loss.item()
            text_rc_loss += text_reconstruction_loss.item()
            
        print("Loss in epoch {}/{} iteration {}/{}: {} / BPR loss: {} / Matching loss: {} / Item reconstruction: {} / Text reconstruction: {}".format(
            epoch, total_epoch, step, total_step, mean_loss / iterss, bpr_loss / iterss, gt_loss / iterss, rc_loss / iterss, text_rc_loss / iterss
    ))
    
    def predict(self, user_ids, log_seqs, item_indices):
        """
        Dự đoán điểm sử dụng dot product giữa user embedding và item embedding.
        """
        # Đồng bộ thiết bị
        model_stage = self.args.model_stage
        device = next(self.recsys.model.parameters()).device
        log_seqs = log_seqs.to(device)
        item_indices = item_indices.to('cpu')

        # Lấy embedding cuối cùng từ chuỗi lịch sử người dùng
        log_feats = self.recsys.model.log2feats(log_seqs.to('cpu')) # [batch_size, seq_len, embedding_dim]
        final_feat = log_feats[:, -1, :]  # [batch_size, embedding_dim]

        # Lấy embedding của các sản phẩm
        if model_stage=='text_collab_rec':
            _, item_embs = self.get_item_emb(item_indices.to('cpu'))
        elif model_stage=='pretrain' or 'sasrec':
            item_embs = self.recsys.model.item_emb(torch.LongTensor(item_indices).to(device))

        # Tính dot product giữa user embedding và item embedding
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits  # Trả về điểm dự đoán