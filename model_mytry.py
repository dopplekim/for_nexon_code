from copy import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import lightning as L

from einops import rearrange

from transformer_encoder.utils import PositionalEncoding
from transformer_encoder import TransformerEncoder

from utils import gamma_function, sample_with_temperature as sampler
from params import *


class MaskgitTransformer(L.LightningModule):
    def __init__(self, n_token = 1024, d_model = 512, d_ff = 2048, n_heads = 8, n_layers = 6, dropout = 0.1, max_len = MAX_LENGTH):
        super(MaskgitTransformer, self).__init__()
        
        # Save hyperparameters automatically
        self.save_hyperparameters()

        self.loss_acc_weights = torch.tensor([3.0, 2.0, 2.0, 1.5, 1.2, 1.0, 1.0, 1.0])    # weights for averaging loss & acc during training
        self.logit_weights = torch.tensor([1.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])   # weights for amplifying low RVQ layer logits during inference
        self.n_token = n_token

        # Layers
        self.embs = nn.ModuleList([ nn.Embedding(num_embeddings=n_token, embedding_dim=d_model) for _ in range(8) ])
        self.pos_enc = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len)
        self.encoder = TransformerEncoder(d_model=d_model, d_ff=d_ff, n_heads=n_heads, n_layers=n_layers, dropout=dropout)
        #self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout)
        #self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(d_model, n_token*8)
        #self.fc = nn.Linear(d_model, n_token)

        # Loss
        self.criterion = nn.CrossEntropyLoss()

    def common_step(self, batch, batch_idx, gamma = 0.9):
        _, _, noisy_rvq, clean_rvq = batch

        B, _, T = noisy_rvq.size()   # shape: (B, 8, T)

        attn_mask = (clean_rvq != pad_token).to(clean_rvq.device) # (shape: (B, 8, T)) (see ./transformer_encoder/multi_head_attention.py)

        # embedding
        noisy_emb = torch.stack([self.embs[i](noisy_rvq[:,i]) for i in range(8)], dim = 1)        # shape: (B, 8, T, D)
        clean_emb = torch.stack([self.embs[i](clean_rvq[:,i]) for i in range(8)], dim = 1)        # shape: (B, 8, T, D)

        # masking
        mixed_emb = clean_emb.clone()      # shape: (B, 8, T, D)

        idxs = torch.where(attn_mask)      # idxs[0] = batch idx / idxs[1] = layer idx / idx[2] = time index
        n_total = len(idxs[0])
        n_total_noisy = max(int(gamma * n_total), 1)
        select = torch.randperm(n_total)[:n_total_noisy]
        noisy_idxs = (idxs[0][select], idxs[1][select], idxs[2][select])

        mixed_emb[noisy_idxs] = noisy_emb[noisy_idxs]

        # sum
        mixed_emb = mixed_emb.sum(dim = 1)  # shape: (B, T, D)
        attn_mask = attn_mask[:,0]          # shape: (B, T)

        # predict all layers
        enhanced_logit = self.forward(mixed_emb, attn_mask)  # shape: (B, 8, T, C)

        # sum of loss for each layer
        total_weight = 0.
        loss = 0.
        acc = 0.
        #####
        n_total_masking = 0.
        #####
        
        for j in range(8):
            is_jth_layer = torch.where(noisy_idxs[1] == j)[0]
            if is_jth_layer.numel() > 0:
                noisy_idxs_jth_layer = (noisy_idxs[0][is_jth_layer], noisy_idxs[2][is_jth_layer])
                enhanced_logit_jth_layer = enhanced_logit[:,j][noisy_idxs_jth_layer]     # shape: (n_total_noisy_jth_layer, C)
                sample_rvq_jth_layer = torch.argmax(enhanced_logit_jth_layer, dim = -1)  # shape: (n_total_noisy_jth_layer,)
                #####
                n_masking_one_layer = sample_rvq_jth_layer.size()[0]
                #####
                clean_rvq_jth_layer = clean_rvq[:,j][noisy_idxs_jth_layer]               # shape: (n_total_noisy_jth_layer,)
                loss += self.criterion(enhanced_logit_jth_layer, clean_rvq_jth_layer) * self.loss_acc_weights[j] * n_masking_one_layer
                #####acc += torch.eq(sample_rvq_jth_layer, clean_rvq_jth_layer).sum() / clean_rvq_jth_layer.numel() * 100 * self.loss_acc_weights[j]
                #acc += torch.eq(sample_rvq_jth_layer, clean_rvq_jth_layer).sum() / clean_rvq_jth_layer.numel() * 100
                acc += torch.eq(sample_rvq_jth_layer, clean_rvq_jth_layer).sum()
                n_total_masking += n_masking_one_layer
                #####total_weight += self.loss_acc_weights[j]
                '''
                if torch.isnan(loss) or torch.isnan(acc):
                    print(sample_rvq_jth_layer)
                    print(clean_rvq_jth_layer)
                    import sys; sys.exit()
                '''
        #####loss = loss / total_weight
        #####acc = acc / total_weight
        loss = loss / n_total_masking
        acc = acc / n_total_masking * 100


        '''
        enhanced_logit = enhanced_logit[noisy_idxs]          # shape: (n_total_noisy, C)
        clean_rvq = clean_rvq[noisy_idxs]                    # shape: (n_total_noisy)

        loss = self.criterion(enhanced_logit, clean_rvq)

        sample_rvq = torch.argmax(enhanced_prob, dim = -1)   # shape: (n_total_noisy,)
        acc = torch.eq(sample_rvq, clean_rvq).sum() / clean_rvq.numel() * 100

        del noisy_rvq, clean_rvq, mixed_emb, enhanced_logit, sample_rvq, attn_mask, noisy_idxs
        '''
        del noisy_rvq, clean_rvq, mixed_emb, enhanced_logit, attn_mask, enhanced_logit_jth_layer, sample_rvq_jth_layer, clean_rvq_jth_layer, noisy_idxs
        return loss, acc

    def training_step(self, batch, batch_idx):
        gamma = torch.rand(size = (1,))
        loss, acc = self.common_step(batch, batch_idx, gamma)
        self.log_dict({'train_loss': loss, 'train_acc': acc}, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        return loss

    def validation_step(self, batch, batch_idx):
        gamma = torch.rand(size = (1,))
        loss, acc = self.common_step(batch, batch_idx, gamma)
        self.log_dict({'val_loss': loss, 'val_acc': acc}, on_step = False, on_epoch = True, prog_bar = True, logger = True)

    def test_step(self, batch, batch_idx):
        gamma = torch.rand(size = (1,))
        loss, acc = self.common_step(batch, batch_idx, gamma)
        self.log_dict({'test_loss': loss, 'test_acc': acc}, on_step = False, on_epoch = True, prog_bar = True, logger = True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = lr)
        return optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx = 0):
        return self(batch)

    def forward(self, mixed_emb, attn_mask = None):
        '''
        mixed_emb shape: (B, T, D)
        attn_mask shape: (B, T) (1 = used / 0 = masked)
        '''
        # Positional Encoding
        mixed_emb = self.pos_enc(mixed_emb)  # shape: (B, T, D)

        # Encoder
        pred = self.encoder(mixed_emb, attn_mask)  # shape: (B, T, D)

        # FC
        #out = self.fc(pred)   # shape: (B, T, C)
        out = self.fc(pred)   # shape: (B, T, 8*C)

        # reshape
        out = rearrange(out, "b t (e c) -> b e t c", e = 8)  # shape: (B, 8, T, C)

        return out

    def inference(self, noisy, max_pred_iter = 10, temperature = 1.2, 
                  get_intermediate_results = False, verbose = False):   #~~~~~~~~~~~~~~
        '''
        noisy shape: (1, 8, T)
        '''
        C = self.n_token
        T = noisy.size(-1)
        idx_batch = torch.stack([ torch.arange(8) ] * T, dim = -1).view(-1)   # shape: (8, T) --> (8*T,), content: [0, 0, ..., 0, 1, 1, ... 1, ...]
        idx_time = torch.tile(torch.arange(T), (8,)).view(-1)                 # shape: (8, T) --> (8*T,), content: [0, 1, ..., T, 0, 1, .., T, ...]

        # noisy embedding
        noisy_emb = torch.stack([ self.embs[i](noisy[:,i]) for i in range(8) ], dim = 1)   # shape: (1, 8, T, D)

        #~~~~~~~~~~~~~~
        if get_intermediate_results:
            enhanced_list = []
            enhanced_list += [noisy.clone()]
        #~~~~~~~~~~~~~~

        # enhanced RVQ token
        enhanced = noisy.clone()   # shape: (1, 8, T)

        # enhanced embedding (initially same as noisy_emb --> iteratively changed to predicted token's embedding)
        enhanced_emb = noisy_emb.clone()  # shape: (1, 8, T, D) 

        # padding mask (1 = used / 0 = masked)
        attn_mask = (noisy[:,0] != pad_token).to(noisy.device)   # shape: (1, T)

        # whether each token is changed 
        is_changed = torch.full((noisy.numel(),), False)  # shape: (8*T,)
        #is_changed = torch.full(noisy.shape, False)  # shape: (1, 8, T)

        if verbose:
            print(noisy[..., 0, :50], torch.sum(is_changed))
            print("-" * 50)

        with torch.no_grad():

            for ip in range(max_pred_iter):

                # Predict
                #logit = self.forward(enhanced_emb, attn_mask)    # shape: (1, T, C)
                logit = self.forward(enhanced_emb.sum(dim = 1), attn_mask)      # shape: (1, 8, T, C)
                logit[..., -1] = float("-inf")                                  # mask logit for padding token
                logit *= self.logit_weights[None,:,None,None].to(logit.device)  # amplify lower layer's probability

                logit = logit.reshape(-1, C)            # shape: (8*T, C)
                prob = F.softmax(logit, dim = -1)       # shape: (8*T, C)

                # Sample
                sample_rvq = sampler(prob, temperature).squeeze()      # shape: (8*T,)
                prob_sample = prob[torch.arange(8*T), sample_rvq]      # shape: (8*T,)
                prob_sample[is_changed] = float('inf')                 # so that already-changed tokens will always sampled in top (n_total - n_noisy) indices (shape: (8*T,))
                #sample_rvq = sampler(prob.view(-1, C), temperature)#.view(1, -1, T, C)    # shape: (1, 8, T, C)
                #prob_sample = prob[0, idx_batch, idx_time, sample_rvq].view(1, 8, T)      # shape: (8*T) --> (1, 8, T)
                #prob_sample = torch.tensor([prob[0,j,sample_rvq[0,j]] for j in range(T)]).view(1, T)   # shape: (1, T)
                #prob_sample[is_changed] = float('inf')   # so that already-changed tokens will always sampled in top (n_total - n_noisy) indices

                # Calculate (# of timesteps that will be masked)
                r = (ip+1) / max_pred_iter
                gamma = gamma_function(r)
                n_total = 8*T
                n_noisy = int(gamma * n_total)
                #n_noisy = max(int(gamma * T), 1)

                # Find batch & timestep indices with top (T - n_noisy) confidence (i.e. probability)
                prob_topk, indices_topk = torch.topk(prob_sample, k = n_total - n_noisy)  # top (n_total-n_noisy) probability & where it is observed (range: 0~(8*T-1))
                indices_topk = indices_topk[prob_topk != float('inf')]                    # remove already-changed indices
                #indices_2d_topk = torch.unravel_index(indices_topk, (8, T))               # 2D index where top probabilities are observed (indices_2d_topk[0] range: 0~7 / indices_2d_topk[1] range: 0~(T-1))
                #prob_topk, indices_topk = torch.topk(prob_sample.view(-1), k = n_total - n_noisy)  # top (n_total-n_noisy) probability & where it is observed (range: 0~(T-1))
                #indices_topk = indices_topk[prob_topk != float('inf')]                       # remove already-changed indices

                # Reflect predictions at indices_2d_topk
                is_changed[indices_topk] = True
                for j in range(8):
                    is_jth_layer = ((indices_topk // T) == j)
                    idx_time_jth_layer = indices_topk[is_jth_layer] % T
                    sample_rvq_jth_layer = sample_rvq[indices_topk[is_jth_layer]]
                    enhanced[0, j, idx_time_jth_layer] = sample_rvq_jth_layer
                    enhanced_emb[0, j, idx_time_jth_layer] = self.embs[j](sample_rvq_jth_layer)

                #~~~~~~~~~~~~~~
                if get_intermediate_results:
                    enhanced_list += [enhanced.clone()]

                if verbose:
                    print(enhanced[..., 0, :50], sum(is_changed), enhanced.numel(), enhanced.shape)
                    print("-" * 50)
                #~~~~~~~~~~~~~~

        #~~~~~~~~~~~~~~
        if get_intermediate_results:
            return enhanced_list
        else:
            return enhanced
        #~~~~~~~~~~~~~~

