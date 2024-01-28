import math
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torchinfo import summary
from torch.nn import Transformer

def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt, device):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)
    
    return src_mask, tgt_mask

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])
   
# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Linear(vocab_size, emb_size)

    def forward(self, tokens):
        return self.embedding(tokens)
   
# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_size: int,
                 tgt_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_size)
        self.src_tok_emb = TokenEmbedding(src_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src,
                trg,
                src_mask,
                tgt_mask,
                src_padding_mask=None,
                tgt_padding_mask=None,
                memory_key_padding_mask=None):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src, src_mask):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt, memory, tgt_mask):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)
        
def train(model, data_loader, criterion, optimiser, device):
    model.train()
    losses = 0

    for src, tgt in tqdm(data_loader):
        pv, hrv, nonhrv, weather = src
        src = pv.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]
        
        src_mask, tgt_mask = create_mask(src, tgt_input, device)

        logits = model(src, tgt_input, src_mask, tgt_mask)

        optimiser.zero_grad()

        tgt_out = tgt[1:, :]
        loss = criterion(logits.reshape(-1), tgt_out.reshape(-1))
        loss.backward()

        optimiser.step()
        losses += loss.item()

    return losses / len(list(data_loader))

def validate(model, data_loader, criterion, device):
    model.eval()
    with torch.no_grad():
        losses = 0

        for src, tgt in tqdm(data_loader):
            pv, hrv, nonhrv, weather = src
            src = pv.to(device)
            tgt = tgt.to(device)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask = create_mask(src, tgt_input, device)

            logits = model(src, tgt_input, src_mask, tgt_mask)

            tgt_out = tgt[1:, :]
            loss = criterion(logits.reshape(-1), tgt_out.reshape(-1))
            losses += loss.item()

    return losses / len(list(data_loader))

# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol, device):
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len-1):
        memory = memory.to(device)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(device)
        out = model.decode(ys, memory, tgt_mask)
        # TODO:
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
    return ys

def test(model, data_loader, criterion, device):
    pass

def collate_fn(batch):    
    pv_batch, hrv_batch, non_hrv_batch, weather_batch, tgt_batch = [], [], [], [], []
    for pv, hrv, non_hrv, weather, pv_tgt in batch:
        pv_batch.append(pv)
        hrv_batch.append(hrv)
        non_hrv_batch.append(non_hrv)
        weather_batch.append(weather)
        tgt_batch.append(pv_tgt)
        
    pv_batch = torch.tensor(np.asarray(pv_batch))
    tgt_batch = torch.tensor(np.asarray(tgt_batch))
    tgt_batch = torch.cat([pv_batch[:, -1].reshape(-1, 1), tgt_batch], dim=1)

    return (pv_batch.transpose(1, 0).unsqueeze(2).float(), 
            torch.tensor(np.asarray(hrv_batch)), 
            torch.tensor(np.asarray(non_hrv_batch)), 
            torch.tensor(np.asarray(weather_batch))), tgt_batch.transpose(1, 0).unsqueeze(2).float()

        
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    torch.manual_seed(0)

    SRC_SIZE = 1
    TGT_SIZE = 1
    EMB_SIZE = 512
    NHEAD = 8
    FFN_HID_DIM = 512
    BATCH_SIZE = 6
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3

    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                    NHEAD, SRC_SIZE, TGT_SIZE, FFN_HID_DIM)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.to(device)

    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    
    src = torch.rand(16, BATCH_SIZE, SRC_SIZE)
    tgt = torch.rand(49, BATCH_SIZE, TGT_SIZE)
    
    tgt_input = tgt[:-1, :, :]
    
    src_mask, tgt_mask = create_mask(src, tgt_input, device)

    summary(transformer, input_size=[src.shape,
                                    tgt_input.shape,
                                    src_mask.shape,
                                    tgt_mask.shape])
    
    src, tgt_input = src.to(device), tgt_input.to(device)
    src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)
    logits = transformer(src, tgt_input, src_mask, tgt_mask)
    tgt_out = tgt[1:, :].to(device)
    print(logits.shape, tgt_out.shape)
    print(logits.reshape(-1).shape, tgt_out.reshape(-1).shape)
    loss = loss_fn(logits.reshape(-1), tgt_out.reshape(-1))