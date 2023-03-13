import torch.nn as nn
import torch
import json
from preprocessing import data_preprocessing, preprocess_single_string, padding, vocab_to_int


device='cpu'

class RNNNet(nn.Module):    
    '''
    vocab_size: int, размер словаря (аргумент embedding-слоя)
    emb_size:   int, размер вектора для описания каждого элемента последовательности
    hidden_dim: int, размер вектора скрытого состояния
    batch_size: int, размер batch'а

    '''
    
    def __init__(self, 
                 vocab_size: int, 
                 emb_size: int, 
                 hidden_dim: int, 
                 seq_len: int, 
                 n_layers: int = 1) -> None:
        super().__init__()
        
        self.seq_len  = seq_len 
        self.emb_size = emb_size 
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, self.emb_size)
        self.rnn_cell  = nn.RNN(self.emb_size, self.hidden_dim, batch_first=True, num_layers=n_layers)
        self.linear    = nn.Sequential(
            nn.Linear(self.hidden_dim * self.seq_len, 256),
            nn.Dropout(),
            nn.Sigmoid(),
            nn.Linear(256, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.input = x.size(0)
        x = self.embedding(x.to(device))
        output, _ = self.rnn_cell(x, self.init_hidden()) 
        # print(f'RNN output: {output.shape}')
        output = output.contiguous().view(output.size(0), -1)
        out = self.linear(output.squeeze(0))
        return out
    
    def init_hidden(self):
        self.h_0 = torch.zeros((self.n_layers, self.input, self.hidden_dim), device=device)
        return self.h_0

with open('data/vocab_seq2seq.json') as f:
    vocab_to_int = json.load(f)   

VOCAB_SIZE = len(vocab_to_int)+1
SEQ_LEN = 32
N_LAYERS = 2
EMBEDDING_DIM = 64
HIDDEN_DIM = 32      

def load_model(): 
    model = RNNNet(
    VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, SEQ_LEN
).to(device) 
    model.load_state_dict(torch.load('data/seq2seq_Rnn.pth', map_location='cpu'))
    model.eval()
    return model

def predict_sentiment(review):
    model = load_model()
    prediction = model.to(device)(preprocess_single_string(review, seq_len=SEQ_LEN, vocab_to_int=vocab_to_int).unsqueeze(0).to(device)).sigmoid().item()
    if prediction > 0.5:
        return f''' Positive
Score of prediction:  {prediction:.3f}'''
    else:
        return f''' Negative
Score of prediction:  {prediction:.3f}'''


