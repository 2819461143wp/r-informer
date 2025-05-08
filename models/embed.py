import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math

#三角式绝对位置编码
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    

#相对位置编码
class RelativePositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # 创建位置编码表
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 初始化正弦位置编码
        self.pe.data[0, :, 0::2] = torch.sin(position * div_term)
        self.pe.data[0, :, 1::2] = torch.cos(position * div_term)

    def forward(self, seq_len):
        return self.pe[:, :seq_len, :]

# 旋转位置编码
class RotaryPositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000, theta: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.theta = theta
        
    def precompute_freqs_cis(self, dim: int, seq_len: int, device, theta: float = 10000.0):
        # 计算词向量元素两两分组之后，每组元素对应的旋转角度
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim))
        # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
        t = torch.arange(seq_len, device=device)
        # freqs.shape = [seq_len, dim // 2] 
        freqs = torch.outer(t, freqs).float()
        # 计算结果是个复数向量
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis.unsqueeze(0)  # 添加batch维度 [1, seq_len, dim//2]
    
    def apply_rotary_emb(self, xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # xq.shape = [batch_size, seq_len, dim]
        # xq_.shape = [batch_size, seq_len, dim // 2, 2]
        xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2).contiguous()
        xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2).contiguous()
        
        # 转为复数域
        xq_ = torch.view_as_complex(xq_)
        xk_ = torch.view_as_complex(xk_)
        
        # 确保freqs_cis在正确的设备上
        freqs_cis = freqs_cis.to(xq_.device)
        
        # 扩展 freqs_cis 到匹配 batch_size
        if freqs_cis.shape[0] == 1:
            freqs_cis = freqs_cis.expand(xq_.shape[0], -1, -1)
            
        # 应用旋转操作，然后将结果转回实数域
        # xq_out.shape = [batch_size, seq_len, dim]
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
        return xq_out.type_as(xq), xk_out.type_as(xk)

    def forward(self, x):
        seq_len = x.size(1)
        freqs_cis = self.precompute_freqs_cis(self.d_model, seq_len, device=x.device, theta=self.theta)
        # 将输入分成query和key（在这里它们相同）
        return self.apply_rotary_emb(x, x, freqs_cis)[0]  # 只返回query部分，因为它们相同

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4; hour_size = 24
        weekday_size = 7; day_size = 32; month_size = 13

        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        if freq=='t':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def forward(self, x):
        x = x.long()
        
        minute_x = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:,:,2])
        day_x = self.day_embed(x[:,:,1])
        month_x = self.month_embed(x[:,:,0])
        
        return hour_x + weekday_x + day_x + month_x + minute_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        #原始freq_map（无季节编码）
        # freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        #添加季节编码
        # freq_map = {'h': 5, 't': 6, 's': 7, 'm': 2, 'a': 2, 'w': 3, 'd': 4, 'b': 4}
        freq_map = {'h': 6, 't': 7, 's': 8, 'm': 3, 'a': 3, 'w': 4, 'd': 5, 'b': 5}

        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x):
        return self.embed(x)

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = RotaryPositionEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type!='timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        x = self.position_embedding(x)
        return self.dropout(x)


class DataEmbeddingWithLocalRNN(nn.Module):
    def __init__(self, c_in, d_model, rnn_type, ksize, dropout=0.1):
        super(DataEmbeddingWithLocalRNN, self).__init__()
        self.value_embedding = TokenEmbedding(c_in, d_model)
        # self.position_embedding = PositionalEmbedding(d_model)
        self.position_embedding = RelativePositionEmbedding(d_model)
        self.temporal_embedding = TemporalEmbedding(d_model, embed_type='fixed', freq='h')
        self.local_rnn = LocalRNN(d_model, d_model, rnn_type, ksize, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        x = self.local_rnn(x)
        return self.dropout(x)

class LocalRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type, ksize, dropout=0.1):
        super(LocalRNN, self).__init__()
        self.ksize = ksize
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size

        if rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=False)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=False)
        else:
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, bidirectional=False)

        # # 添加投影层，将双向输出(hidden_size*2)转换回原始维度(hidden_size)
        # self.projection = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        outputs = []
        for i in range(0, seq_len, self.ksize):
            x_chunk = x[:, i:i+self.ksize, :]
            if x_chunk.size(1) < self.ksize:
                padding = torch.zeros(batch_size, self.ksize - x_chunk.size(1), x_chunk.size(2)).to(x.device)
                x_chunk = torch.cat([x_chunk, padding], dim=1)
            output, _ = self.rnn(x_chunk)
            # # 应用投影层
            # output = self.projection(output)
            outputs.append(output)

        outputs = torch.cat(outputs, dim=1)
        outputs = outputs[:, :seq_len, :]
        return self.dropout(outputs)