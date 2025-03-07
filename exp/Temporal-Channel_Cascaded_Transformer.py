# Attention: GELU激活函数要求torch version >= 1.4.0

# -*-coding:utf-8-*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import math
import numpy as np
import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
import scipy.io as sci
from tools.common_utils import *
import datetime
from tensorboardX import SummaryWriter
from sklearn.model_selection import train_test_split
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

# Transformer Parameters
d_model = 256  # Embedding Size
channel_size = 192
d_ff = 2048  # FeedForward dimension (62-256-62线性提取的过程)
d_k = d_v = 128 # dimension of K(=Q), V
n_layers = 2  # number of Encoder of Decoder Layer
n_heads = 16   # number of heads in Multi-Head Attention

# Train Parameters
epochs = 700
batch_size_train = 8
batch_size_test = 8
train_acc = np.zeros(epochs)
test_acc = np.zeros(epochs)
best_accuracy = 0.
best_epoch = 0


# Save Weights and load weight
save_weights_path = '/home/yzm/PycharmProjects/BCI_Transformer/model_Cascade_Transformer/checkpoints/'

# get data
def getCharacter_data():
    # get all data and label
    character_data = sci.loadmat('../data/Character_imagine/character_imagine_1-process_10-26.mat')
    data = character_data['data']
    label = character_data['label']
    label[:] = label[:] - 1
    label = np.reshape(label, (-1, 1, 1))
    # for solit stratify
    temp_label = label.squeeze(2)
    # NaN to 0
    data = np.nan_to_num(data)
    # split train_data and test_data
    train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.2, random_state=0, stratify=temp_label, shuffle=True)
    # numpy to Tensor
    train_data_tensor = torch.Tensor(train_data)
    train_labels_tensor = torch.LongTensor(train_label.astype(np.uint8))
    test_data_tensor = torch.Tensor(test_data)
    test_labels_tensor = torch.LongTensor(test_label.astype(np.uint8))

    return train_data_tensor, train_labels_tensor, test_data_tensor, test_labels_tensor

# data package
class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_outputs[idx]


# Transformer ScaledDot_Attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


# Transformer Multi Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,
                                                                    2)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1,
                                                  1)
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  n_heads * d_v)
        output = self.fc(context)
        d_model = output.shape[2]
        return nn.LayerNorm(d_model).cuda()(output + residual), attn


# Transformer PoswiseFeedForwardNet
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU()
        )
        self.batchNorm = nn.BatchNorm1d(d_ff)
        self.fc2 = nn.Sequential(
            nn.Dropout(0.8),
            nn.Linear(d_ff, d_model, bias=False),
        )

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        input_fc1 = self.fc(inputs)
        input_fc1 = input_fc1.permute(0, 2, 1)
        input_bn = self.batchNorm(input_fc1)
        input_bn = input_bn.permute(0, 2, 1)

        output = self.fc2(input_bn)
        d_model = output.shape[2]
        return nn.LayerNorm(d_model).cuda()(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self, d_model):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model)
        self.pos_ffn = PoswiseFeedForwardNet(d_model)

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn

# Transformer Encoder on time dimension
class Time_Encoder(nn.Module):
    def __init__(self):
        super(Time_Encoder, self).__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, 204, 192))
        self.dropout = nn.Dropout(0.6)
        self.layers = nn.ModuleList([EncoderLayer(d_model=192) for _ in range(n_layers)])

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        '''
        enc_outputs = enc_inputs
        b, n, _ = enc_outputs.shape

        # Position Embedding
        enc_outputs += self.pos_emb[:,:n]
        enc_outputs = self.dropout(enc_outputs)

        enc_self_attn_mask = None
        enc_self_attns = []

        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

# Transformer Encoder on channel dimension
class Channel_Encoder(nn.Module):
    def __init__(self):
        super(Channel_Encoder, self).__init__()
        self.src_emb = nn.Linear(201, d_model ,bias=False)
        self.pos_emb = nn.Parameter(torch.randn(1, 204, 256))
        self.cls_token = nn.Parameter(torch.randn(1, 4, 256))
        self.dropout = nn.Dropout(0.6)
        self.layers = nn.ModuleList([EncoderLayer(d_model=256) for _ in range(n_layers)])

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        '''
        enc_outputs = self.src_emb(enc_inputs)
        
        # cls_token
        b, n, _ = enc_outputs.shape
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        enc_outputs = torch.cat((cls_tokens, enc_outputs), dim=1)
        
        # Position embedding
        enc_outputs += self.pos_emb[:,:(n+4)]
        enc_outputs = self.dropout(enc_outputs)
        
        enc_self_attn_mask = None
        enc_self_attns = []

        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

# Classification
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, emb_size * 2),
            nn.GELU(),
            nn.Dropout(0.6),
            nn.Linear(emb_size * 2, emb_size),
            nn.GELU(),
            nn.Dropout(0.6),
            nn.Linear(emb_size, n_classes)
        )

    def forward(self, x):
        out = self.clshead(x)
        return out

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.time_encoder = Time_Encoder().cuda()
        self.channel_encoder = Channel_Encoder().cuda()
        self.classification = ClassificationHead(d_model, 26)

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        dec_inputs: [batch_size, tgt_len, d_model]
        '''

        # Time Transformer [8, 201, 192]->[8, 201, 192]
        enc_inputs_time = enc_inputs
        enc_time_outputs, enc_self_attns = self.time_encoder(enc_inputs_time)

        # Channel transformer [8, 192, 201]->[8, 196, 256]
        enc_channel_inputs = enc_time_outputs + enc_inputs
        enc_channel_inputs = rearrange(enc_channel_inputs, 'b t c->b c t')
        enc_outputs_channel, enc_self_attns = self.channel_encoder(enc_channel_inputs)

        # Classification [8, 196, 256]->[8, 4, 256]->[8, 26]
        cls_output_channel = enc_outputs_channel[:,:4]
        classres_output = self.classification(cls_output_channel)
        return classres_output, enc_self_attns

# saveWeights
def saveWeights(train_acc, test_acc, epoch, model, optimizer):
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict()}
    for root, dir, files in os.walk("checkpoints"):
        if len(files) != 0:
            for file in files:
                if float(file.split('_')[4]) < test_acc:    # test_acc小于当前acc的模型被替换，大于当前acc的模型仍然存在,如果当前acc比所有模型都好，那所有模型都会被删除，只剩当前最好的acc
                    os.remove(os.path.join(root, file))
                    save_file = save_weights_path + 'EEGCharacter_Transformer_%d_%.4f_%.4f_weights.pth' % (
                        epoch, train_acc, test_acc)
                    torch.save(state, save_file)
                    print('The model parameters have been saved successed')
        else:
            save_file = save_weights_path + 'EEGCharacter_Transformer_%d_%.4f_%.4f_weights.pth' % (
                epoch, train_acc, test_acc)
            torch.save(state, save_file)
            print('The model parameters have been saved successed')

# learning_rate_decay
def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 60 epochs"""
    if epoch > 100:
        lr *= (0.9 ** (epoch // 30))
    else:
        lr *= (0.98 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# log
log_note = None
if log_note is None:
    log_note = str(datetime.datetime.now())
    log_note = log_note.replace(' ', '-').replace(':', '_').split('.')[0]
log_save_path = "/home/yzm/PycharmProjects/BCI_Transformer/model_Cascade_Transformer/log/" + log_note
writer = SummaryWriter(log_dir="./tensorboard_Character_logs", flush_secs=2)

def main():
    enc_inputs, dec_outputs, enc_inputs_test, dec_outputs_test = getCharacter_data()
    # 封装loader
    loader = Data.DataLoader(MyDataSet(enc_inputs, dec_outputs), batch_size_train, True)
    loader_test = Data.DataLoader(MyDataSet(enc_inputs_test, dec_outputs_test), batch_size_test, True)
    # 定义模型和优化器
    model = Transformer().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=7e-5, betas=(0.9, 0.999), eps=1e-8)
    lr_init = optimizer.param_groups[0]['lr']

    for epoch in range(epochs):
        train_correct = 0  # 训练正确的次数
        train_total = 0
        train_loss_sum = 0

        test_correct = 0  # 训练正确的次数
        test_total = 0
        test_loss_sum = 0

        batch_id = 0
        batch_id_test = 0
        adjust_learning_rate(optimizer, epoch, lr_init)
        learning_rate = optimizer.param_groups[0]['lr']

        for enc_inputs, dec_outputs in loader:
            model.train()
            enc_inputs, dec_outputs = enc_inputs.cuda(), dec_outputs.cuda()
            class_res, enc_self_attns = model(enc_inputs)

            # loss
            dec_outputs = dec_outputs.squeeze(2)
            pre_outputs = class_res.view(-1, class_res.size(-1))
            loss = criterion(pre_outputs, dec_outputs.view(-1))

            # acc
            label_acc = dec_outputs.squeeze(1).cpu().numpy()
            output_acc = torch.argmax(class_res, 1).cpu().numpy()
            predict_acc = output_acc
            predict_acc = torch.LongTensor(predict_acc)
            label_acc = torch.LongTensor(label_acc)

            train_total += label_acc.size(0)
            train_correct += (predict_acc == label_acc).sum()  # 获取训练正确地次数
            train_loss_sum += loss.item()
            batch_id += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_acc_0 = 100 * train_correct.double() / train_total
        train_acc = train_acc_0.cpu().numpy()  # train_acc_0是一个tensor数组，需要先转为cpu.tensor，再转为numpy
        train_loss = train_loss_sum / batch_id

        for enc_inputs_test, dec_outputs_test in loader_test:

            model.eval()
            enc_inputs_test, dec_outputs_test = enc_inputs_test.cuda(), dec_outputs_test.cuda()
            class_res_test, enc_self_attns_test = model(enc_inputs_test)

            # loss
            dec_outputs_test = dec_outputs_test.squeeze(2)
            pre_outputs = class_res_test.view(-1, class_res_test.size(-1))
            loss_test = criterion(pre_outputs, dec_outputs_test.view(-1))


            label_test_acc = dec_outputs_test.squeeze().cpu().numpy()
            output_acc = torch.argmax(class_res_test, 1).cpu().numpy()
            predict_test_acc = output_acc
            predict_test_acc = torch.LongTensor(predict_test_acc)
            label_test_acc = torch.LongTensor(label_test_acc)

            test_total += label_test_acc.size(0)
            test_loss_sum += loss_test.item()
            batch_id_test += 1
            test_correct += (predict_test_acc == label_test_acc).sum()  # 获取训练正确地次数

        test_acc_0 = 100 * test_correct.double() / test_total
        test_acc = test_acc_0.cpu().numpy()
        test_loss = test_loss_sum / batch_id_test

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/test', test_acc, epoch)

        print('Epoch:', '%04d' % (epoch + 1), 'train_loss =', '{:.6f}'.format(train_loss), 'train_acc =',('%.4f' % train_acc),'test_loss =', ('%.6f' % test_loss), 'test_acc =', ('%.4f' % test_acc), 'learning_rate =', ('%.8f' % learning_rate))
        print_to_log_file(log_save_path, 'Epoch:', '%04d' % (epoch + 1), 'train_loss =', ('%.6f' % train_loss), 'train_acc =',
              ('%.4f' % train_acc), 'test_loss =', ('%.6f' % test_loss), 'test_acc =', ('%.4f' % test_acc), print_to_console=False)

        # save model weights
        if test_acc >= 95.27:
            saveWeights(train_acc, test_acc, epoch + 1, model, optimizer)


        # record the best train accuracy and best train test accuracy and best test accuracy
        if epoch == 0:
            best_accuracy = test_acc
            best_train_accuracy = train_acc
            best_test_train_accuracy = train_acc
            best_epoch = epoch
            stop_count = 0
        else:
            if best_accuracy <= test_acc:
                best_accuracy = test_acc
                best_test_train_accuracy = train_acc
                best_epoch = epoch
                stop_count = 0
            else:
                stop_count += 1
        best_train_accuracy = max(train_acc, best_train_accuracy)

    # print the highest accuracy as the best accuracy
    print('best epoch : %d/%d   best train accuracy : %f%%   best test train accuracy : %f%%   best test accuracy : %f%%' \
                 % (best_epoch + 1, epochs, best_train_accuracy, best_test_train_accuracy, best_accuracy))

    print_to_log_file(log_save_path, f"best_epoch: {best_epoch + 1}    "
                                     f"best_train_acc:{best_train_accuracy:.4f}    "
                                     f"best_test_train_acc:{best_test_train_accuracy:.4f}    "
                                     f"best_test_acc:{best_accuracy:.4f}", print_to_console=False)

writer.close()

if __name__ == '__main__':
    for i in range(0,1):
        print('--------------------Time' + str(i) + '------------------')
        import time
        torch.cuda.synchronize()  # 增加同步操作
        start = time.time()
        main()
        torch.cuda.synchronize()  # 增加同步操作
        end = time.time()
        elapsed_time = end - start
        print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))




