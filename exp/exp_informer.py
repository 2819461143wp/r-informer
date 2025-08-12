from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

import warnings

warnings.filterwarnings('ignore')
import json


class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
        self.train_losses = []  # 存储每个 epoch 的训练损失
        self.test_losses = []  # 存储每个 epoch 的测试损失
        self.mse_train = []  # 存储每个 epoch 的训练 mse
        self.mse_test = []  # 存储每个 epoch 的测试 mse
        self.mae_train = []  # 存储每个 epoch 的训练 mae
        self.mae_test = []  # 存储每个 epoch 的测试 mae
        self.rmse_train = []  # 存储每个 epoch 的训练 rmse
        self.rmse_test = []  # 存储每个 epoch 的测试 rmse
        self.mape_train = []  # 存储每个 epoch 的训练 mape
        self.mape_test = []  # 存储每个 epoch 的测试 mape
        self.mspe_train = []  # 存储每个 epoch 的训练 mspe
        self.mspe_test = []  # 存储每个 epoch 的测试 mspe
        self.rse_train = []  # 存储每个 epoch 的训练 rse
        self.rse_test = []  # 存储每个 epoch 的测试 rse
        self.corr_train = []  # 存储每个 epoch 的训练 corr
        self.corr_test = []  # 存储每个 epoch 的测试 corr
        self.spearman_corr_train = []  # 存储每个 epoch 的训练 spearman_corr
        self.spearman_corr_test = []  # 存储每个 epoch 的测试 spearman_corr
        self.euclidean_dist_train = []  # 存储每个 epoch 的训练 euclidean_dist
        self.euclidean_dist_test = []  # 存储每个 epoch 的测试 euclidean_dist
        self.dtw_dist_train = []  # 存储每个 epoch 的训练 dtw_dist
        self.dtw_dist_test = []  # 存储每个 epoch 的测试 dtw_dist
        self.r2_score_train = []  # 存储每个 epoch 的训练 r2_score
        self.r2_score_test = []  # 存储每个 epoch 的测试 r2_score
        self.accuracy_train = []  # 存储每个 epoch 的训练 accuracy
        self.accuracy_test = []  # 存储每个 epoch 的测试 accuracy

    def convert_to_serializable(self, obj):
        if isinstance(obj, dict):
            return {k: self.convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
            return obj

    def save_losses(self, filename="losses.json"):
        """将训练和测试损失保存到文件中"""
        losses_dict = {
            "train_losses": self.train_losses,
            "test_losses": self.test_losses
        }

        losses_dict = self.convert_to_serializable(losses_dict)

        with open(filename, 'w') as f:
            json.dump(losses_dict, f, indent=4)  # 使用 json 保存为可读格式
        print(f"Losses saved to {filename}")

    def save_train_metrics(self, filename="train_metrics.json"):
        """将训练指标保存到文件中"""
        metrics_dict = {
            "mse": self.mse_train,
            "mae": self.mae_train,
            "rmse": self.rmse_train,
            "mape": self.mape_train,
            "mspe": self.mspe_train,
            "rse": self.rse_train,
            "corr": self.corr_train,
            "spearman_corr": self.spearman_corr_train,
            "euclidean_dist": self.euclidean_dist_train,
            "dtw_dist": self.dtw_dist_train,
            "r2_score": self.r2_score_train,
            "accuracy": self.accuracy_train
        }

        metrics_dict = self.convert_to_serializable(metrics_dict)

        with open(filename, 'w') as f:
            json.dump(metrics_dict, f, indent=4)
        print(f"Train metrics saved to {filename}")

    def save_test_metrics(self, filename="test_metrics.json"):
        """将测试指标保存到文件中"""
        metrics_dict = {
            "mse": self.mse_test,
            "mae": self.mae_test,
            "rmse": self.rmse_test,
            "mape": self.mape_test,
            "mspe": self.mspe_test,
            "rse": self.rse_test,
            "corr": self.corr_test,
            "spearman_corr": self.spearman_corr_test,
            "euclidean_dist": self.euclidean_dist_test,
            "dtw_dist": self.dtw_dist_test,
            "r2_score": self.r2_score_test,
            "accuracy": self.accuracy_test
        }

        metrics_dict = self.convert_to_serializable(metrics_dict)

        with open(filename, 'w') as f:
            json.dump(metrics_dict, f, indent=4)
        print(f"Test metrics saved to {filename}")

    def _build_model(self):
        model_dict = {
            'informer': Informer,
            'informerstack': InformerStack,
        }
        if self.args.model == 'informer' or self.args.model == 'informerstack':
            e_layers = self.args.e_layers if self.args.model == 'informer' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in,
                self.args.c_out,
                self.args.seq_len,
                self.args.label_len,
                self.args.pred_len,
                self.args.factor,
                self.args.d_model,
                self.args.n_heads,
                e_layers,  # self.args.e_layers,
                self.args.d_layers,
                self.args.d_ff,
                self.args.dropout,
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device,
                self.args.task_type,
                self.args.num_classes
            ).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTh1': Dataset_ETT_hour,
            'ETTh2': Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_minute,
            'ETTm2': Dataset_ETT_minute,
            'WTH': Dataset_Custom,
            'ECL': Dataset_Custom,
            'Solar': Dataset_Custom,
            'custom': Dataset_Custom,
            'qiantangjiang': Dataset_Custom,
            'QianTangRiver2020-2024WorkedFull': Dataset_Custom,
            'new_data': Dataset_Custom,
            'sensor_data': Dataset_Custom,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed != 'timeF' else 1

        if flag == 'test':
            shuffle_flag = False;
            drop_last = True;
            batch_size = args.batch_size;
            freq = args.freq
        elif flag == 'pred':
            shuffle_flag = False;
            drop_last = False;
            batch_size = 1;
            freq = args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True;
            drop_last = True;
            batch_size = args.batch_size;
            freq = args.freq
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols,
            task_type=args.task_type
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.task_type == 'classification':
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            pred, true = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        # vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1

                model_optim.zero_grad()
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)

                loss = criterion(pred, true)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            # vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            # 记录损失值
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Test Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, test_loss))
            early_stopping(test_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            # 在每个epoch结束时保存模型
            torch.save(self.model.state_dict(), os.path.join(path, 'checkpoint.pth'))
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # 保存损失值和训练指标
        self.save_losses(filename=os.path.join(self.args.checkpoints, setting, "losses.json"))
        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')

        self.model.eval()

        preds = []
        trues = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        if self.args.task_type == 'classification':
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            pred_labels = np.argmax(preds, axis=1)
            
            accuracy = accuracy_score(trues, pred_labels)
            precision, recall, f1, _ = precision_recall_fscore_support(trues, pred_labels, average='weighted')
            
            print('\nClassification Metrics:')
            print(f'Accuracy: {accuracy:.6f}')
            print(f'Precision: {precision:.6f}')
            print(f'Recall: {recall:.6f}')
            print(f'F1 Score: {f1:.6f}')
            
            report = classification_report(trues, pred_labels)
            print('\nClassification Report:')
            print(report)

            folder_path = './results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            with open(os.path.join(folder_path, 'classification_report.txt'), 'w') as f:
                f.write(report)

        else:
            preds = np.array(preds)
            trues = np.array(trues)
            print('test shape:', preds.shape, trues.shape)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            print('test shape:', preds.shape, trues.shape)

            # result save
            folder_path = './results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # 计算并记录测试指标
            mae, mse, rmse, mape, mspe, rse, corr, spearman_corr_val, euclidean_dist_val, dtw_dist_val, r2_score, accuracy = metric(
                preds, trues)

            print('\nTest Metrics:')
            print(f'MAE: {np.mean(mae):.6f}')
            print(f'MSE: {np.mean(mse):.6f}')
            print(f'RMSE: {np.mean(rmse):.6f}')
            print(f'MAPE: {np.mean(mape):.6f}')
            print(f'MSPE: {np.mean(mspe):.6f}')
            print(f'RSE: {np.mean(rse):.6f}')
            print(f'CORR: {np.mean(corr):.6f}')
            print(f'Spearman: {np.mean(spearman_corr_val):.6f}')
            print(f'Euclidean: {np.mean(euclidean_dist_val):.6f}')
            print(f'DTW: {np.mean(dtw_dist_val):.6f}')
            print(f'R2: {np.mean(r2_score):.6f}')
            print(f'Accuracy: {np.mean(accuracy):.6f}')

            # 同时保存numpy格式
            np.save(folder_path + 'test_metrics.npy', np.array(
                [mae, mse, rmse, mape, mspe, rse, corr, spearman_corr_val, euclidean_dist_val, dtw_dist_val, r2_score,
                 accuracy]))
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()

        preds = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
            pred, true = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)

        if self.args.task_type == 'classification':
            batch_y = batch_y.long().to(self.device)
            
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch_x, batch_x_mark, None, None)
            else:
                outputs = self.model(batch_x, batch_x_mark, None, None)
            
            return outputs, batch_y
        else:
            batch_y = batch_y.float()
            batch_y_mark = batch_y_mark.float().to(self.device)

            # decoder input
            if self.args.padding == 0:
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
            elif self.args.padding == 1:
                dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
            
            # encoder - decoder
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            if self.args.inverse:
                outputs = dataset_object.inverse_transform(outputs)
            
            f_dim = -1 if self.args.features == 'MS' else 0
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

            return outputs, batch_y
