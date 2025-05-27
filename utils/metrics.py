import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw



def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


# 相关系数R
def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def huber(true, pred, delta=1.0):
    true = torch.tensor(true, dtype=torch.float32)
    pred = torch.tensor(pred, dtype=torch.float32)
    loss = torch.where(torch.abs(true - pred) < delta, 0.5 * ((true - pred) ** 2),
                       delta * torch.abs(true - pred) - 0.5 * (delta ** 2))
    return torch.mean(loss).item()

# 皮尔逊相关系数
def pearson_corr(pred, true):
    pred = np.array(pred, dtype=np.float64).flatten()
    true = np.array(true, dtype=np.float64).flatten()
    return pearsonr(pred, true)[0]  # 只返回相关系数，不返回p值

# 斯皮尔曼相关系数
def spearman_corr(pred, true):
    pred = np.array(pred, dtype=np.float64).flatten()
    true = np.array(true, dtype=np.float64).flatten()
    return spearmanr(pred, true)[0]  # 只返回相关系数，不返回p值


# log cosh 损失
def log_cosh_loss(pred, target):
    pred = torch.tensor(pred, dtype=torch.float32)
    target = torch.tensor(target, dtype=torch.float32)
    loss = torch.mean(torch.log(torch.cosh(pred - target)))
    return loss

# 欧几里得距离
def euclidean_dist(pred, true):
    pred = np.array(pred, dtype=np.float64).flatten()
    true = np.array(true, dtype=np.float64).flatten()
    return euclidean(pred, true)

# 动态时间规整距离
def dtw_dist(pred, true):
    pred = np.array(pred, dtype=np.float64).flatten()
    true = np.array(true, dtype=np.float64).flatten()
    distance, _ = fastdtw(pred, true, dist=euclidean)
    return distance


# R² = 1 - SSres/SStot
# SSres：残差平方和，SStot：总平方和
def R2(pred, true):
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


def calculate_accuracy(pred, true):
    """
    计算分类准确率（Accuracy）

    参数:
    pred (numpy.ndarray): 预测值
    true (numpy.ndarray): 真实值

    返回:
    float: 准确率，范围在0-1之间
    """
    import numpy as np

    # 确保输入是numpy数组
    pred = np.array(pred)
    true = np.array(true)

    # 对于分类问题，通常使用argmax获取预测的类别
    if pred.ndim > 1:
        pred = np.argmax(pred, axis=-1)

    # 计算准确率
    correct = np.sum(pred == true)
    accuracy = correct / len(true)

    return accuracy


def metric(preds, trues):
    # Detach tensors and convert to NumPy arrays if they are PyTorch tensors
    if torch.is_tensor(preds):
        # Move to CPU and then convert to NumPy
        preds = preds.detach().cpu().numpy()
    if torch.is_tensor(trues):
        # Move to CPU and then convert to NumPy
        trues = trues.detach().cpu().numpy()
    
    mae = MAE(preds, trues)
    mse = MSE(preds, trues)
    rmse = RMSE(preds, trues)
    mape = MAPE(preds, trues)
    mspe = MSPE(preds, trues)
    rse = RSE(preds, trues)
    corr = CORR(preds, trues)
    # huber_loss = huber(preds, trues, 1.0)
    # log_cosh = log_cosh_loss(preds, trues)
    # spearman_corr_val = spearman_corr(preds, trues)
    # euclidean_dist_val = euclidean_dist(preds, trues)
    dtw_dist_val = dtw_dist(preds, trues)
    r2_score = R2(preds, trues)
    accuracy = calculate_accuracy(preds, trues)

    return mae, mse, rmse, mape, mspe, rse, corr, dtw_dist_val, r2_score, accuracy
