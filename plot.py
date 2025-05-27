import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime
from scipy.interpolate import interp1d
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    return np.mean((pred - true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    mask = true != 0
    if not np.any(mask):
        return float('inf')
    return np.mean(np.abs((pred[mask] - true[mask]) / true[mask]))

def CORR(pred, true):
    # 使用 numpy 的 corrcoef 函数计算相关系数
    if len(pred) < 2:  # 处理长度过短的情况
        return 0
    correlation = np.corrcoef(pred, true)
    return correlation[0, 1] if not np.isnan(correlation[0, 1]) else 0

def R2(pred, true):
    # 使用 sklearn 的 r2_score
    return r2_score(true, pred)

def RSE(pred, true):
    numerator = np.sum((true - pred) ** 2)
    denominator = np.sum((true - true.mean()) ** 2)
    return np.sqrt(numerator / denominator) if denominator != 0 else float('inf')
def find_best_fitting_window(pred, true, window_size=24):
    best_metrics = None
    best_start_idx = 0
    best_score = -float('inf')

    for i in range(len(pred) - window_size):
        window_pred = pred[i:i+window_size]
        window_true = true[i:i+window_size]

        # 使用utils/metrics.py中的标准指标
        mae = MAE(window_pred, window_true)
        mse = MSE(window_pred, window_true)
        rmse = RMSE(window_pred, window_true)
        mape = MAPE(window_pred, window_true)
        corr = CORR(window_pred, window_true)
        r2 = R2(window_pred, window_true)
        rse = RSE(window_pred, window_true)

        # 归一化处理
        # mean_true = np.mean(np.abs(window_true))
        # if mean_true > 0:
        #     # 归一化到[0,1]，值越小越好
        #     norm_mae = min(mae / mean_true, 1)
        #     norm_rmse = min(rmse / mean_true, 1)
        #     norm_mape = min(mape, 1)
        # else:
        #     continue  # 跳过均值为0的窗口

        score = 0
        if r2 <= 1 and r2 > 0:  # R2在(0,1]范围内
            score += r2 * 0.2  # R2越大越好
        if corr <= 1 and corr > 0:  # 相关系数在(0,1]范围内
            score += corr * 0.2  # 相关系数越大越好
        # if norm_mae >= 0:
        #     score += (1 - norm_mae) * 0.2  # MAE越小越好
        # if norm_rmse >= 0:
        #     score += (1 - norm_rmse) * 0.2  # RMSE越小越好
        # if norm_mape >= 0:
        #     score += (1 - norm_mape) * 0.2  # MAPE越小越好

        if score > best_score:
            best_score = score
            best_start_idx = i
            best_metrics = {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'mape': mape * 100,  # 转换为百分比
                'corr': corr,
                'r2': r2,
                'rse': rse,
                'score': score
            }

    return best_start_idx, best_start_idx + window_size, best_metrics


def plot_prediction_vs_true(pred_path, true_path, data_path, save_path=None):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 加载数据
    pred = np.load(pred_path)
    true = np.load(true_path)

    # 确保数据维度正确
    if pred.ndim > 2:
        pred = pred.reshape(-1)
    if true.ndim > 2:
        true = true.reshape(-1)

    # 读取日期
    df = pd.read_csv(data_path)
    dates = pd.to_datetime(df['date'])

    # 修正数据长度匹配问题
    min_len = min(len(dates), len(pred), len(true))
    dates = dates[-min_len:].reset_index(drop=True)  # 重置索引
    pred = pred[-min_len:]
    true = true[-min_len:]

    # 找出最佳拟合段
    start_idx, end_idx, metrics = find_best_fitting_window(pred, true)

    # 绘制全局图
    plt.figure(figsize=(15, 10))

    # 子图1：全局预测
    plt.subplot(2, 1, 1)
    plt.plot(dates, true, 'b-', label='真实值', linewidth=1)
    plt.plot(dates, pred, 'r--', label='预测值', linewidth=1)
    plt.title('全局预测对比')
    plt.xlabel('日期')
    plt.ylabel('数值')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xticks(rotation=45)


    # 子图2：最佳拟合段
    plt.subplot(2, 1, 2)
    # 提取最佳拟合段数据并确保索引正确
    best_dates = dates[start_idx:end_idx].reset_index(drop=True)  # 重要：重置索引
    best_true = true[start_idx:end_idx]
    best_pred = pred[start_idx:end_idx]

    # 将日期转换为数值以便插值
    date_nums = np.arange(len(best_dates))
    # 创建更密集的x点进行插值
    x_smooth = np.linspace(0, len(best_dates) - 1, len(best_dates) * 5)

    # 对真实值和预测值进行平滑插值
    true_smooth = interp1d(date_nums, best_true, kind='cubic', bounds_error=False)(x_smooth)
    pred_smooth = interp1d(date_nums, best_pred, kind='cubic', bounds_error=False)(x_smooth)

    # 对日期进行插值（使用datetime64的数值表示）
    date_values = best_dates.values.astype(np.int64)
    date_interp = interp1d(date_nums, date_values, kind='linear', bounds_error=False)(x_smooth)
    smooth_dates = pd.to_datetime(date_interp)

    # 绘制平滑曲线
    plt.plot(best_dates, best_true, 'bo', alpha=0.3, markersize=2)  # 原始数据点
    plt.plot(best_dates, best_pred, 'ro', alpha=0.3, markersize=2)  # 原始数据点
    plt.plot(smooth_dates, true_smooth, 'b-', label='真实值', linewidth=2)  # 平滑曲线
    plt.plot(smooth_dates, pred_smooth, 'r-', label='预测值', linewidth=2)  # 平滑曲线

    plt.title(f'最佳拟合段')
    plt.xlabel('日期')
    plt.ylabel('数值')
    plt.legend(loc='best')
    plt.grid(True)
    plt.gcf().autofmt_xdate()
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # 打印详细指标
    print(f"\n最佳拟合段信息：")
    print(f"开始时间：{dates[start_idx]}")
    print(f"结束时间：{dates[end_idx - 1]}")
    print(f"R²值：{metrics['r2']:.4f}")
    print(f"相关系数：{metrics['corr']:.4f}")
    print(f"MAE：{metrics['mae']:.4f}")
    print(f"RMSE：{metrics['rmse']:.4f}")
    print(f"MAPE：{metrics['mape']:.4f}%")
    print(f"综合评分：{metrics['score']:.4f}")

    return {
        'dates': dates[start_idx:end_idx],
        'true_values': true[start_idx:end_idx],
        'predicted_values': pred[start_idx:end_idx],
        'metrics': metrics
    }

# 使用示例
best_fit_data = plot_prediction_vs_true(
    'pred.npy',
    'true.npy',
    '../../data/ETT/QianTangRiver2020-2024WorkedFull.csv'
)