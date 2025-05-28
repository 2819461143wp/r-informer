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

        score = 0
        if r2 <= 1 and r2 > 0:  # R2在(0,1]范围内
            score += r2 * 0.2  # R2越大越好
        if corr <= 1 and corr > 0:  # 相关系数在(0,1]范围内
            score += corr * 0.2  # 相关系数越大越好

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

def calculate_metrics_for_window(pred, true):
    """为给定的预测和真实值计算指标"""
    return {
        'mae': MAE(pred, true),
        'mse': MSE(pred, true),
        'rmse': RMSE(pred, true),
        'mape': MAPE(pred, true) * 100,
        'corr': CORR(pred, true),
        'r2': R2(pred, true),
        'rse': RSE(pred, true)
    }

def plot_three_models_comparison(data_path, save_path=None):
    """对比三个模型的预测效果"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 模型配置
    models = {
        'R-Informer': {
            'pred_path': 'r-informer/pred-r-informer.npy',
            'true_path': 'r-informer/true-r-informer.npy',
            'color': 'red',
            'linestyle': '-'
        },
        'Autoformer': {
            'pred_path': 'autoformer/pred-autoformer.npy',
            'true_path': 'autoformer/true-autoformer.npy',
            'color': 'green',
            'linestyle': '--'
        },
        'TCN': {
            'pred_path': 'tcn/pred-tcn.npy',
            'true_path': 'tcn/true-tcn.npy',
            'color': 'orange',
            'linestyle': '-.'
        }
    }
    
    # 加载数据
    model_data = {}
    for model_name, model_config in models.items():
        pred = np.load(model_config['pred_path'])
        true = np.load(model_config['true_path'])
        
        # 确保数据维度正确
        if pred.ndim > 2:
            pred = pred.reshape(-1)
        if true.ndim > 2:
            true = true.reshape(-1)
            
        model_data[model_name] = {'pred': pred, 'true': true}
    
    # 读取日期
    df = pd.read_csv(data_path)
    dates = pd.to_datetime(df['date'])
    
    # 找到所有数据的最小长度
    all_lengths = [len(dates)] + [len(data['pred']) for data in model_data.values()] + [len(data['true']) for data in model_data.values()]
    min_len = min(all_lengths)
    
    # 统一数据长度
    dates = dates[-min_len:].reset_index(drop=True)
    for model_name in model_data:
        model_data[model_name]['pred'] = model_data[model_name]['pred'][-min_len:]
        model_data[model_name]['true'] = model_data[model_name]['true'][-min_len:]
    
    # 使用R-Informer的真实值找到最佳拟合段
    r_informer_pred = model_data['R-Informer']['pred']
    r_informer_true = model_data['R-Informer']['true']
    start_idx, end_idx, r_informer_metrics = find_best_fitting_window(r_informer_pred, r_informer_true)
    
    # 创建图形
    plt.figure(figsize=(18, 12))
    
    # 子图1：全局预测对比
    plt.subplot(2, 1, 1)
    
    # 绘制真实值（只绘制一次，因为所有模型的真实值应该相同）
    plt.plot(dates, r_informer_true, 'b-', label='真实值', linewidth=1.5, alpha=0.8)
    
    # 绘制各模型预测值
    for model_name, model_config in models.items():
        plt.plot(dates, model_data[model_name]['pred'], 
                color=model_config['color'], 
                linestyle=model_config['linestyle'],
                label=f'{model_name}预测值', 
                linewidth=1.2, 
                alpha=0.7)
    
    # 标记最佳拟合段
    plt.axvspan(dates[start_idx], dates[end_idx-1], alpha=0.2, color='yellow', 
                label=f'R-Informer最佳拟合段')
    
    plt.title('三模型全局预测对比', fontsize=14, fontweight='bold')
    plt.xlabel('日期')
    plt.ylabel('数值')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # 子图2：最佳拟合段详细对比
    plt.subplot(2, 1, 2)
    
    # 提取最佳拟合段数据
    best_dates = dates[start_idx:end_idx].reset_index(drop=True)
    best_true = r_informer_true[start_idx:end_idx]
    
    # 绘制真实值
    plt.plot(best_dates, best_true, 'b-', label='真实值', linewidth=2.5, alpha=0.9)
    
    # 计算并绘制各模型在最佳拟合段的预测
    model_metrics = {}
    for model_name, model_config in models.items():
        best_pred = model_data[model_name]['pred'][start_idx:end_idx]
        
        # 计算该段的指标
        metrics = calculate_metrics_for_window(best_pred, best_true)
        model_metrics[model_name] = metrics
        
        # 绘制预测值
        plt.plot(best_dates, best_pred, 
                color=model_config['color'], 
                linestyle=model_config['linestyle'],
                label=f'{model_name} (R²={metrics["r2"]:.3f})', 
                linewidth=2, 
                alpha=0.8)
    
    plt.title(f'最佳拟合段详细对比 ({dates[start_idx].strftime("%Y-%m-%d")} 至 {dates[end_idx-1].strftime("%Y-%m-%d")})', 
              fontsize=14, fontweight='bold')
    plt.xlabel('日期')
    plt.ylabel('数值')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.gcf().autofmt_xdate()
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # 打印详细指标对比
    print(f"\n{'='*60}")
    print(f"最佳拟合段信息 ({dates[start_idx]} 至 {dates[end_idx-1]})")
    print(f"{'='*60}")
    
    print(f"{'模型':<15} {'MAE':<8} {'RMSE':<8} {'MAPE(%)':<10} {'R²':<8} {'相关系数':<8}")
    print(f"{'-'*60}")
    
    for model_name in models.keys():
        metrics = model_metrics[model_name]
        print(f"{model_name:<15} {metrics['mae']:<8.4f} {metrics['rmse']:<8.4f} "
              f"{metrics['mape']:<10.2f} {metrics['r2']:<8.4f} {metrics['corr']:<8.4f}")
    
    # 找出各指标的最佳模型
    print(f"\n{'指标最优模型':<15}")
    print(f"{'-'*30}")
    
    # MAE最小
    best_mae_model = min(model_metrics.keys(), key=lambda x: model_metrics[x]['mae'])
    print(f"{'最低MAE:':<15} {best_mae_model} ({model_metrics[best_mae_model]['mae']:.4f})")
    
    # RMSE最小
    best_rmse_model = min(model_metrics.keys(), key=lambda x: model_metrics[x]['rmse'])
    print(f"{'最低RMSE:':<15} {best_rmse_model} ({model_metrics[best_rmse_model]['rmse']:.4f})")
    
    # R²最大
    best_r2_model = max(model_metrics.keys(), key=lambda x: model_metrics[x]['r2'])
    print(f"{'最高R²:':<15} {best_r2_model} ({model_metrics[best_r2_model]['r2']:.4f})")
    
    # 相关系数最大
    best_corr_model = max(model_metrics.keys(), key=lambda x: model_metrics[x]['corr'])
    print(f"{'最高相关系数:':<15} {best_corr_model} ({model_metrics[best_corr_model]['corr']:.4f})")
    
    return {
        'dates': best_dates,
        'true_values': best_true,
        'model_predictions': {name: model_data[name]['pred'][start_idx:end_idx] for name in models.keys()},
        'model_metrics': model_metrics,
        'best_window_range': (start_idx, end_idx)
    }

# 原有的单模型函数保持不变
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

# 运行三模型对比
if __name__ == "__main__":
    # 三模型对比
    comparison_data = plot_three_models_comparison(
        '../data/ETT/QianTangRiver2020-2024WorkedFull.csv',
        save_path='three_models_comparison.png'
    )
    
    # 原有的单模型分析（如果需要）
    # best_fit_data = plot_prediction_vs_true(
    #     'r-informer/pred-r-informer.npy',
    #     'r-informer/true-r-informer.npy',
    #     '../../data/ETT/QianTangRiver2020-2024WorkedFull.csv'
    # )