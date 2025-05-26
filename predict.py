import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from exp.exp_informer import Exp_Informer
from utils.tools import dotdict
from utils.metrics import metric # 新增导入

# 新增的 test 函数
def test_model_performance(experiment, settings_str, arguments):
    print('>>>>>>>开始测试<<<<<<<<<<<<<<<<<<<<<<<<')
    # 注意：这里的 _get_data 会根据 arguments.batch_size 加载数据
    test_data_obj, test_data_loader = experiment._get_data(flag='test')
    
    if len(test_data_loader) == 0:
        print("警告: 测试数据加载器为空，无法执行测试。请检查测试数据集或批处理大小。")
        print('>>>>>>>测试跳过（无数据）<<<<<<<<<<<<<<<<<<<<<<<<')
        return

    experiment.model.eval()
    
    predictions_list = []
    ground_truths_list = []
    
    with torch.no_grad(): # 测试时不需要计算梯度
        for i, (batch_x_data, batch_y_data, batch_x_timestamps, batch_y_timestamps) in enumerate(test_data_loader):
            prediction_output, ground_truth_output = experiment._process_one_batch(
                test_data_obj, batch_x_data, batch_y_data, batch_x_timestamps, batch_y_timestamps)
            predictions_list.append(prediction_output.detach().cpu().numpy())
            ground_truths_list.append(ground_truth_output.detach().cpu().numpy())

    if not predictions_list:
        print("警告: 未收集到预测结果（测试数据加载后列表仍为空），无法计算指标。")
        print('>>>>>>>测试完成（无预测结果）<<<<<<<<<<<<<<<<<<<<<<<<')
        return

    predictions_arr = np.array(predictions_list)
    ground_truths_arr = np.array(ground_truths_list)
    
    # predictions_arr 的形状是 (num_batches, batch_size, pred_len, c_out)
    # ground_truths_arr 的形状是 (num_batches, batch_size, pred_len, c_out)
    # 我们需要将其 reshape 为 (total_samples, pred_len, c_out)
    predictions_arr = predictions_arr.reshape(-1, predictions_arr.shape[-2], predictions_arr.shape[-1])
    ground_truths_arr = ground_truths_arr.reshape(-1, ground_truths_arr.shape[-2], ground_truths_arr.shape[-1])

    print('测试集形状 (预测值, 真实值):', predictions_arr.shape, ground_truths_arr.shape)

    if predictions_arr.shape[0] != ground_truths_arr.shape[0]:
        print(f"警告: 预测 ({predictions_arr.shape[0]}) 和真实 ({ground_truths_arr.shape[0]}) 样本数量不匹配。跳过指标计算。")
        print('>>>>>>>测试完成（样本数量不匹配）<<<<<<<<<<<<<<<<<<<<<<<<')
        return
    
    if predictions_arr.shape[0] == 0:
        print("警告: 重塑后的预测数组为空（没有有效样本），无法计算指标。")
        print('>>>>>>>测试完成（无有效样本）<<<<<<<<<<<<<<<<<<<<<<<<')
        return

    # 从 metric 函数获取指标
    # metric 函数返回: mae, mse, rmse, mape, mspe, rse, corr, spearman_corr_val, euclidean_dist_val, dtw_dist_val, r2_score, accuracy
    metrics_output = metric(predictions_arr, ground_truths_arr)
    mae, mse, rmse, mape, mspe, rse, corr, spearman_corr_val, euclidean_dist_val, dtw_dist_val, r2, accuracy = metrics_output
    
    print(f'测试集指标:')
    print(f'  RMSE: {rmse:.4f}')
    print(f'  MAE:  {mae:.4f}')
    print(f'  MSE:  {mse:.4f}')
    print(f'  R2:   {r2:.4f}') # r2 对应 metric 函数返回的 r2_score
    print(f'  MAPE: {mape:.4f}')
    
    results_folder_path = './results/' + settings_str + '/'
    if not os.path.exists(results_folder_path):
        os.makedirs(results_folder_path)
    
    np.save(results_folder_path + 'test_set_predictions.npy', predictions_arr)
    np.save(results_folder_path + 'test_set_ground_truths.npy', ground_truths_arr)
    
    test_metrics_summary = {
        'RMSE': rmse, 'MAE': mae, 'MSE': mse, 'R2': r2, 'MAPE': mape
    }
    with open(results_folder_path + 'test_set_metrics_summary.txt', 'w') as f:
        for metric_name, metric_value in test_metrics_summary.items():
            f.write(f'{metric_name}: {metric_value:.4f}\n')
    print(f'测试集预测和指标已保存至: {results_folder_path}')
    print('>>>>>>>测试完成<<<<<<<<<<<<<<<<<<<<<<<<')

# 设置参数
args = dotdict()

# 基本配置
args.model = 'informer'  # 模型名称
args.data = 'qiantangjiang'  # 数据集名称
args.root_path = './data/ETT/'  # 数据集根目录
args.data_path = 'qiantangjiang.csv'  # 数据文件名称
args.features = 'MS'  # 预测任务，选项：M（多变量预测多变量），S（单变量预测单变量），MS（多变量预测单变量）
args.target = 'O2'  # 目标列名（针对 features='MS' 或 'S'）
args.freq = 'h'  # 时间频率，选项：s（秒），t（分钟），h（小时），d（天），b（工作日），w（周），m（月）
args.detail_freq = 'h'  # 详细时间频率，用于预测
args.checkpoints = './checkpoints/'  # 模型检查点保存位置

# 预测任务配置
args.seq_len = 96  # 输入序列长度
args.label_len = 48  # 起始标记长度
args.pred_len = 24  # 预测序列长度

# 模型参数
args.enc_in = 5  # 编码器输入维度 (根据 qiantangjiang 数据集)
args.dec_in = 5  # 解码器输入维度 (根据 qiantangjiang 数据集)
args.c_out = 1  # 输出维度 (根据 features='MS')
args.d_model = 128  # 模型维度，调整为与权重文件一致
args.n_heads = 8  # 注意力头数
args.e_layers = 2  # 编码器层数
args.d_layers = 1  # 解码器层数
args.d_ff = 2048  # 前馈网络维度
args.factor = 5  # 注意力因子
args.padding = 0  # 填充方式
args.distil = True  # 是否使用蒸馏操作
args.dropout = 0.3  # dropout 率
args.attn = 'prob'  # 注意力机制，选项：prob（概率注意力），full（全注意力）
args.embed = 'timeF'  # 时间特征编码，选项：timeF（固定），learned（可学习）
args.activation = 'relu'  # 激活函数，选项：relu，gelu
args.output_attention = False  # 是否输出注意力
args.mix = True  # 是否混合注意力
args.inverse = False  # 是否反转输出

# 优化参数
args.num_workers = 0  # 数据加载器的工作线程数
args.train_epochs = 1000  # 训练轮数
args.batch_size = 256  # 批次大小
args.patience = 30  # 早停耐心值
args.learning_rate = 0.01  # 学习率
args.loss = 'mse'  # 损失函数
args.lradj = 'type1'  # 学习率调整方式

# GPU 设置
args.use_gpu = True  # 不使用 GPU，强制使用 CPU
args.gpu = 0  # GPU 设备 ID（不使用）
args.use_multi_gpu = False  # 不使用多 GPU
args.devices = '0,1,2,3'  # 多 GPU 设备 ID（不使用）

# 设置保存文件夹
setting = f'{args.model}_{args.data}_ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}_at{args.attn}_fc{args.factor}_eb{args.embed}_dt{args.distil}_mx{args.mix}'
print(f'使用设置: {setting}')

# 初始化实验
exp = Exp_Informer(args)

# 执行预测
print('>>>>>>>开始预测<<<<<<<<<<<<<<<<<<<<<<<<')
# 加载特定权重文件
checkpoint_path = '/Users/andyhuang/Desktop/r-informer/checkpoints/checkpoint.pth'
if os.path.exists(checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    # 去除 DataParallel 前缀 'module.'，以便在CPU设备加载权重
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # 去掉 'module.' 前缀
        else:
            new_state_dict[k] = v
    exp.model.load_state_dict(new_state_dict)
    print(f'已加载权重文件: {checkpoint_path}')
else:
    print(f'权重文件 {checkpoint_path} 未找到，使用默认权重')

# 获取预测结果
pred_data, pred_loader = exp._get_data(flag='pred')
exp.model.eval()
preds = []

for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
    pred, true = exp._process_one_batch(pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
    preds.append(pred.detach().cpu().numpy())

preds = np.array(preds)
preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

# 保存预测结果
folder_path = './results/' + setting + '/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
np.save(folder_path + 'real_prediction.npy', preds)
print(f'预测结果已保存至: {folder_path}real_prediction.npy')

# 绘制真实值与预测值的曲线图
# 假设我们只绘制第一个样本的前几个预测点
sample_idx = 0
true_values = true.detach().cpu().numpy().reshape(-1, true.shape[-2], true.shape[-1])[sample_idx]
pred_values = preds[sample_idx]

plt.figure(figsize=(10, 6))
plt.plot(range(len(true_values)), true_values, label='True Values', color='blue')
plt.plot(range(len(true_values) - len(pred_values), len(true_values)), pred_values, label='Predicted Values', color='red')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.title('True vs Predicted Values')
plt.legend()
plt.grid(True)
plt.savefig(folder_path + 'prediction_plot.png')
plt.close()
print(f'预测曲线图已保存至: {folder_path}prediction_plot.png')

print('>>>>>>>预测完成<<<<<<<<<<<<<<<<<<<<<<<<')

# 执行测试
# 确保 exp, setting, 和 args 在此作用域内可用
test_model_performance(exp, setting, args)