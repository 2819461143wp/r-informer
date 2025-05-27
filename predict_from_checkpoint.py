import argparse
import os
import torch
from exp.exp_informer import Exp_Informer

def predict_from_checkpoint():
    parser = argparse.ArgumentParser(description='从checkpoint生成预测结果')
    
    # 基本参数
    parser.add_argument('--model', type=str, default='informer',
                        help='模型类型，options: [informer, informerstack]')
    parser.add_argument('--data', type=str, default='custom', help='数据集名称')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='数据根路径')
    parser.add_argument('--data_path', type=str, default='QianTangRiver2020-2024WorkedFull.csv', help='数据文件名')
    parser.add_argument('--features', type=str, default='MS',
                        help='预测任务类型，options:[M, S, MS]; M:多变量预测多变量, S:单变量预测单变量, MS:多变量预测单变量')
    parser.add_argument('--target', type=str, default='O2', help='目标特征')
    parser.add_argument('--freq', type=str, default='h',
                        help='时间特征编码频率，options:[s:秒, t:分钟, h:小时, d:天, b:工作日, w:周, m:月]')
    parser.add_argument('--checkpoints', type=str, required=True, help='checkpoint目录路径')

    # 模型参数
    parser.add_argument('--seq_len', type=int, default=96, help='输入序列长度')
    parser.add_argument('--label_len', type=int, default=48, help='解码器起始标记长度')
    parser.add_argument('--pred_len', type=int, default=24, help='预测序列长度')
    parser.add_argument('--enc_in', type=int, default=5, help='编码器输入大小')
    parser.add_argument('--dec_in', type=int, default=5, help='解码器输入大小')
    parser.add_argument('--c_out', type=int, default=5, help='输出大小')
    parser.add_argument('--d_model', type=int, default=128, help='模型维度')
    parser.add_argument('--n_heads', type=int, default=8, help='多头注意力头数')
    parser.add_argument('--e_layers', type=int, default=2, help='编码器层数')
    parser.add_argument('--d_layers', type=int, default=1, help='解码器层数')
    parser.add_argument('--s_layers', type=str, default='3,2,1', help='堆叠编码器层数')
    parser.add_argument('--d_ff', type=int, default=2048, help='前馈网络维度')
    parser.add_argument('--factor', type=int, default=5, help='ProbSparse注意力因子')
    parser.add_argument('--padding', type=int, default=0, help='填充类型')
    parser.add_argument('--distil', action='store_false',
                    help='是否在编码器中使用蒸馏', default=True)
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout率')
    parser.add_argument('--attn', type=str, default='prob', help='注意力类型，options:[prob, full]')
    parser.add_argument('--embed', type=str, default='timeF',
                    help='时间特征编码，options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='relu', help='激活函数')
    parser.add_argument('--output_attention', action='store_true', help='是否输出注意力')
    parser.add_argument('--mix', action='store_false', help='在生成解码器中使用混合注意力', default=True)
    
    # 数据处理参数
    parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载器工作进程数')
    parser.add_argument('--batch_size', type=int, default=256, help='批次大小')
    parser.add_argument('--inverse', action='store_true', help='反转输出数据', default=False)
    
    # GPU相关参数
    parser.add_argument('--use_gpu', type=bool, default=True, help='是否使用GPU')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--use_multi_gpu', action='store_true', help='是否使用多GPU')
    parser.add_argument('--devices', type=str, default='0', help='多GPU设备ID')

    parser.add_argument('--use_amp', action='store_true', 
                        help='是否使用自动混合精度', default=False)

    args = parser.parse_args()

    # 处理cols参数
    if args.cols:
        args.cols = args.cols.split(',')
    
    # GPU设置
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu:
        print(f"Use GPU: {torch.cuda.get_device_name(args.gpu)}")
    
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # 处理s_layers参数
    args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ', '').split(',')]
    
    # 设置detail_freq
    args.detail_freq = args.freq
    args.freq = args.freq[-1:]

    # 创建实验实例
    exp = Exp_Informer(args)

    # 设置路径
    setting = f"{args.model}_{args.data}_ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}"
    checkpoint_path = os.path.join(args.checkpoints, setting, 'checkpoint.pth')
    
    # 确认checkpoint文件存在
    if not os.path.exists(checkpoint_path):
        print(f"找不到checkpoint文件: {checkpoint_path}")
        # 尝试直接在指定目录下查找
        alt_path = os.path.join(args.checkpoints, 'checkpoint.pth')
        if os.path.exists(alt_path):
            print(f"找到替代checkpoint文件: {alt_path}")
            checkpoint_path = alt_path
            setting = os.path.basename(os.path.dirname(args.checkpoints))
        else:
            raise FileNotFoundError(f"无法找到有效的checkpoint文件")
            
    print(f'Loading model from {checkpoint_path}')
    
    # 加载模型参数
    # 改为
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if args.use_gpu else 'cpu'))
    # 创建新的state_dict来移除'module.'前缀
    new_state_dict = {}
    for k, v in checkpoint.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    # 加载修改后的state_dict
    exp.model.load_state_dict(new_state_dict)
    
    # 执行测试并生成pred.npy和true.npy
    exp.test(setting)
    
    print(f"预测结果已保存到 ./results/{setting}/")
    print(f"pred.npy 和 true.npy 文件已生成")

if __name__ == "__main__":
    predict_from_checkpoint()