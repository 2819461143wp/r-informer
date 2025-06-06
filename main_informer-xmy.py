import argparse
import os
import torch
from sklearn.model_selection import ParameterGrid
from exp.exp_informer import Exp_Informer


# 封装一个类，使其符合 scikit-learn 风格
class InformerWrapper:
    def __init__(self, args):
        self.args = args
        self.best_loss = float('inf')
        self.best_args = None

    def fit(self, X=None, y=None):
        Exp = Exp_Informer
        for params in ParameterGrid(self.get_param_grid()):
            current_args = self.args.copy()
            current_args.update(params)
            current_args = self._process_args(current_args)

            setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(
                current_args['model'], current_args['data'], current_args['features'],
                current_args['seq_len'], current_args['label_len'], current_args['pred_len'],
                current_args['d_model'], current_args['n_heads'], current_args['e_layers'],
                current_args['d_layers'], current_args['d_ff'], current_args['attn'],
                current_args['factor'], current_args['embed'], current_args['distil'],
                current_args['mix'], current_args['des'], 0)

            exp = Exp(current_args)
            exp.train(setting)
            loss = exp.test(setting)

            if loss < self.best_loss:
                self.best_loss = loss
                self.best_args = current_args

    def get_param_grid(self):
        param_grid = {
            'd_model': [256, 512],
            'n_heads': [4, 8],
            'e_layers': [1, 2]
        }
        return param_grid

    def _process_args(self, args):
        args['use_gpu'] = True if torch.cuda.is_available() and args['use_gpu'] else False

        if args['use_gpu'] and args['use_multi_gpu']:
            args['devices'] = args['devices'].replace(' ', '')
            device_ids = args['devices'].split(',')
            args['device_ids'] = [int(id_) for id_ in device_ids]
            args['gpu'] = args['device_ids'][0]

        data_parser = {
            'ETTh1': {'data': 'ETTh1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
            'ETTh2': {'data': 'ETTh2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
            'ETTm1': {'data': 'ETTm1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
            'ETTm2': {'data': 'ETTm2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
            'WTH': {'data': 'WTH.csv', 'T': 'WetBulbCelsius', 'M': [12, 12, 12], 'S': [1, 1, 1], 'MS': [12, 12, 1]},
            'ECL': {'data': 'ECL.csv', 'T': 'MT_320', 'M': [321, 321, 321], 'S': [1, 1, 1], 'MS': [321, 321, 1]},
            'Solar': {'data': 'solar_AL.csv', 'T': 'POWER_136', 'M': [137, 137, 137], 'S': [1, 1, 1], 'MS': [137, 137, 1]},
        }
        if args['data'] in data_parser.keys():
            data_info = data_parser[args['data']]
            args['data_path'] = data_info['data']
            args['target'] = data_info['T']
            args['enc_in'], args['dec_in'], args['c_out'] = data_info[args['features']]

        args['s_layers'] = [int(s_l) for s_l in args['s_layers'].replace(' ', '').split(',')]
        args['detail_freq'] = args['freq']
        args['freq'] = args['freq'][-1:]

        return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')

    # 原有的参数定义
    parser.add_argument('--model', type=str, required=True, default='informer',
                        help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='data')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length of Informer encoder')
    parser.add_argument('--label_len', type=int, default=48, help='start token length of Informer decoder')
    parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
    parser.add_argument('--padding', type=int, default=0, help='padding type')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
    parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=1, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()
    args_dict = vars(args)

    wrapper = InformerWrapper(args_dict)
    wrapper.fit()

    print("Best loss:", wrapper.best_loss)
    print("Best parameters:", wrapper.best_args)

    if args.do_predict:
        best_exp = Exp_Informer(wrapper.best_args)
        setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(
            wrapper.best_args['model'], wrapper.best_args['data'], wrapper.best_args['features'],
            wrapper.best_args['seq_len'], wrapper.best_args['label_len'], wrapper.best_args['pred_len'],
            wrapper.best_args['d_model'], wrapper.best_args['n_heads'], wrapper.best_args['e_layers'],
            wrapper.best_args['d_layers'], wrapper.best_args['d_ff'], wrapper.best_args['attn'],
            wrapper.best_args['factor'], wrapper.best_args['embed'], wrapper.best_args['distil'],
            wrapper.best_args['mix'], wrapper.best_args['des'], 0)
        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        best_exp.predict(setting, True)

    torch.cuda.empty_cache()