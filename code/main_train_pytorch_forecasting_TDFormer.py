# -*- coding: utf-8 -*-
import os
import json
import time
import hashlib
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from shutil import copyfile
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder,EncoderNormalizer,MultiNormalizer,TorchNormalizer
from pytorch_forecasting.metrics import MAE, RMSE, MultiHorizonMetric, MultiLoss, QuantileLoss
from pytorch_forecasting.data import GroupNormalizer
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('agg')

import argparse
parser = argparse.ArgumentParser(description='PyTorch Forecasting')
parser.add_argument('--model_name', type=str,default='TDFormer', metavar='M', 
                    help='iTransModel ModernTCNModel PatchTSTModel  PathFormerModel')

parser.add_argument('--gpu', type=str, default="1", metavar='N',
                    help='选择的gpu')
parser.add_argument('--df', type=str, default="data/processed/spring.csv", metavar='N',
                    help=" ")
parser.add_argument('--select_station', type=str, default='f', metavar='N',
                    help='FJ')
parser.add_argument('--select_season', type=int, default=1, metavar='N',
                    help='选择的季节')

parser.add_argument('--inp_length', type=int, default=24, metavar='N',
                    help='输入序列长度')
parser.add_argument('--outp_length', type=int, default=48, metavar='N',
                    help='预测长度')

parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help=' ')
parser.add_argument('--ablation', type=str, default='full',
                    choices=['full', 'no_variable', 'no_feature', 'no_temporal'],
                    help='TDFormer ablation variant')
parser.add_argument('--num_runs', type=int, default=1,
                    help='number of repeated runs for this variant')
parser.add_argument('--max_epochs', type=int, default=100,
                    help='maximum training epochs per run')
parser.add_argument('--seed', type=int, default=2024,
                    help='random seed shared by all ablation variants')
parser.add_argument('--run_hpo', action='store_true',
                    help='run the original Optuna search after normal training')
parser.add_argument('--metrics_json', type=str, default='',
                    help='optional path for machine-readable evaluation metrics')
parser.add_argument('--dataset_profile', type=str, default='wind',
                    choices=['original', 'wind'], help='dataset-specific preprocessing profile')
parser.add_argument('--target_capacity', type=float, default=300.0,
                    help='rated capacity used to normalize wind power')
parser.add_argument('--year_start', type=str, default='2022-01-03 00:00:00', help='optional inclusive dataset window start')
parser.add_argument('--year_end', type=str, default='2023-01-03 00:00:00', help='optional exclusive dataset window end')
parser.add_argument('--ablation_design', type=str, default='original',
                    choices=['original', 'residual_gated', 'residual_gated_v2'])
parser.add_argument('--gate_init', type=float, default=0.05)
parser.add_argument('--variable_temperature', type=float, default=0.7)
parser.add_argument('--variable_gate_init', type=float, default=0.1)
parser.add_argument('--module_ablation', type=str, default='full',
                    choices=['none', 'full', 'wo_tan', 'wo_fan', 'wo_van'],
                    help='whole-module removal; wo_van replaces the complete VSN with a flat linear projection')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--hidden_continuous_size', type=int, default=32)
parser.add_argument('--attention_head_size', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--gradient_clip_val', type=float, default=0.5)
parser.add_argument('--lr_scheduler_patience', type=int, default=10)
args = parser.parse_args()
if args.hidden_size % args.attention_head_size != 0:
    parser.error('hidden_size must be divisible by attention_head_size')
if not 0 <= args.dropout < 1 or args.learning_rate <= 0:
    parser.error('Require dropout in [0, 1) and a positive learning_rate')
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
pl.seed_everything(args.seed, workers=True)
run_started_at = time.time()
'''
可以切换4个季节的数据；
'''
# 自定义损失函数【不统计target=0的样本，非常重要，因为存在大量缺测数据】
from pytorch_forecasting.metrics.base_metrics import MultiHorizonMetric
class CustomLoss(MultiHorizonMetric):
    def loss(self, y_pred, target):
        mask = target != 0
        loss = torch.zeros_like(target)
        loss[mask] = (self.to_prediction(y_pred)[mask] - target[mask]).abs()
        return loss


# 打印日志到文件
def print_log(print_string, log_file, visible=True):
    """打印并写入指定文件.
           Arguments:
               print_string (string): 打印的字符串
               log_file (FILE *):  写入的文件指针

           Returns:
               nothing
    """
    if visible:
        print("{}".format(print_string))
    # 写入日志文件
    log_file.write('{}\n'.format(print_string))
    # 刷新缓存区，将数据写入
    log_file.flush()

#------------------------------------------------- step 1: 数据准备 ----------------------------------------
# -------------------------------数据分析和数据预处理【至关重要】---------------------------------------
# load data 【选择】
df = pd.read_csv(args.df, encoding='gbk')

# 【极其重要】【可选：FJ】
select_station=args.select_station
print(df.keys())

#将缺失值替换为np.nan,并转数据类型为float.
if args.dataset_profile == 'wind':
    station_mask = df['站点编号'].astype(str).str.lower() == select_station.lower()
    df = df[station_mask].copy()
    if len(df) == 0:
        raise ValueError('No wind rows found for station {}'.format(select_station))
    df['出力(MW)'] = pd.to_numeric(df['出力(MW)'].replace('<NULL>', np.nan), errors='coerce')
    df['出力(MW)'] = df['出力(MW)'].interpolate(limit_direction='both').clip(lower=0)
    df['Time stamp'] = pd.to_datetime(df['时间'])
    df = df.sort_values('Time stamp')
    df['datetime_column'] = df['Time stamp']
    df.set_index('datetime_column', inplace=True)
    target_capacity = args.target_capacity if args.target_capacity > 0 else 300.0
    df['target'] = df['出力(MW)'] / (target_capacity + 1e-6)
    if args.year_start:
        df = df[df.index >= pd.Timestamp(args.year_start)]
    if args.year_end:
        df = df[df.index < pd.Timestamp(args.year_end)]
elif 'FJ' in select_station:
    df['出力(MW)']=df['出力(MW)'].replace('<NULL>', np.nan).astype(np.float32)
    #用线性插值的方式填充缺失值  【非常重要将nan值插值处理】
    df['出力(MW)'] = df['出力(MW)'].interpolate()
    df['出力(MW)'][df['出力(MW)']<0]=0
    df['Time stamp']=pd.to_datetime(df['datetime'], format='%Y/%m/%d %H:%M')
    # df.set_index('Time stamp', inplace=True)
    df['datetime_column'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M')
    df.set_index('datetime_column', inplace=True)
else:
    df['实际功率']=df['实际功率'].replace('<NULL>', np.nan).astype(np.float32)
    #用线性插值的方式填充缺失值  【非常重要将nan值插值处理】
    df['实际功率'] = df['实际功率'].interpolate()
    df['实际功率'][df['实际功率']<0]=0

    df['Time stamp']=pd.to_datetime(df['Datetime'], format='%Y/%m/%d %H:%M')
    # df.set_index('Time stamp', inplace=True)
    df['datetime_column'] = pd.to_datetime(df['Datetime'], format='%Y-%m-%d %H:%M')
    df.set_index('datetime_column', inplace=True)

# 增加一列，站点编号的字典映射
if args.dataset_profile == 'wind':
    pass
elif 'FJ' in select_station:
    df['target'] = df['出力(MW)'] / (48 + 1e-6)



#  ------------------------------------- 数据集确定 2020年---------------------------------
if args.dataset_profile == 'wind':
    pass
elif 'FJ' in select_station:
    df = df[(df.index > '2022-01-03 00:00:00') & (df.index < '2023-01-03 00:00:00')]


# -------------------------------------------  日期特征 ----------------------------------------
df['day_of_week']=df['Time stamp'].dt.dayofweek
df['day']=df['Time stamp'].dt.dayofweek
# 月
df['month']=df['Time stamp'].dt.month
# 季节        三月到五月为春季；六月到八月为夏季；九月到十一月为秋季；十二月到第二年的二月为冬季。
# spring summer fall  winter
df['season']=df['Time stamp'].dt.month.map({3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3,12:4,1:4,2:4})

df["day_of_month"]=df['Time stamp'].dt.day
# 小时
df['hour'] = df['Time stamp'].dt.hour
# head
print('df.head():\n\n',df.head())
# 获取统计数据（均值、方差、最大值、最小值、分位数（25%，50%，75%））    Pandas显示小数位数：pd.round(2)
describe=df.describe().round(2) # .round(2)显示2位有效数字
print('df.describe():\n\n',describe)

for key in df.keys():
    try:
        print('Nan process for {}'.format(key))
        if key in ["Datetime", 'Time stamp', 'time_idx', '日期', '时间', '站点编号']:
            continue
        # 将缺失值替换为np.nan,并转数据类型为float.
        df[key] = df[key].replace('<NULL>', np.nan).astype(float)
        df[key] = df[key].replace('None', np.nan).astype(float)
        df[key] = df[key].replace('NA', np.nan).astype(float)
        # 用线性插值的方式填充缺失值  【非常重要将nan值插值处理】
        df[key] = df[key].interpolate()
        # 对taregt进行插值，但不存入输入变量中
        # if key!='target':
        #     time_varying_known_reals.append(key)
    except:
        print('Error Nan process for {}'.format(key))
        continue


# 检查和验证
print('df.index.is_unique:{}'.format(df.index.is_unique))
# 检查重复索引
duplicate_indices = df[df.index.duplicated()]
print('\nduplicate_indices:\n',duplicate_indices)
# 删除重复索引行
df = df.drop_duplicates()
# 重新设置索引
df = df.reset_index(drop=True)
# 检查和验证
print('df.index.is_unique:{}'.format(df.index.is_unique))

# --------------- 数据筛选 -------------------------------

# ***------季节选择------***
season_list=['Spring', 'Summer', 'Autumn', 'Winter']
# 春季
select_season=args.select_season
mask_season=df['season']==select_season
mask_len=np.sum(mask_season)

print('train dataset:{},val:{},test:{}'.format(0.8*mask_len, 0.1*mask_len, 0.1*mask_len))
# val_mask
# test_mask
df['constant']=df['target']*0
# convert data type column Timestamp from object to datetime
# df['Time stamp'] = pd.to_datetime(df['Time stamp'], format='%b %d, %I:%M %p').dt.strftime('%b %d %H:%M:%S.%f')
# 获取时序递增的time_idx
df['time_idx']=df.index.to_series()



if args.dataset_profile == 'wind':
    time_varying_known_reals = [
        '气压(Pa）', '相对湿度（%）', '云量', '10米风速（10m/s）', '10米风向（°)',
        '温度（K）', '辐照强度（J/m2）', '降水（m）', '100m风速（100m/s）', '100m风向（°)'
    ]
else:
    time_varying_known_reals=['10米高度风速 (米/秒)', '10米高度风向 (度)', '50米高度风速 (米/秒)','50米高度风向 (度)',
                              '地面气压 (千帕)', '2米高度相对湿度 (%)', '2米高度温度 (摄氏度)', '修正降水量 (毫米/小时)']  # ,'全天空地表短波直接法向辐射 (瓦时/平方米)'
# 去除异常值
for variable in time_varying_known_reals:
    df[variable][np.isnan(np.array(df[variable]))] = 0
    df[variable][np.isinf(np.array(df[variable]))] = 0

time_varying_unknown_reals = ['target']
df['target'][np.isnan(np.array(df['target']))] = 0
df['target'][np.isinf(np.array(df['target']))] = 0

# --------------------- 历史长度和预测长度 -------------------------
max_prediction_length = args.outp_length
max_encoder_length = args.inp_length

# ---(5)构建数据集---
'''
训练集：季节*0.8
验证集：季节*0.1
测试集：季节*0.1
'''
training = TimeSeriesDataSet(
        df[mask_season][0:int(0.8*mask_len)],  # 行表样本，列表特征 (18900,31), 54*350=18900, 6*350=2100, 60*350=21000
        time_idx="time_idx",  # 【时间索引字段，确定样本的顺序】
        target='target', #['System power generated | (kW)','Wind speed | (m/s)'],  # 【发电量】
        # 分成了组
        group_ids=['constant'],  # 【无】
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,  # 【构建数据集时的最小预测长度，1】
        max_prediction_length=max_prediction_length,  # 【构建数据集时的最小预测长度，6】

        # 特征（除了target外的）
        time_varying_known_reals=time_varying_known_reals,
        # 未来未知
        time_varying_unknown_reals=time_varying_unknown_reals, # 【随时间变化且在未来未知的连续变量】
        #  # use softplus and normalize by group  【如何计算】   Softplus函数可以看作是ReLU函数的平滑。Softplus(x)=log(1+e^x)
        target_normalizer=TorchNormalizer(),#MultiNormalizer([TorchNormalizer(),TorchNormalizer()]),   # EncoderNormalizer(), TorchNormalizer()
        # add_relative_time_idx=True,
        # add_target_scales=True,
        add_encoder_length=False,
        allow_missing_timesteps=True
    )

batch_size = args.batch_size  # set this between 32 to 128
# 从数据集中获取数据加载器    # train ( bool , optional ) -- 如果数据加载器用于训练或预测，如果为真，将打乱并丢弃最后一批
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
# ---  保存文件名，设置特征数量【必须】 -------
for x, (y, weight) in iter(train_dataloader):
    print("['encoder_cont']:", x['encoder_cont'].shape)  # [batch_size, max_encoder_length, 5]
    print("['decoder_cont']:", x['decoder_cont'].shape)  # [batch_size, max_prediction_length, 15]
    en_feats_num = x['encoder_cat'].shape[-1] + x['encoder_cont'].shape[-1]
    de_feats_num = x['decoder_cat'].shape[-1] + x['decoder_cont'].shape[-1]
    print('y:', y[0].shape)     #
    # print('weight:',weight.shape)
    break


validing = TimeSeriesDataSet(
    # df[lambda x: x.day_of_month >= 27],              # 行表样本，列表特征 (18900,31), 54*350=18900, 6*350=2100, 60*350=21000
    df[mask_season][int(0.8 * mask_len):int(0.9 * mask_len)],
    time_idx="time_idx",  # 【时间索引字段，确定样本的顺序】
    target='target',  # ['System power generated | (kW)','Wind speed | (m/s)'],  # 【发电量】
    # 分成了组
    group_ids=['constant'],  # 【无】
    min_encoder_length=max_encoder_length,
    max_encoder_length=max_encoder_length,
    min_prediction_length=max_prediction_length,  # 【构建数据集时的最小预测长度，1】
    max_prediction_length=max_prediction_length,  # 【构建数据集时的最小预测长度，6】

    time_varying_known_reals=time_varying_known_reals,
    # 未来未知
    time_varying_unknown_reals=time_varying_unknown_reals,  # 【随时间变化且在未来未知的连续变量】
    #  # use softplus and normalize by group  【如何计算】   Softplus函数可以看作是ReLU函数的平滑。Softplus(x)=log(1+e^x)
    target_normalizer=TorchNormalizer(),
    # MultiNormalizer([TorchNormalizer(),TorchNormalizer()]),   # EncoderNormalizer(), TorchNormalizer()
    # add_relative_time_idx=True,
    # add_target_scales=True,
    add_encoder_length=False,
    allow_missing_timesteps=True
)

#  从一个数据集中抽取出一个验证集 （按照留1法抽取）
# validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)

# ---(6)组织数据加载器---

val_dataloader = validing.to_dataloader(train=False, batch_size=batch_size, num_workers=0)


testing= TimeSeriesDataSet(
    # df[lambda x: x.day_of_month >= 27],              # 行表样本，列表特征 (18900,31), 54*350=18900, 6*350=2100, 60*350=21000
    df[mask_season][int(0.9 * mask_len):],
    time_idx="time_idx",  # 【时间索引字段，确定样本的顺序】
    target='target',  # ['System power generated | (kW)','Wind speed | (m/s)'],  # 【发电量】
    # 分成了组
    group_ids=['constant'],  # 【无】
    min_encoder_length=max_encoder_length,
    max_encoder_length=max_encoder_length,
    min_prediction_length=max_prediction_length,  # 【构建数据集时的最小预测长度，1】
    max_prediction_length=max_prediction_length,  # 【构建数据集时的最小预测长度，6】

    time_varying_known_reals=time_varying_known_reals,
    # 未来未知
    time_varying_unknown_reals=time_varying_unknown_reals,  # 【随时间变化且在未来未知的连续变量】
    #  # use softplus and normalize by group  【如何计算】   Softplus函数可以看作是ReLU函数的平滑。Softplus(x)=log(1+e^x)
    target_normalizer=TorchNormalizer(),
    # MultiNormalizer([TorchNormalizer(),TorchNormalizer()]),   # EncoderNormalizer(), TorchNormalizer()
    # add_relative_time_idx=True,
    # add_target_scales=True,
    add_encoder_length=False,
    allow_missing_timesteps=True
)



test_dataloader = testing.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

# ----------------------------------------- 创建基线模型 ----------------------------------------------
'''
# 通过简单地重复上次观察到的交易量来评估Baseline预测未来 6 个月的模型，为我们提供了一个我们想要超越的简单基准。
'''

import time, datetime
modelname = '{}_{}_{}_g{}_seed{}'.format(
    args.model_name, args.ablation_design, args.ablation, args.gate_init, args.seed
)
if args.module_ablation != 'none':
    modelname = '{}_{}_{}_seed{}'.format(
        args.model_name, args.ablation_design, args.module_ablation, args.seed
    )
save_dir_1 = 'results_{}/{}/{}/'.format(select_station, str(len(df)) + '_' + season_list[select_season - 1], str(args.inp_length)+'-'+str(args.outp_length)) + \
               modelname + time.strftime('_%Y-%m-%d %X', time.localtime()).replace(':', '-')
if not os.path.exists(save_dir_1):
    os.makedirs(save_dir_1)
ablation_flags = {
    'full': dict(use_variable_attention=True, use_feature_attention=True, use_temporal_attention=True),
    'no_variable': dict(use_variable_attention=False, use_feature_attention=True, use_temporal_attention=True),
    'no_feature': dict(use_variable_attention=True, use_feature_attention=False, use_temporal_attention=True),
    'no_temporal': dict(use_variable_attention=True, use_feature_attention=True, use_temporal_attention=False),
}

count=0
while count < args.num_runs:
    count+=1
    # --------------------------------------step 2: 网络构建--------------------------------------------------------
    # from pytorch_forecasting.metrics import MAE, RMSE, MultiHorizonMetric, MultiLoss, QuantileLoss
    from OurModels.TDFormer import TDFormer

    model_kwargs = dict(
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,  # 【除学习率之外最重要的参数】
        attention_head_size=args.attention_head_size,
        dropout=args.dropout,
        hidden_continuous_size=args.hidden_continuous_size,
        output_size=1,  # 【和分位数quantiles的数量对应】
        loss=CustomLoss(),  # 【quantiles=[0.1, 0.5, 0.9]】
        reduce_on_plateau_patience=args.lr_scheduler_patience,
        optimizer="Adam",
        residual_gated=args.ablation_design in ['residual_gated', 'residual_gated_v2'],
        gate_init=args.gate_init,
        variable_residual_v2=args.ablation_design == 'residual_gated_v2',
        variable_temperature=args.variable_temperature,
        variable_gate_init=args.variable_gate_init,
        **ablation_flags[args.ablation],
    )
    initialization_audit = {}
    if args.module_ablation != 'none':
        if args.ablation != 'full':
            raise ValueError('Do not combine legacy weighting ablation with whole-module ablation')
        from module_ablation_utils import build_matched_model
        model, initialization_audit = build_matched_model(training, model_kwargs, args.module_ablation)
    else:
        model = TDFormer.from_dataset(training, **model_kwargs)
    initial_state_hash = hashlib.sha256()
    for parameter_name, parameter_value in model.state_dict().items():
        initial_state_hash.update(parameter_name.encode('utf-8'))
        initial_state_hash.update(parameter_value.detach().cpu().contiguous().numpy().tobytes())
    initial_state_sha256 = initial_state_hash.hexdigest()
    # -----------------------------------------  保存文件名，设置特征数量【必须】 -----------------------------------
    after_name = 'in_l_{}_out_l_{}_Pcap'.format(max_encoder_length, max_prediction_length)
    import time, datetime

    save_dir = os.path.join(save_dir_1, modelname+str(count))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    curr_time = time.strftime('%Y-%m-%d %X', time.localtime()).replace(':', '-')

    log_path = os.path.join(save_dir,
                            'log_Feas_{}_{}_{}.txt'.format(len(training.reals), en_feats_num, after_name))
    log = open(log_path, 'w', encoding='UTF-8')
    # 保存训练结果
    study_model_path = os.path.join(save_dir, str(model).split('\n')[0][:-1])
    if not os.path.exists(study_model_path):
        os.makedirs(study_model_path)

    print_log(('{} train dataset:{},val:{},test:{}'.format(season_list[select_season-1],0.8*mask_len, 0.1*mask_len, 0.1*mask_len)),log)
    print_log(('Input Feature num:{} ,name:{}'.format(len(training.reals), training.reals)), log)

    print_log(('time_varying_unknown_reals num:{}，decoder num:{} ,name:{}'.format(len(training.time_varying_unknown_reals),
                                                                                  de_feats_num - len(
                                                                                      training.time_varying_unknown_reals),
                                                                                  training.time_varying_unknown_reals)),log)

    print_log('model:{}\n'.format(model),log)

    print_log((f"Number of parameters in network(参数数量（Params）: {model.size()/1e3:.1f}k"),log)

    print_log(('model model：\n{}'.format(model)),log)

    # for x, (y, weight) in iter(train_dataloader):
    #     print("['encoder_cat']:", x['encoder_cat'].shape)  # 350*24*15
    #     print("['encoder_cont']:", x['encoder_cont'].shape)  # 350*24*15
    #     en_feats_num = x['encoder_cat'].shape[-1] + x['encoder_cont'].shape[-1]
    #     print("['decoder_cont']:", x['decoder_cont'].shape)  # 350*6*15
    #     de_feats_num = x['decoder_cat'].shape[-1] + x['decoder_cont'].shape[-1]
    #     print('y.shape:', y.shape)  # 350*6
    #
    #     input = x
    #     out = model(x)
    #     # 计算模型的可训练参数
    #     print_log('Parameters:{} k'.format(model.size() / 1000),log)
    #
    #     loss=MAE()(out[0].squeeze(), y)
    #     print(loss)
    #     loss.backward()
    #     # 基于torchprofile 计算FLOPs
    #     from torchprofile import profile_macs
    #
    #     macs = profile_macs(model, input)
    #     print_log(('FLOPs:{} k'.format(str(macs / (1000 ** 2))[:6])),log)
    #     break



    # --------------------------------------step 3: 训练--------------------------------------------------------
    # 提前停止的回call函数
    # Keep early-stopping behavior aligned with the baseline training script.
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=1e-4,
        patience=20,
        verbose=False,
        mode='min',
    )
    # 学习率日志
    lr_logger = LearningRateMonitor()  # log the learning rate
    # 记录结果到张量板
    logger = TensorBoardLogger(study_model_path)  # logging results to a tensorboard


    # 训练器设置
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        gpus=1,
        gradient_clip_val=args.gradient_clip_val,
        limit_train_batches=30,  # coment in for training, running valiation every 30 batches
        # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
        callbacks=[early_stop_callback],
        # logger=logger,
        logger=False,
        default_root_dir=study_model_path,
    )
    #'''
    # 训练网络        ckpt_path: Path/URL of the checkpoint from which training is resumed.
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    # *** 加载最优模型 ***
    best_model_path = trainer.checkpoint_callback.best_model_path
    print('best_model_path:',best_model_path)

    # 加载最佳模型参数
    best_model = TDFormer.load_from_checkpoint(best_model_path)
    # 最优前推
    actuals = torch.cat([y for x, (y, weight) in iter(test_dataloader)])              # torch.Size([984, 1])
    predictions = best_model.predict(test_dataloader)                                 # torch.Size([984, 1])
    # ----------------------------  统计常用指标 ------------------

    # 数据
    y_true=actuals.cpu().detach().numpy().reshape(-1)   # actuals.max().item()
    y_pred=predictions.cpu().detach().numpy().reshape(-1)  # actuals.max().item()

    mask = y_true > 0
    print('y_true:{},y_pred:{},np.sum(mask):{}'.format(y_true.shape, y_pred.shape, np.sum(mask)))
    # 仅统计真实功率大于0的位置，保持原有MAE和RMSE口径。
    MAE = np.mean(np.abs(y_true[mask] - y_pred[mask]))
    RMSE = np.sqrt(np.mean(np.square(y_true[mask] - y_pred[mask])))
    #  打印实验结果
    print_log('MAE:{:.2f}%, RMSE:{:.2f}%\n'.format(MAE * 100, RMSE * 100), log)

    # 保存模型
    saved_checkpoint_path = log_path[:-4]+'_MAE_'+str(MAE*100)[:5]+'_RMSE_'+str(RMSE*100)[:5]+'.ckpt'
    saved_prediction_path = log_path[:-4] + '_MAE_' + str(MAE * 100)[:5] + '_RMSE_' + str(RMSE * 100)[:5] + '.npy'
    trainer.save_checkpoint(saved_checkpoint_path)
    np.save(saved_prediction_path, y_pred)
    if args.metrics_json:
        metrics_path = os.path.abspath(args.metrics_json)
        metrics_parent = os.path.dirname(metrics_path)
        if metrics_parent:
            os.makedirs(metrics_parent, exist_ok=True)
        metrics = {
            'status': 'complete',
            'variant': args.module_ablation if args.module_ablation != 'none' else args.ablation,
            'module_ablation': args.module_ablation,
            'initialization_audit': initialization_audit,
            'data_path': os.path.abspath(args.df),
            'model_hyperparameters': {
                'learning_rate': args.learning_rate, 'hidden_size': args.hidden_size,
                'attention_head_size': args.attention_head_size,
                'dropout': args.dropout, 'hidden_continuous_size': args.hidden_continuous_size,
                'gradient_clip_val': args.gradient_clip_val, 'limit_train_batches': 30,
                'optimizer': 'Adam', 'reduce_on_plateau_patience': args.lr_scheduler_patience,
            },
            'seed': args.seed,
            'input_length': args.inp_length,
            'prediction_length': args.outp_length,
            'train_window_count': len(training),
            'validation_window_count': len(validing),
            'test_window_count': len(testing),
            'evaluation_positive_count': int(np.sum(mask)),
            'early_stopping_config': {
                'monitor': 'val_loss', 'min_delta': 1e-4, 'patience': 20, 'mode': 'min',
            },
            'max_epochs': args.max_epochs,
            'batch_size': args.batch_size,
            'gpu': args.gpu,
            'dataset_profile': args.dataset_profile,
            'station': args.select_station,
            'target_capacity': args.target_capacity,
            'year_start': args.year_start,
            'year_end': args.year_end,
            'ablation_design': args.ablation_design,
            'gate_init': args.gate_init,
            'variable_temperature': args.variable_temperature,
            'variable_gate_init': args.variable_gate_init,
            'initial_state_sha256': initial_state_sha256,
            'parameter_count': sum(parameter.numel() for parameter in model.parameters()),
            'validation_loss_at_stop': float(trainer.callback_metrics['val_loss'].detach().cpu()),
            'checkpoint_policy': 'original default ModelCheckpoint without monitor; last saved epoch',
            'runtime_seconds': time.time() - run_started_at,
            'gpu_peak_allocated_mib': torch.cuda.max_memory_allocated() / (1024 ** 2),
            'gpu_peak_reserved_mib': torch.cuda.max_memory_reserved() / (1024 ** 2),
            'epochs_completed': int(trainer.current_epoch),
            'early_stopped': bool(early_stop_callback.stopped_epoch > 0),
            'early_stopping_stopped_epoch': int(early_stop_callback.stopped_epoch),
            'early_stopping_best_score': (
                float(early_stop_callback.best_score.detach().cpu())
                if early_stop_callback.best_score is not None
                else None
            ),
            'best_checkpoint_score': (
                float(trainer.checkpoint_callback.best_model_score.detach().cpu())
                if trainer.checkpoint_callback.best_model_score is not None
                else None
            ),
            'best_model_path': os.path.abspath(best_model_path),
            'saved_checkpoint_path': os.path.abspath(saved_checkpoint_path),
            'saved_prediction_path': os.path.abspath(saved_prediction_path),
            'metrics': {
                'mae': float(MAE),
                'rmse': float(RMSE),
            },
        }
        temporary_metrics_path = metrics_path + '.tmp'
        with open(temporary_metrics_path, 'w', encoding='utf-8') as metrics_file:
            json.dump(metrics, metrics_file, ensure_ascii=False, indent=2)
        os.replace(temporary_metrics_path, metrics_path)
    if not args.run_hpo:
        print_log('Optuna search skipped (use --run_hpo to enable it).', log)
        log.close()
        continue
    # --------------------------------------- 超参数调优 ---------------------------------
    '''
    使用 [optuna]( https://optuna.org/ ) 进行的超参数调整直接构建到 pytorch-forecasting 中。
    例如，我们可以使用该 optimize_hyperparameters()函数来优化 TFT 的超参数。
    '''
    # '''
    import pickle
    # from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
    from OurModels.optimize import optimize_hyperparameters

    #  创建一个study，进行超参数优化； 优化方式：
    study = optimize_hyperparameters(
        TDFormer,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        model_path=study_model_path,  # 设置保存模型的文件夹
        n_trials=20,  # n_trials (int, optional): Number of hyperparameter trials to run. Defaults to 100.
        max_epochs=100,  # max_epochs (int, optional): Maximum number of epochs to run training. Defaults to 20.
        gradient_clip_val_range=(0.1, 0.5),
        hidden_size_range=(50, 200),
        hidden_continuous_size_range=(50, 200),
        attention_head_size_range=(1, 4),
        learning_rate_range=(0.001, 0.2),
        dropout_range=(0.01, 0.2),
        trainer_kwargs=dict(limit_train_batches=64),
        reduce_on_plateau_patience=4,
        use_learning_rate_finder=False, )  # use Optuna to find ideal learning rate or use in-built learning rate finder

    # 保存训练结果，ckpt格式，用pickle保存
    with open(study_model_path + ".pkl", "wb") as fout:
        pickle.dump(study, fout)

    # show best hyperparameters
    print('study.best_trial.params:\n', study.best_trial.params)

    # ----------------------------------------------- 评估性能 ----------------------------------------------------------
    '''
    # PyTorch Lightning 自动检查点训练，因此，我们可以轻松检索最佳模型并加载它。
    '''
    # load the best model according to the validation loss
    # (given that we use early stopping, this is not necessarily the last epoch)
    # 获取最佳模型路径  'lightning_logs\\default\\version_12\\checkpoints\\epoch=9-step=299.ckpt'

    with open(study_model_path + ".pkl", 'rb') as f:
        study = pickle.load(f)
    print('study:', study)
    # 打印最佳模型
    print('best_trial:{}\n\n'.format(study.best_trial))
    # 打印最佳模型的关键参数
    print('study.best_trial  number:{} value:{} params:{}:\n'.format(study.best_trial.number, study.best_trial.value,
                                                                     study.best_trial.params))
    # 打印所有尝试的结果
    print('study.trials:{}\n'.format(study.trials))

    #  *** ---------------------------打印最佳模型，并写入log *** ------------------------------------

    # 打印最佳模型
    print_log(('best_trial:{}\n\n'.format(study.best_trial)), log)
    # 打印最佳模型的关键参数
    print_log(
        ('study.best_trial  number:{} value:{} params:{}:\n'.format(study.best_trial.number, study.best_trial.value,
                                                                    study.best_trial.params)), log)
    # 打印所有尝试的结果
    print_log(('study.trials:{}\n'.format(study.trials)), log)

    # ---- 获取study的最优模型保存结果---
    # 获取study最优模型的文件夹
    study_best_model_dir = os.path.join(study_model_path, 'trial_{}'.format(study.best_trial.number))
    # 获取study最优模型的路径
    study_best_model_path = os.path.join(study_best_model_dir, os.listdir(study_best_model_dir)[0])
    # 加载study最优模型
    # best_model = FFTFT.load_from_checkpoint(study_best_model_path)
    # 加载最佳模型参数
    best_study_model = TDFormer.load_from_checkpoint(study_best_model_path)
    # print('best_tft:\n',best_tft)

    actuals = torch.cat([y for x, (y, weight) in iter(test_dataloader)])  # 350*6
    predictions = best_study_model.predict(test_dataloader)
    print_log(('best_tft MAE: {}'.format((actuals - predictions).abs().mean())), log)

    # ----------------------------  统计常用指标 ------------------


    # 数据
    y_true = actuals.cpu().detach().numpy().reshape(-1)     # actuals.max().item()
    y_pred = predictions.cpu().detach().numpy().reshape(-1)  # actuals.max().item()

    mask = y_true > 0
    print_log(('y_true:{},y_pred:{},np.sum(mask):{}'.format(y_true.shape, y_pred.shape, np.sum(mask))),log)
    # 仅统计真实功率大于0的位置，保持原有MAE和RMSE口径。
    MAE = np.mean(np.abs(y_true[mask] - y_pred[mask]))
    RMSE = np.sqrt(np.mean(np.square(y_true[mask] - y_pred[mask])))
    #  打印实验结果
    print_log('MAE:{:.2f}%, RMSE:{:.2f}%\n'.format(MAE * 100, RMSE * 100), log)
    # 保存模型
    f = open(study_model_path + '.txt', 'w', encoding='UTF-8')
    #  打印实验结果
    print_log('MAE:{:.2f}%, RMSE:{:.2f}%\n'.format(MAE * 100, RMSE * 100), f)
    print_log(('study_best_model_path:{}'.format(study_best_model_path)), f)

    copyfile(study_best_model_path,
             log_path[:-4] + '_Best_Study_MAE_' + str(MAE * 100)[:5] + '_RMSE_' + str(RMSE * 100)[:5] + '.ckpt')
    np.save(log_path[:-4] + '_Best_Study_MAE_' + str(MAE * 100)[:5] + '_RMSE_' + str(RMSE * 100)[:5] + '.npy', y_pred)
    # 使用 shutil.rmtree() 删除非空文件夹
    import shutil

    try:
        shutil.rmtree(study_model_path)
        print('Folder {} and all its contents have been removed.'.format(study_model_path))
    except FileNotFoundError:
        print('The folder {} does not exist.'.format(study_model_path))
    except PermissionError:
        print('You do not have permission to delete the folder {}.'.format(study_model_path))
    except Exception as e:
        print('An error occurred: {}'.format(e))

    # '''
