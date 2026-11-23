from typing import Dict, List, Tuple, Union
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Metric as LightningMetric
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE, MultiHorizonMetric, MultiLoss, QuantileLoss
from pytorch_forecasting.models.base_model import BaseModelWithCovariates
from pytorch_forecasting.models.nn import LSTM, MultiEmbedding
from pytorch_forecasting.utils import autocorrelation, create_mask, detach, integer_histogram, padded_stack, to_list
import math
from typing import Dict, List, Tuple, Union
# from OurModels.Attention import FeatureSE, FeatureAttentionSE, FeatureAttentionSE4D
from pytorch_forecasting.utils import autocorrelation, create_mask, detach, integer_histogram, padded_stack, to_list
from einops import rearrange, repeat, einsum



class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output


class TimeDistributedInterpolation(nn.Module):
    def __init__(self, output_size: int, batch_first: bool = False, trainable: bool = False):
        super().__init__()
        self.output_size = output_size
        self.batch_first = batch_first
        self.trainable = trainable
        if self.trainable:
            self.mask = nn.Parameter(torch.zeros(self.output_size, dtype=torch.float32))
            self.gate = nn.Sigmoid()

    def interpolate(self, x):
        upsampled = F.interpolate(x.unsqueeze(1), self.output_size, mode="linear", align_corners=True).squeeze(1)
        if self.trainable:
            upsampled = upsampled * self.gate(self.mask.unsqueeze(0)) * 2.0
        return upsampled

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.interpolate(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.interpolate(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit"""

    def __init__(self, input_size: int, hidden_size: int = None, dropout: float = None):
        super().__init__()

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = dropout
        self.hidden_size = hidden_size or input_size
        self.fc = nn.Linear(input_size, self.hidden_size * 2)

        self.init_weights()

    def init_weights(self):
        for n, p in self.named_parameters():
            if "bias" in n:
                torch.nn.init.zeros_(p)
            elif "fc" in n:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc(x)
        x = F.glu(x, dim=-1)
        return x


class ResampleNorm(nn.Module):
    def __init__(self, input_size: int, output_size: int = None, trainable_add: bool = True):
        super().__init__()

        self.input_size = input_size
        self.trainable_add = trainable_add
        self.output_size = output_size or input_size

        if self.input_size != self.output_size:
            self.resample = TimeDistributedInterpolation(self.output_size, batch_first=True, trainable=False)

        if self.trainable_add:
            self.mask = nn.Parameter(torch.zeros(self.output_size, dtype=torch.float))
            self.gate = nn.Sigmoid()
        self.norm = nn.LayerNorm(self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_size != self.output_size:
            x = self.resample(x)

        if self.trainable_add:
            x = x * self.gate(self.mask) * 2.0

        output = self.norm(x)
        return output


class AddNorm(nn.Module):
    def __init__(self, input_size: int, skip_size: int = None, trainable_add: bool = True):
        super().__init__()

        self.input_size = input_size
        self.trainable_add = trainable_add
        self.skip_size = skip_size or input_size

        if self.input_size != self.skip_size:
            self.resample = TimeDistributedInterpolation(self.input_size, batch_first=True, trainable=False)

        if self.trainable_add:
            self.mask = nn.Parameter(torch.zeros(self.input_size, dtype=torch.float))
            self.gate = nn.Sigmoid()
        self.norm = nn.LayerNorm(self.input_size)

    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        if self.input_size != self.skip_size:
            skip = self.resample(skip)

        if self.trainable_add:
            # 学习一个参数来激活固定位置的原值状态。激活状态为[0,1]*2
            skip = skip * self.gate(self.mask) * 2.0

        output = self.norm(x + skip)
        return output


class GateAddNorm(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = None,
        skip_size: int = None,
        trainable_add: bool = False,
        dropout: float = None,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size or input_size
        self.skip_size = skip_size or self.hidden_size
        self.dropout = dropout

        self.glu = GatedLinearUnit(self.input_size, hidden_size=self.hidden_size, dropout=self.dropout)
        self.add_norm = AddNorm(self.hidden_size, skip_size=self.skip_size, trainable_add=trainable_add)

    def forward(self, x, skip):
        output = self.glu(x)
        output = self.add_norm(output, skip)
        return output


class GatedResidualNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.1,
        context_size: int = None,
        residual: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.residual = residual

        if self.input_size != self.output_size and not self.residual:
            residual_size = self.input_size
        else:
            residual_size = self.output_size

        if self.output_size != residual_size:
            self.resample_norm = ResampleNorm(residual_size, self.output_size)

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.elu = nn.ELU()

        if self.context_size is not None:
            self.context = nn.Linear(self.context_size, self.hidden_size, bias=False)

        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.init_weights()

        self.gate_norm = GateAddNorm(
            input_size=self.hidden_size,
            skip_size=self.output_size,
            hidden_size=self.output_size,
            dropout=self.dropout,
            trainable_add=False,
        )

    def init_weights(self):
        for name, p in self.named_parameters():
            if "bias" in name:
                torch.nn.init.zeros_(p)
            elif "fc1" in name or "fc2" in name:
                torch.nn.init.kaiming_normal_(p, a=0, mode="fan_in", nonlinearity="leaky_relu")
            elif "context" in name:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x, context=None, residual=None):
        if residual is None:
            residual = x

        if self.input_size != self.output_size and not self.residual:
            residual = self.resample_norm(residual)

        x = self.fc1(x)
        if context is not None:
            context = self.context(context)
            x = x + context
        x = self.elu(x)
        x = self.fc2(x)
        x = self.gate_norm(x, residual)
        return x


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''
    def __init__(self, d_in, d_hid, d_out, dropout, residual=True):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_out)  # position-wise
        self.layer_norm = nn.LayerNorm(d_out, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.residual=residual

    def forward(self, x):
        res = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        if self.residual:
            x += res
        x = self.layer_norm(x)
        return x


class VariableSelectionNetwork(nn.Module):
    def __init__(
        self,
        input_sizes: Dict[str, int],
        hidden_size: int,
        input_embedding_flags: Dict[str, bool] = {},
        dropout: float = 0.1,
        context_size: int = None,
        single_variable_grns: Dict[str, GatedResidualNetwork] = {},
        prescalers: Dict[str, nn.Linear] = {},
        use_variable_attention: bool = True,
        residual_gated: bool = False,
        gate_init: float = 0.05,
        variable_residual_v2: bool = False,
        variable_temperature: float = 0.7,
        variable_gate_init: float = 0.1,
    ):
        """
        Calcualte weights for ``num_inputs`` variables  which are each of size ``input_size``
        """
        super().__init__()

        self.hidden_size = hidden_size                              # 160
        self.input_sizes = input_sizes                              # {'categorical_id': 44, 'encoder_length': 160}
        self.input_embedding_flags = input_embedding_flags          # {'categorical_id': True}
        self.dropout = dropout                                      # 0.1
        self.context_size = context_size                            # None
        self.use_variable_attention = use_variable_attention
        self.residual_gated = residual_gated
        self.variable_residual_v2 = variable_residual_v2
        self.variable_temperature = variable_temperature
        '''
        8个特征
       ModuleDict(
                  (Wind direction | (deg)): Linear(in_features=1, out_features=100, bias=True)
                  (Pressure | (atm)): Linear(in_features=1, out_features=100, bias=True)
                  (Air temperature | ('C)): Linear(in_features=1, out_features=100, bias=True)
                  (month): Linear(in_features=1, out_features=100, bias=True)
                  (day_of_week): Linear(in_features=1, out_features=100, bias=True)
                  (hour): Linear(in_features=1, out_features=100, bias=True)
                  (System power generated | (kW)): Linear(in_features=1, out_features=100, bias=True)
                  (Wind speed | (m/s)): Linear(in_features=1, out_features=100, bias=True)
                )
        '''
        if self.num_inputs > 1 and (self.use_variable_attention or self.residual_gated):
            if self.context_size is not None:
                self.flattened_grn = GatedResidualNetwork(
                    self.input_size_total,
                    min(self.hidden_size, self.num_inputs),
                    self.num_inputs,    # 输出为2
                    self.dropout,
                    self.context_size,      # 上下信息 c
                    residual=False,
                )
            else:
                self.flattened_grn = GatedResidualNetwork(
                    self.input_size_total,
                    min(self.hidden_size, self.num_inputs),
                    self.num_inputs,    # 输出为2
                    self.dropout,
                    residual=False,
                )
        '''
        GatedResidualNetwork(
                              (resample_norm): ResampleNorm(
                                (resample): TimeDistributedInterpolation()
                                (gate): Sigmoid()
                                (norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)
                              )
                              (fc1): Linear(in_features=800, out_features=8, bias=True)     【800->8】
                              (elu): ELU(alpha=1.0)
                              (context): Linear(in_features=100, out_features=8, bias=False)
                              (fc2): Linear(in_features=8, out_features=8, bias=True)
                              (gate_norm): GateAddNorm(
                                (glu): GatedLinearUnit(
                                  (dropout): Dropout(p=0.2, inplace=False)
                                  (fc): Linear(in_features=8, out_features=16, bias=True)
                                )
                                (add_norm): AddNorm(
                                  (norm): LayerNorm((8,), eps=1e-05, elementwise_affine=True)【8->8】
                                )
                              )
                            )
        '''
        self.single_variable_grns = nn.ModuleDict()
        self.prescalers = nn.ModuleDict()
        for name, input_size in self.input_sizes.items():
            if name in single_variable_grns:
                self.single_variable_grns[name] = single_variable_grns[name]
            elif self.input_embedding_flags.get(name, False):
                self.single_variable_grns[name] = ResampleNorm(input_size, self.hidden_size)
            else:
                #  只进行特征选择
                self.single_variable_grns[name] = GatedResidualNetwork(
                    input_size,
                    min(input_size, self.hidden_size),
                    output_size=self.hidden_size,       # 输出仍然是特征层数量
                    dropout=self.dropout,
                )
            '''
            GatedResidualNetwork(
                                  (fc1): Linear(in_features=100, out_features=100, bias=True)
                                  (elu): ELU(alpha=1.0)
                                  (fc2): Linear(in_features=100, out_features=100, bias=True)
                                  (gate_norm): GateAddNorm(
                                    (glu): GatedLinearUnit(
                                      (dropout): Dropout(p=0.2, inplace=False)
                                      (fc): Linear(in_features=100, out_features=200, bias=True)
                                    )
                                    (add_norm): AddNorm(
                                      (norm): LayerNorm((100,), eps=1e-05, elementwise_affine=True)
                                    )
                                  )
                                )
            '''
            # reals need to be first scaled up
            if name in prescalers:
                self.prescalers[name] = prescalers[name]
            # 如果没有，手动创建
            elif not self.input_embedding_flags.get(name, False):
                self.prescalers[name] = nn.Linear(1, input_size)
                #self.prescalers[name] = None
        # 计算attention
        self.softmax = nn.Softmax(dim=-1)
        if self.variable_residual_v2:
            self.variable_delta_projection = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            nn.init.eye_(self.variable_delta_projection.weight)
            self.variable_output_norm = nn.LayerNorm(self.hidden_size)
            gate_probability = min(max(variable_gate_init, 1e-6), 1.0 - 1e-6)
            gate_logit = math.log(gate_probability / (1.0 - gate_probability))
            self.variable_gate_logit = nn.Parameter(torch.full((self.hidden_size,), gate_logit))
        elif self.residual_gated:
            self.variable_gate = nn.Parameter(torch.full((self.hidden_size,), gate_init))

    @property
    def input_size_total(self):
        return sum(size if name in self.input_embedding_flags else size for name, size in self.input_sizes.items())

    @property
    def num_inputs(self):
        return len(self.input_sizes)

    def forward(self, x: Dict[str, torch.Tensor], context: torch.Tensor = None):
        if self.num_inputs > 1:
            # transform single variables
            var_outputs = []
            weight_inputs = []
            '''
            dict_keys(['Wind direction | (deg)', 'Pressure | (atm)', "Air temperature | ('C)", 'month', 'day_of_week', 'hour', 'System power generated | (kW)', 'Wind speed | (m/s)'])
            '''
            for name in self.input_sizes.keys():
                # select embedding belonging to a single input
                variable_embedding = x[name]
                if name in self.prescalers:
                    # 1->100 升高维度（scaled up）
                    variable_embedding = self.prescalers[name](variable_embedding)
                # else:
                #     # 如果没有self.prescalers
                #     variable_embedding = variable_embedding
                weight_inputs.append(variable_embedding)
                # 这些GR
                # variable_embedding: [100, 48, 100]  100->100
                var_outputs.append(self.single_variable_grns[name](variable_embedding))
            # 8个 [100, 48, 100] -> [100, 48, 100, 8]
            var_outputs = torch.stack(var_outputs, dim=-1)

            if self.variable_residual_v2:
                uniform_weights = torch.full(
                    (*var_outputs.shape[:2], 1, self.num_inputs),
                    1.0 / self.num_inputs,
                    dtype=var_outputs.dtype,
                    device=var_outputs.device,
                )
                uniform_outputs = var_outputs.mean(dim=-1)
                if self.use_variable_attention:
                    flat_embedding = torch.cat(weight_inputs, dim=-1)
                    variable_logits = self.flattened_grn(flat_embedding, context)
                    sparse_weights = self.softmax(
                        variable_logits / self.variable_temperature
                    ).unsqueeze(-2)
                    weighted_outputs = (var_outputs * sparse_weights).sum(dim=-1)
                    delta = self.variable_delta_projection(weighted_outputs - uniform_outputs)
                    gate = torch.sigmoid(self.variable_gate_logit).view(1, 1, -1)
                    outputs = self.variable_output_norm(uniform_outputs + gate * delta)
                else:
                    sparse_weights = uniform_weights
                    outputs = self.variable_output_norm(uniform_outputs)
            elif self.residual_gated:
                uniform_weights = torch.full(
                    (*var_outputs.shape[:2], 1, self.num_inputs),
                    1.0 / self.num_inputs,
                    dtype=var_outputs.dtype,
                    device=var_outputs.device,
                )
                uniform_outputs = var_outputs.mean(dim=-1)
                if self.use_variable_attention:
                    flat_embedding = torch.cat(weight_inputs, dim=-1)
                    sparse_weights = self.softmax(self.flattened_grn(flat_embedding, context)).unsqueeze(-2)
                    weighted_outputs = (var_outputs * sparse_weights).sum(dim=-1)
                    gate = self.variable_gate.view(1, 1, -1)
                    outputs = uniform_outputs + gate * (weighted_outputs - uniform_outputs)
                else:
                    sparse_weights = uniform_weights
                    outputs = uniform_outputs
            elif self.use_variable_attention:
                # 计算变量权重 calculate variable weights
                # merge -> [100, 48, 800]
                flat_embedding = torch.cat(weight_inputs, dim=-1)
                # [100, 48, 800] -> [100, 48, 8]        [100, len, num_feats]
                sparse_weights = self.flattened_grn(flat_embedding, context) # context=None
                # [100, len, num_feats] -> [100, len, 1，num_feats]  【归一化】
                sparse_weights = self.softmax(sparse_weights).unsqueeze(-2)  # [100, 48, 1, 8]
            else:
                # Variable-dimension ablation: keep the same per-variable
                # projection, but replace learned selection with equal weights.
                sparse_weights = torch.full(
                    (*var_outputs.shape[:2], 1, self.num_inputs),
                    1.0 / self.num_inputs,
                    dtype=var_outputs.dtype,
                    device=var_outputs.device,
                )
            if not self.residual_gated:
                #  [100,48,100,8] * [100, 48, 1, 8]  -> [100,48,100,8]  特征重新加权求和，且softmax保证了权重和为1。
                outputs = var_outputs * sparse_weights
                # 实现特征合并。合并后的特征由占比权重高的体现。
                outputs = outputs.sum(dim=-1)

            # *** 使用attention替换 ***
            '''
                model = scSE(fea_dim)
                data = torch.ones((100,20,4))
                output = model(data)
            '''
        else:  # for one input, do not perform variable selection but just encoding
            name = next(iter(self.single_variable_grns.keys()))
            variable_embedding = x[name]
            if name in self.prescalers:
                variable_embedding = self.prescalers[name](variable_embedding)
            outputs = self.single_variable_grns[name](variable_embedding)  # fast forward if only one variable
            if outputs.ndim == 3:  # -> batch size, time, hidden size, n_variables
                sparse_weights = torch.ones(outputs.size(0), outputs.size(1), 1, 1, device=outputs.device)  #
            else:  # ndim == 2 -> batch size, hidden size, n_variables
                sparse_weights = torch.ones(outputs.size(0), 1, 1, device=outputs.device)
        # [100,48,100]   [100,48,1,8]
        return outputs, sparse_weights




class FlatVariableProjection(nn.Module):
    """Dimension adapter for w/o VAN: concatenate raw variables and project once.

    This deliberately contains no per-variable prescalers/GRNs, selection network,
    softmax, or residual variable gate. Returned uniform weights are placeholders
    for the unchanged interpretation-output interface, not learned attention.
    """

    def __init__(self, variable_names, hidden_size):
        super().__init__()
        self.variable_names = tuple(variable_names)
        self.projection = nn.Linear(len(self.variable_names), hidden_size)

    def forward(self, inputs, context=None):
        raw_variables = torch.cat([inputs[name] for name in self.variable_names], dim=-1)
        output = self.projection(raw_variables)
        weights = output.new_full(
            (*output.shape[:-1], 1, len(self.variable_names)), 1.0 / len(self.variable_names)
        )
        return output, weights


class PositionalEncoder(torch.nn.Module):
    def __init__(self, d_model, max_seq_len=160):
        super().__init__()
        assert d_model % 2 == 0, "model dimension has to be multiple of 2 (encode sin(pos) and cos(pos))"
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        with torch.no_grad():
            x = x * math.sqrt(self.d_model)
            seq_len = x.size(0)
            pe = self.pe[:, :seq_len].view(seq_len, 1, self.d_model)
            x = x + pe
            return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = None, scale: bool = True):
        super(ScaledDotProductAttention, self).__init__()
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = dropout
        self.softmax = nn.Softmax(dim=2)
        self.scale = scale

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.permute(0, 2, 1))  # query-key overlap

        if self.scale:
            dimension = torch.sqrt(torch.tensor(k.shape[-1]).to(torch.float32))
            attn = attn / dimension

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)
        attn = self.softmax(attn)

        if self.dropout is not None:
            attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, d_model: int, dropout: float = 0.0):
        super(InterpretableMultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = self.d_q = self.d_v = d_model // n_head
        self.dropout = nn.Dropout(p=dropout)

        self.v_layer = nn.Linear(self.d_model, self.d_v)
        self.q_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_q) for _ in range(self.n_head)])
        self.k_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_k) for _ in range(self.n_head)])
        self.attention = ScaledDotProductAttention()
        self.w_h = nn.Linear(self.d_v, self.d_model, bias=False)

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if "bias" not in name:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.zeros_(p)

    def forward(self, q, k, v, mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        heads = []
        attns = []
        vs = self.v_layer(v)
        for i in range(self.n_head):
            qs = self.q_layers[i](q)
            ks = self.k_layers[i](k)
            head, attn = self.attention(qs, ks, vs, mask)
            head_dropout = self.dropout(head)
            heads.append(head_dropout)
            attns.append(attn)

        head = torch.stack(heads, dim=2) if self.n_head > 1 else heads[0]
        attn = torch.stack(attns, dim=2)

        outputs = torch.mean(head, dim=2) if self.n_head > 1 else head
        outputs = self.w_h(outputs)
        outputs = self.dropout(outputs)

        return outputs, attn

class FeatureAttentionSE(nn.Module):
    '''
    基于SE原理，首先将不关注的维度进行自适应pooling。
    '''
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.Conv_Squeeze = nn.Conv1d(in_channels, in_channels // 2, kernel_size=1, bias=False)
        self.Conv_Excitation = nn.Conv1d(in_channels // 2, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, U):
        # [n, c, len] -> [n, c, 1]
        z = self.avgpool(U)             # shape: [bs, c, h, w] to [bs, c, 1, 1]
        # [n, c, 1]-> [n, c/2, 1]
        z = self.Conv_Squeeze(z)        # shape: [bs, c/2, 1, 1]
        # [n, c/2, 1]->[n, c, 1]
        z = self.Conv_Excitation(z)     # shape: [bs, c, 1, 1]
        # [n, c, 1]
        z = self.sigmoid(z)
        # [n, c,len]*[n, c,len] ,其中：expand_as，将c复制到所有维度 [n, c,len]
        outputs=U * z.expand_as(U)
        return outputs


class ResidualFeatureAttentionSE(nn.Module):
    """Centered SE recalibration with an exact identity ablation path."""

    def __init__(self, in_channels: int, enabled: bool = True, gate_init: float = 0.05):
        super().__init__()
        self.enabled = enabled
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.Conv_Squeeze = nn.Conv1d(in_channels, in_channels // 2, kernel_size=1, bias=False)
        self.Conv_Excitation = nn.Conv1d(in_channels // 2, in_channels, kernel_size=1, bias=False)
        self.gate_scale = nn.Parameter(torch.full((1, in_channels, 1), gate_init))

    def forward(self, inputs):
        if not self.enabled:
            return inputs
        logits = self.Conv_Excitation(self.Conv_Squeeze(self.avgpool(inputs)))
        centered_gate = 2.0 * torch.sigmoid(logits) - 1.0
        return inputs + self.gate_scale * inputs * centered_gate

class TDFormer(BaseModelWithCovariates):
    def __init__(
        self,
        hidden_size: int = 16,
        lstm_layers: int = 1,
        dropout: float = 0.1,
        output_size: Union[int, List[int]] = 7,
        loss: MultiHorizonMetric = None,
        attention_head_size: int = 4,
        max_encoder_length: int = 10,
        static_categoricals: List[str] = [],
        static_reals: List[str] = [],
        time_varying_categoricals_encoder: List[str] = [],
        time_varying_categoricals_decoder: List[str] = [],
        categorical_groups: Dict[str, List[str]] = {},
        time_varying_reals_encoder: List[str] = [],
        time_varying_reals_decoder: List[str] = [],
        x_reals: List[str] = [],
        x_categoricals: List[str] = [],
        hidden_continuous_size: int = 8,
        hidden_continuous_sizes: Dict[str, int] = {},
        embedding_sizes: Dict[str, Tuple[int, int]] = {},
        embedding_paddings: List[str] = [],
        embedding_labels: Dict[str, np.ndarray] = {},
        learning_rate: float = 1e-3,
        log_interval: Union[int, float] = -1,
        log_val_interval: Union[int, float] = None,
        log_gradient_flow: bool = False,
        reduce_on_plateau_patience: int = 1000,
        monotone_constaints: Dict[str, int] = {},
        share_single_variable_networks: bool = False,
        use_variable_attention: bool = True,
        use_feature_attention: bool = True,
        use_temporal_attention: bool = True,
        residual_gated: bool = False,
        gate_init: float = 0.05,
        variable_residual_v2: bool = False,
        variable_temperature: float = 0.7,
        variable_gate_init: float = 0.1,
        module_ablation: str = 'none',
        **kwargs,
    ):

        # ---默认损失函数：分位数损失---
        if loss is None:
            loss = QuantileLoss()
        self.save_hyperparameters()
        super().__init__(loss=loss, **kwargs)
        if module_ablation not in ('none', 'full', 'wo_tan', 'wo_fan', 'wo_van'):
            raise ValueError('Unknown whole-module ablation: {}'.format(module_ablation))

        # 升维：将所有的特征，都映射到指定的维度。1-> hidden_continuous_size(160)
        self.prescalers = nn.ModuleDict(
            {
                name: nn.Linear(1, self.hparams.hidden_continuous_sizes.get(name, self.hparams.hidden_continuous_size))
                for name in self.reals
                if self.hparams.module_ablation != 'wo_van'
            }
        )
        # 编码器变量选择
        encoder_input_sizes = {
            name: self.hparams.embedding_sizes[name][1] for name in self.hparams.time_varying_categoricals_encoder
        }
        encoder_input_sizes.update(
            {
                name: self.hparams.hidden_continuous_sizes.get(name, self.hparams.hidden_continuous_size)
                for name in self.hparams.time_varying_reals_encoder
            }
        )
        # 解码器变量选择
        decoder_input_sizes = {
            name: self.hparams.embedding_sizes[name][1] for name in self.hparams.time_varying_categoricals_decoder
        }
        decoder_input_sizes.update(
            {
                name: self.hparams.hidden_continuous_sizes.get(name, self.hparams.hidden_continuous_size)
                for name in self.hparams.time_varying_reals_decoder
            }
        )
        # 与时间相关的信息创建单个变量的GRNs，在解码器和编码器之间共享
        if self.hparams.share_single_variable_networks:
            self.shared_single_variable_grns = nn.ModuleDict()
            for name, input_size in encoder_input_sizes.items():
                self.shared_single_variable_grns[name] = GatedResidualNetwork(
                    input_size,
                    min(input_size, self.hparams.hidden_size),
                    self.hparams.hidden_size,
                    self.hparams.dropout,
                )
            for name, input_size in decoder_input_sizes.items():
                if name not in self.shared_single_variable_grns:
                    self.shared_single_variable_grns[name] = GatedResidualNetwork(
                        input_size,
                        min(input_size, self.hparams.hidden_size),
                        self.hparams.hidden_size,
                        self.hparams.dropout,
                    )
        # --- 编码器的VSN网络 --- #
        self.encoder_variable_selection = VariableSelectionNetwork(
            input_sizes=encoder_input_sizes,
            hidden_size=self.hparams.hidden_size,
            input_embedding_flags={name: True for name in self.hparams.time_varying_categoricals_encoder},
            dropout=self.hparams.dropout,
            context_size=self.hparams.hidden_size,
            prescalers=self.prescalers,                                 # 特征升维
            single_variable_grns={}
            if not self.hparams.share_single_variable_networks
            else self.shared_single_variable_grns,
            use_variable_attention=self.hparams.use_variable_attention,
            residual_gated=self.hparams.residual_gated,
            gate_init=self.hparams.gate_init,
            variable_residual_v2=self.hparams.variable_residual_v2,
            variable_temperature=self.hparams.variable_temperature,
            variable_gate_init=self.hparams.variable_gate_init,
        )
  
        # --- 解码器的VSN网络 --- #
        self.decoder_variable_selection = VariableSelectionNetwork(
            input_sizes=decoder_input_sizes,
            hidden_size=self.hparams.hidden_size,
            input_embedding_flags={name: True for name in self.hparams.time_varying_categoricals_decoder},
            dropout=self.hparams.dropout,
            context_size=self.hparams.hidden_size,
            prescalers=self.prescalers,                                     # 特征升维
            single_variable_grns={}
            if not self.hparams.share_single_variable_networks
            else self.shared_single_variable_grns,
            use_variable_attention=self.hparams.use_variable_attention,
            residual_gated=self.hparams.residual_gated,
            gate_init=self.hparams.gate_init,
            variable_residual_v2=self.hparams.variable_residual_v2,
            variable_temperature=self.hparams.variable_temperature,
            variable_gate_init=self.hparams.variable_gate_init,
        )

        if self.hparams.module_ablation == 'wo_van':
            # Physically replace the complete VAN/VSN, rather than only its
            # learned variable weighting. Global prescalers are also absent.
            self.encoder_variable_selection = FlatVariableProjection(
                encoder_input_sizes, self.hparams.hidden_size
            )
            self.decoder_variable_selection = FlatVariableProjection(
                decoder_input_sizes, self.hparams.hidden_size
            )
            if hasattr(self, 'shared_single_variable_grns'):
                del self.shared_single_variable_grns

        if self.hparams.residual_gated:
            self.fusion_encoder = ResidualFeatureAttentionSE(
                hidden_size, enabled=self.hparams.use_feature_attention, gate_init=self.hparams.gate_init
            )
            self.fusion_decoder = ResidualFeatureAttentionSE(
                hidden_size, enabled=self.hparams.use_feature_attention, gate_init=self.hparams.gate_init
            )
            self.TAN_encoder = InterpretableMultiHeadAttention(d_model=self.hparams.hidden_size, n_head=self.hparams.attention_head_size, dropout=self.hparams.dropout )
            self.TAN_decoder = InterpretableMultiHeadAttention(d_model=self.hparams.hidden_size, n_head=self.hparams.attention_head_size, dropout=self.hparams.dropout )
            self.temporal_gate_encoder = nn.Parameter(
                torch.full((1, 1, self.hparams.hidden_size), self.hparams.gate_init)
            )
            self.temporal_gate_decoder = nn.Parameter(
                torch.full((1, 1, self.hparams.hidden_size), self.hparams.gate_init)
            )
        else:
            self.fusion_encoder = FeatureAttentionSE(hidden_size) if self.hparams.use_feature_attention else nn.Identity()
            self.fusion_decoder = FeatureAttentionSE(hidden_size) if self.hparams.use_feature_attention else nn.Identity()

            if self.hparams.use_temporal_attention:
                self.TAN_encoder = InterpretableMultiHeadAttention(d_model=self.hparams.hidden_size, n_head=self.hparams.attention_head_size, dropout=self.hparams.dropout )
                self.TAN_decoder = InterpretableMultiHeadAttention(d_model=self.hparams.hidden_size, n_head=self.hparams.attention_head_size, dropout=self.hparams.dropout )
            else:
                self.TAN_encoder = None
                self.TAN_decoder = None

        if self.hparams.module_ablation == 'wo_fan':
            self.fusion_encoder = nn.Identity()
            self.fusion_decoder = nn.Identity()
        if self.hparams.module_ablation == 'wo_tan':
            self.TAN_encoder = None
            self.TAN_decoder = None
            if hasattr(self, 'temporal_gate_encoder'):
                del self.temporal_gate_encoder
                del self.temporal_gate_decoder
        
        # lstm 编码器
        self.lstm_encoder = LSTM(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.lstm_layers,
            dropout=self.hparams.dropout if self.hparams.lstm_layers > 1 else 0,
            batch_first=True,
        )
        # lstm 解码器
        self.lstm_decoder = LSTM(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.lstm_layers,
            dropout=self.hparams.dropout if self.hparams.lstm_layers > 1 else 0,
            batch_first=True,
        )

        # LSTM后面接的门选择和残差学习           skip connection for lstm
        self.post_lstm_gate_encoder = GatedLinearUnit(self.hparams.hidden_size, dropout=self.hparams.dropout)
        self.post_lstm_gate_decoder = self.post_lstm_gate_encoder
        # self.post_lstm_gate_decoder = GatedLinearUnit(self.hparams.hidden_size, dropout=self.hparams.dropout)
        self.post_lstm_add_norm_encoder = AddNorm(self.hparams.hidden_size, trainable_add=False)
        # 这个地方要注意一下！
        self.post_lstm_add_norm_decoder = AddNorm(self.hparams.hidden_size, trainable_add=True)
        # self.post_lstm_add_norm_decoder = self.post_lstm_add_norm_encoder


        # 多头可解释的attention           attention for long-range processing
        self.multihead_attn = InterpretableMultiHeadAttention(d_model=self.hparams.hidden_size, n_head=self.hparams.attention_head_size, dropout=self.hparams.dropout )

        # GAN
        self.post_attn_gate_norm = GateAddNorm(self.hparams.hidden_size, dropout=self.hparams.dropout, trainable_add=False)
        # GRN再次特征选择
        self.pos_wise_ff = GatedResidualNetwork(self.hparams.hidden_size, self.hparams.hidden_size, self.hparams.hidden_size, dropout=self.hparams.dropout)

        # 再GAN进行      output processing -> no dropout at this late stage
        self.pre_output_gate_norm = GateAddNorm(self.hparams.hidden_size, dropout=None, trainable_add=False)
        # 线性回归层
        if self.n_targets > 1:  # if to run with multiple targets
            self.output_layer = nn.ModuleList(
                [nn.Linear(self.hparams.hidden_size, output_size) for output_size in self.hparams.output_size]
            )
        else:
            self.output_layer = nn.Linear(self.hparams.hidden_size, self.hparams.output_size)

    @classmethod
    def from_dataset(
        cls,
        dataset: TimeSeriesDataSet,
        allowed_encoder_known_variable_names: List[str] = None,
        **kwargs,
    ):
        """
        Create model from dataset.

        Args:
            dataset: timeseries dataset
            allowed_encoder_known_variable_names: List of known variables that are allowed in encoder, defaults to all
            **kwargs: additional arguments such as hyperparameters for model (see ``__init__()``)

        Returns:
            TemporalFusionTransformer
        """
        # add maximum encoder length
        new_kwargs = dict(max_encoder_length=dataset.max_encoder_length)
        new_kwargs.update(cls.deduce_default_output_parameters(dataset, kwargs, QuantileLoss()))

        # update defaults
        new_kwargs.update(kwargs)

        # create class and return
        return super().from_dataset(
            dataset, allowed_encoder_known_variable_names=allowed_encoder_known_variable_names, **new_kwargs
        )

    def expand_static_context(self, context, timesteps):
        """
        add time dimension to static context
        """
        return context[:, None].expand(-1, timesteps, -1)

    def get_attention_mask(self, encoder_lengths: torch.LongTensor, decoder_length: int):
        """
        Returns causal mask to apply for self-attention layer.

        Args:
            self_attn_inputs: Inputs to self attention layer to determine mask shape
        """
        # indices to which is attended
        attend_step = torch.arange(decoder_length, device=self.device)
        # indices for which is predicted
        predict_step = torch.arange(0, decoder_length, device=self.device)[:, None]
        # do not attend to steps to self or after prediction
        # todo: there is potential value in attending to future forecasts if they are made with knowledge currently
        #   available
        #   one possibility is here to use a second attention layer for future attention (assuming different effects
        #   matter in the future than the past)
        #   or alternatively using the same layer but allowing forward attention - i.e. only masking out non-available
        #   data and self
        decoder_mask = attend_step >= predict_step
        # do not attend to steps where data is padded
        encoder_mask = create_mask(encoder_lengths.max(), encoder_lengths)
        # combine masks along attended time - first encoder and then decoder
        mask = torch.cat(
            (
                encoder_mask.unsqueeze(1).expand(-1, decoder_length, -1),
                decoder_mask.unsqueeze(0).expand(encoder_lengths.size(0), -1, -1),
            ),
            dim=2,
        )
        return mask

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        input dimensions: n_samples x time x variables
        """
        encoder_lengths = x["encoder_lengths"]
        decoder_lengths = x["decoder_lengths"]
        x_cont = torch.cat([x["encoder_cont"], x["decoder_cont"]], dim=1)  # concatenate in time dimension

        timesteps = x_cont.size(1)  # encode + decode length

        max_encoder_length = int(encoder_lengths.max())

        input_vectors={}

        input_vectors.update(
            {
                name: x_cont[..., idx].unsqueeze(-1)
                for idx, name in enumerate(self.hparams.x_reals)
                if name in self.reals
            }
        )

        if len(self.static_variables) > 0:
            static_embedding = {name: input_vectors[name][:, 0] for name in self.static_variables}
            static_embedding, static_variable_selection = self.static_variable_selection(static_embedding)
        else:
            static_embedding = torch.zeros(
                (x_cont.size(0), self.hparams.hidden_size), dtype=self.dtype, device=self.device
            )
            static_variable_selection = torch.zeros((x_cont.size(0), 0), dtype=self.dtype, device=self.device)

        # 编码器的嵌入输出          embeddings_varying_encoder： torch.Size([64, 168, 160])
        embeddings_varying_encoder = {
            name: input_vectors[name][:, :max_encoder_length] for name in self.encoder_variables
        }
        # ------------------------------编码器的VSN,特征选择-----------------------------
        embeddings_varying_encoder, encoder_sparse_weights = self.encoder_variable_selection(embeddings_varying_encoder)
        # 嵌入后变量的解码器
        embeddings_varying_decoder = {
            name: input_vectors[name][:, max_encoder_length:] for name in self.decoder_variables  # select decoder
        }
        # --------------------------------解码器的VSN,特征选择---------------------------------
        embeddings_varying_decoder, decoder_sparse_weights = self.decoder_variable_selection(embeddings_varying_decoder)


        embeddings_varying_encoder_FAN = self.fusion_encoder(embeddings_varying_encoder.permute((0,2,1))).permute((0,2,1))
        embeddings_varying_decoder_FAN = self.fusion_decoder(embeddings_varying_decoder.permute((0,2,1))).permute((0,2,1))

        if self.hparams.module_ablation == 'wo_tan':
            encoder_attn_output = embeddings_varying_encoder_FAN
            decoder_attn_output = embeddings_varying_decoder_FAN
        elif self.hparams.residual_gated:
            if self.hparams.use_temporal_attention:
                encoder_attention_delta, _ = self.TAN_encoder(
                    q=embeddings_varying_encoder_FAN,
                    k=embeddings_varying_encoder_FAN,
                    v=embeddings_varying_encoder_FAN,
                )
                decoder_attention_delta, _ = self.TAN_decoder(
                    q=embeddings_varying_decoder_FAN,
                    k=embeddings_varying_decoder_FAN,
                    v=embeddings_varying_decoder_FAN,
                )
                encoder_attn_output = (
                    embeddings_varying_encoder_FAN
                    + self.temporal_gate_encoder * encoder_attention_delta
                )
                decoder_attn_output = (
                    embeddings_varying_decoder_FAN
                    + self.temporal_gate_decoder * decoder_attention_delta
                )
            else:
                encoder_attn_output = embeddings_varying_encoder_FAN
                decoder_attn_output = embeddings_varying_decoder_FAN
        elif self.hparams.use_temporal_attention:
            encoder_attn_output, _ = self.TAN_encoder(
                q=embeddings_varying_encoder_FAN,
                k=embeddings_varying_encoder_FAN,
                v=embeddings_varying_encoder_FAN,
            )

            decoder_attn_output, _ = self.TAN_decoder(
                q=embeddings_varying_decoder_FAN,
                k=embeddings_varying_decoder_FAN,
                v=embeddings_varying_decoder_FAN,
            )
        else:
            encoder_attn_output = embeddings_varying_encoder_FAN
            decoder_attn_output = embeddings_varying_decoder_FAN



        # 运行LSTM编码器         run local encoder
        encoder_output, (hidden, cell) = self.lstm_encoder(
            encoder_attn_output,
            lengths=encoder_lengths,
            enforce_sorted=False
        )
        # encoder_output =embeddings_varying_encoder_MAH
        # 运行LSTM解码器，注意：使用编码器的hidden和cell         run local decoder
        decoder_output, _ = self.lstm_decoder(
            decoder_attn_output,
            (hidden, cell),            # 编码器的hidden和cell作为解码器的输入   【隐状态起到了至关重要的作用！】
            lengths=decoder_lengths,
            enforce_sorted=False,
        )

        # 编码器后Gate+AddNorm，作用：残差skip
        lstm_output_encoder = self.post_lstm_gate_encoder(encoder_output)
        # add_norm
        lstm_output_encoder = self.post_lstm_add_norm_encoder(lstm_output_encoder, embeddings_varying_encoder)
        # 解码器后Gate+AddNorm，作用：残差skip
        lstm_output_decoder = self.post_lstm_gate_decoder(decoder_output)
        # add_norm
        lstm_output_decoder = self.post_lstm_add_norm_decoder(lstm_output_decoder, embeddings_varying_decoder)

        # --- 将编码器-解码器的LSTM结果concat在一起 --- #
        lstm_output = torch.cat([lstm_output_encoder, lstm_output_decoder], dim=1)

        # 核心：Multi-head Attention，注意：q只计算需要预测的。
        attn_output, attn_output_weights = self.multihead_attn(
            q=lstm_output[:, max_encoder_length:],  # query only for predictions
            k=lstm_output,
            v=lstm_output,
            mask=self.get_attention_mask(
                encoder_lengths=encoder_lengths, decoder_length=timesteps - max_encoder_length
            ),
        )

        # Attention后的第一个Gate_add_norm     自适应的抑制不相关或噪声特征的影响，同时放大关键特征的信号
        attn_output = self.post_attn_gate_norm(attn_output, lstm_output[:, max_encoder_length:])
        # GRN 再进行特征提取，此处没有特征融合。
        output = self.pos_wise_ff(attn_output)

        # 残差连接  skip connection over temporal fusion decoder (not LSTM decoder despite the LSTM output contains
        output = self.pre_output_gate_norm(output, lstm_output[:, max_encoder_length:])

        # output_layer 最后的线性回归层
        if self.n_targets > 1:  # if to use multi-target architecture
            output = [output_layer(output) for output_layer in self.output_layer]
        else:
            output = self.output_layer(output)

        return self.to_network_output(
            prediction=self.transform_output(output, target_scale=x["target_scale"]),
            attention=attn_output_weights,
            static_variables=static_variable_selection,
            encoder_variables=encoder_sparse_weights,
            decoder_variables=decoder_sparse_weights,
            decoder_lengths=decoder_lengths,
            encoder_lengths=encoder_lengths,
        )

    def create_log(self, x, y, out, batch_idx, **kwargs):
        log = super().create_log(x, y, out, batch_idx, **kwargs)
        if self.log_interval > 0:
            log["interpretation"] = self._log_interpretation(out)
        return log

    def _log_interpretation(self, out):
        # calculate interpretations etc for latter logging
        interpretation = self.interpret_output(
            detach(out),
            reduction="sum",
            attention_prediction_horizon=0,  # attention only for first prediction horizon
        )
        return interpretation

    def epoch_end(self, outputs):
        """
        run at epoch end for training or validation
        """
        if self.log_interval > 0:
            self.log_interpretation(outputs)

    def interpret_output(
        self,
        out: Dict[str, torch.Tensor],
        reduction: str = "none",
        attention_prediction_horizon: int = 0,
        attention_as_autocorrelation: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        interpret output of model

        Args:
            out: output as produced by ``forward()``
            reduction: "none" for no averaging over batches, "sum" for summing attentions, "mean" for
                normalizing by encode lengths
            attention_prediction_horizon: which prediction horizon to use for attention
            attention_as_autocorrelation: if to record attention as autocorrelation - this should be set to true in
                case of ``reduction != "none"`` and differing prediction times of the samples. Defaults to False

        Returns:
            interpretations that can be plotted with ``plot_interpretation()``
        """

        # histogram of decode and encode lengths
        encoder_length_histogram = integer_histogram(out["encoder_lengths"], min=0, max=self.hparams.max_encoder_length)
        decoder_length_histogram = integer_histogram(
            out["decoder_lengths"], min=1, max=out["decoder_variables"].size(1)
        )

        # mask where decoder and encoder where not applied when averaging variable selection weights
        encoder_variables = out["encoder_variables"].squeeze(-2)
        encode_mask = create_mask(encoder_variables.size(1), out["encoder_lengths"])
        encoder_variables = encoder_variables.masked_fill(encode_mask.unsqueeze(-1), 0.0).sum(dim=1)
        encoder_variables /= (
            out["encoder_lengths"]
            .where(out["encoder_lengths"] > 0, torch.ones_like(out["encoder_lengths"]))
            .unsqueeze(-1)
        )

        decoder_variables = out["decoder_variables"].squeeze(-2)
        decode_mask = create_mask(decoder_variables.size(1), out["decoder_lengths"])
        decoder_variables = decoder_variables.masked_fill(decode_mask.unsqueeze(-1), 0.0).sum(dim=1)
        decoder_variables /= out["decoder_lengths"].unsqueeze(-1)

        # static variables need no masking
        static_variables = out["static_variables"].squeeze(1)
        # attention is batch x time x heads x time_to_attend
        # average over heads + only keep prediction attention and attention on observed timesteps
        attention = out["attention"][
            :, attention_prediction_horizon, :, : out["encoder_lengths"].max() + attention_prediction_horizon
        ].mean(1)

        if reduction != "none":  # if to average over batches
            static_variables = static_variables.sum(dim=0)
            encoder_variables = encoder_variables.sum(dim=0)
            decoder_variables = decoder_variables.sum(dim=0)

            # reorder attention or averaging
            for i in range(len(attention)):  # very inefficient but does the trick
                if 0 < out["encoder_lengths"][i] < attention.size(1) - attention_prediction_horizon - 1:
                    relevant_attention = attention[
                        i, : out["encoder_lengths"][i] + attention_prediction_horizon
                    ].clone()
                    if attention_as_autocorrelation:
                        relevant_attention = autocorrelation(relevant_attention)
                    attention[i, -out["encoder_lengths"][i] - attention_prediction_horizon :] = relevant_attention
                    attention[i, : attention.size(1) - out["encoder_lengths"][i] - attention_prediction_horizon] = 0.0
                elif attention_as_autocorrelation:
                    attention[i] = autocorrelation(attention[i])

            attention = attention.sum(dim=0)
            if reduction == "mean":
                attention = attention / encoder_length_histogram[1:].flip(0).cumsum(0).clamp(1)
                attention = attention / attention.sum(-1).unsqueeze(-1)  # renormalize
            elif reduction == "sum":
                pass
            else:
                raise ValueError(f"Unknown reduction {reduction}")

            attention = torch.zeros(
                self.hparams.max_encoder_length + attention_prediction_horizon, device=self.device
            ).scatter(
                dim=0,
                index=torch.arange(
                    self.hparams.max_encoder_length + attention_prediction_horizon - attention.size(-1),
                    self.hparams.max_encoder_length + attention_prediction_horizon,
                    device=self.device,
                ),
                src=attention,
            )
        else:
            attention = attention / attention.sum(-1).unsqueeze(-1)  # renormalize

        interpretation = dict(
            attention=attention,
            static_variables=static_variables,
            encoder_variables=encoder_variables,
            decoder_variables=decoder_variables,
            encoder_length_histogram=encoder_length_histogram,
            decoder_length_histogram=decoder_length_histogram,
        )
        return interpretation



