import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence


def clones(module, n):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class AttLayer(nn.Module):

    def __init__(self, word_emb_dim, attention_hidden_dim):
        super().__init__()
        # build attention network
        self.attention = nn.Sequential(
            nn.Linear(word_emb_dim, attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(attention_hidden_dim, 1),
            nn.Flatten(),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # x (N,H,E)
        attention_weight = torch.unsqueeze(self.attention(x), 2) #(N,H,1)
        y = torch.sum(x * attention_weight, dim=1)#(N,E)
        return y, attention_weight


class MultiHeadedAttention(nn.Module):
    """
    MultiheadedAttention:

    http://nlp.seas.harvard.edu/2018/04/03/attention.html#attention
    """

    def __init__(self, h, d_k, word_dim, dropout=0):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        # We assume d_v always equals d_k
        self.d_k = d_k
        self.h = h
        d_model = h * d_k
        self.linears = clones(nn.Linear(word_dim, d_model), 3)
        self.final = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [liner(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for liner, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.final(x), self.attn


class AddNorm(nn.Module):
    """残差连接后进行层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.dropout1(Y) + self.dropout2(X)


# class GRU_Attention(nn.Module):
#     def __init__(self, hidden_size):
#         super(GRU_Attention, self).__init__()
#         self.hidden_size = hidden_size
#         self.attn = nn.Linear(2 * hidden_size, 1)
#         self.attlayer = AttLayer(self.hidden_size,64)
#
#     def forward(self, hidden, encoder_outputs):
#         # hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1) #(32,256,300)
#         # energy = F.relu(self.attn(encoder_outputs))# (32,256,1)
#         # attn_weights = F.softmax(energy, dim=1) #(32,256,1)
#         # weighted_encoding = torch.bmm(attn_weights.permute(0, 2, 1), encoder_outputs) #(32,1,256) (32,256,300) -> (32,1,300)
#         # return weighted_encoding.squeeze(1)  # (32,300)
#         weighted_encoding, weight = self.attlayer(encoder_outputs)
#         return weighted_encoding, weight
#
#
#
# class GRUWithAttention(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers=1):
#         super(GRUWithAttention, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#
#         self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
#         self.attention = GRU_Attention(hidden_size)
#
#     def forward(self, input, max_length):
#         output, hidden = self.gru(input)
#         output, _ = pad_packed_sequence(output, batch_first=True, total_length=max_length)
#         weighted_encoding, weight = self.attention(hidden[-1], output)
#         return output, hidden, weighted_encoding, weight


class RNNBase(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(RNNBase, self).__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        self.attn = nn.Linear(hidden_size, 1)
        self.params = None
        self.state = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_params(self, input_size, hidden_size):
        num_inputs = num_outputs = input_size

        def normal(shape):
            return torch.randn(size=shape, device=self.device) * 0.01

        def three():
            return (normal((num_inputs, hidden_size)),
                    normal((hidden_size, hidden_size)),
                    torch.zeros(hidden_size, device=self.device))

        W_xz, W_hz, b_z = three() # 更新门参数
        W_xr, W_hr, b_r = three() # 重置门参数
        W_xh, W_hh, b_h = three() # 候选隐状态参数
        # 输出层参数
        W_hq = normal((hidden_size, num_outputs))
        b_q = torch.zeros(num_outputs, device=self.device)

        # 附加梯度
        params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
        for param in params:
            param.requires_grad_(True)
        return params


    def init_state(self, batch_size, hidden_size):
        return (torch.zeros((batch_size, hidden_size), device=self.device),)


    def gru(self, inputs):
        W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = self.params
        H,  = self.state
        seq_len, batch_size = len(inputs), len(inputs[0])
        # outputs = []
        # hiddens = []
        # X 的形状：(批量大小，input_size)
        outputs = torch.empty(seq_len, batch_size, self.input_size, device=self.device)
        hiddens = torch.empty(seq_len, batch_size, self.hidden_size, device=self.device)
        for index, X in enumerate(inputs):
            Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
            R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
            H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
            H = Z * H + (1 - Z) * H_tilda
            Y = H @ W_hq + b_q
            # outputs.append(Y)
            # hiddens.append(H)
            outputs[index] = Y.detach()
            hiddens[index] = H.detach()
        # 其实拼接得到的 hinden 形状为：[seq_len, batch, hidden_size]
        # hinden = torch.cat(hiddens.unsqueeze(0), dim=0)
        # return torch.cat(outputs.unsqueeze(0), dim=0), (H,), hinden
        return outputs, (H,), hiddens


    def attention_layer(self, hidden):
        # encode_input:[seq_len, batch_size, input_size]
        # hidden:[seq_len, batch_size, hidden_size]
        hidden = hidden.transpose(0,1)
        energy = F.tanh(self.attn(hidden))# (32,256,1)
        attn_weights = F.softmax(energy, dim=1) #(32,256,1)
        context = torch.sum(hidden * attn_weights, dim=1) # (32,256,300) * (32,256,1) -> (N,E)
        return context, attn_weights

    def forward(self, input):
        # input:[batch_size, seq_len, input_size]
        batch_size = len(input) # 32
        if self.params is None:
            # 第一次迭代
            self.params = self.get_params(self.input_size, self.hidden_size)
        if self.state is None:
            # 第一次迭代
            self.state = self.init_state(batch_size, self.hidden_size)
        else:
            # 第二次以上迭代
            if not isinstance(self.state, tuple):
                self.state.detach_()
            else:
                for s in self.state:
                    s.detach_()
        input = input.transpose(0,1) # input:[seq_len, batch_size, input_size]
        outputs, self.state, hinden = self.gru(input) # self.state 是最后一个时间步的隐状态， hinden,每个时间步的隐状态
        context, attn_weights = self.attention_layer(hinden)
        return outputs.transpose(0,1), context,attn_weights


class GRUWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirection = False):
        super(GRUWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bidirectional=bidirection)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bidirection)

    def attention_layer(self, encoder_input, final_state):
        # encoder_input: [batch_size, seq_len, hidden_size * num_directions(=1)]
        batch_size = len(encoder_input) # 32
        # hidden : [batch_size, hidden_size * num_directions(=1), n_layer(=1)]
        # hidden = torch.cat((final_state[0], final_state[1]),dim=1).unsqueeze(2)
        hidden = final_state[0].unsqueeze(2)
        # attention_weights: [batch_size, n_step]
        attention_weights = torch.bmm(encoder_input, hidden).squeeze(2) # [batch_size, seq_len, 1]
        soft_attention_weights = F.softmax(attention_weights, 1)

        # contex:[batch_size, hidden_size * num_directions(=1)]
        context = torch.bmm(encoder_input.transpose(1,2), soft_attention_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attention_weights

    # def attention_layer(self, encoder_input, final_state):
    #     # encoder_input: [batch_size, seq_len, hidden_size * num_directions(=1)]
    #     batch_size = len(encoder_input) # 32
    #     # hidden : [batch_size, hidden_size * num_directions(=1), n_layer(=1)]
    #     hidden = final_state[0].unsqueeze(1).repeat(1, 40, 1)  # (32,40,300)
    #     # attention_weights: [batch_size, n_step]
    #     attention_weights = torch.bmm(hidden, encoder_input.transpose(1,2))
    #     # attention_weights = attention_weights.squeeze(2) # (32,40,256)
    #     # soft_attention_weights = F.softmax(attention_weights, 1)
    #     soft_attention_weights = torch.softmax(attention_weights, dim=-1)
    #     # contex:[batch_size, hidden_size * num_directions(=1)]
    #     context = torch.bmm(soft_attention_weights, encoder_input)#(32,40,256) (32,256,300)-> (32,40,256)
    #     return context, soft_attention_weights


    def forward(self, input, rnn="gru"):
        input = input.transpose(0,1) # input: [seq_len, batch_size, embedding_dim]

        # output: [seq_len, batch_size, hidden_size * num_directions(=2)]
        # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=1), batch_size, hidden_size]
        if rnn == "gru":
            output, final_hidden_state = self.gru(input)
        else:
            output, final_hidden_state = self.lstm(input)
        output = output.transpose(0, 1) # output: [batch_size, seq_len, hidden_size * num_directions(=1)]

        # attention_output:[batch_size, hidden_size * num_directions(=1)]
        # attention_weight:[batch_size, n_step]
        context, soft_attention_weights = self.attention_layer(output, final_hidden_state)
        return output, context, soft_attention_weights



# class AddNorm(nn.Module):
#     """残差连接后进行层规范化"""
#     def __init__(self, normalized_shape, dropout, **kwargs):
#         super(AddNorm, self).__init__(**kwargs)
#         self.dropout = nn.Dropout(dropout)
#         self.ln = nn.LayerNorm(normalized_shape)
#
#     def forward(self, X, Y, Z):
#         return self.ln(self.dropout(Y) + Z + X)

class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, inputs):
        return inputs


class Dice(nn.Module):
    """The Data Adaptive Activation Function in DIN,which can be viewed as a generalization of PReLu and can adaptively
    adjust the rectified point according to distribution of input data.

    Input shape:
        - 2 dims: [batch_size, embedding_size(features)]
        - 3 dims: [batch_size, num_features, embedding_size(features)]

    Output shape:
        - Same shape as input.

    References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the
        24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.]
        (https://arxiv.org/pdf/1706.06978.pdf)
        - https://github.com/zhougr1993/DeepInterestNetwork, https://github.com/fanoping/DIN-pytorch
    """

    def __init__(self, emb_size, dim=2, epsilon=1e-8, device='cpu'):
        super(Dice, self).__init__()
        assert dim == 2 or dim == 3

        self.bn = nn.BatchNorm1d(emb_size, eps=epsilon)
        self.sigmoid = nn.Sigmoid()
        self.dim = dim

        # wrap alpha in nn.Parameter to make it trainable
        if self.dim == 2:
            self.alpha = nn.Parameter(torch.zeros((emb_size,)).to(device))
        else:
            self.alpha = nn.Parameter(torch.zeros((emb_size, 1)).to(device))

    def forward(self, x):
        assert x.dim() == self.dim
        if self.dim == 2:
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
        else:
            x = torch.transpose(x, 1, 2)
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
            out = torch.transpose(out, 1, 2)
        return out


def activation_layer(act_name, hidden_size=None, dice_dim=2):
    """Construct activation layers

    Args:
        act_name: str or nn.Module, name of activation function
        hidden_size: int, used for Dice activation
        dice_dim: int, used for Dice activation
    Return:
        act_layer: activation layer
    """
    act_layer = None
    if isinstance(act_name, str):
        if act_name.lower() == 'sigmoid':
            act_layer = nn.Sigmoid()
        elif act_name.lower() == 'linear':
            act_layer = Identity()
        elif act_name.lower() == 'relu':
            act_layer = nn.ReLU(inplace=True)
        elif act_name.lower() == 'dice':
            assert dice_dim
            act_layer = Dice(hidden_size, dice_dim)
        elif act_name.lower() == 'prelu':
            act_layer = nn.PReLU()
    elif issubclass(act_name, nn.Module):
        act_layer = act_name()
    else:
        raise NotImplementedError
    return act_layer


class AttentionSequencePoolingLayer(nn.Module):
    """The Attentional sequence pooling operation used in DIN & DIEN.

        Arguments
          - **att_hidden_units**:list of positive integer, the attention net layer number and units in each layer.

          - **att_activation**: Activation function to use in attention net.

          - **weight_normalization**: bool.Whether normalize the attention score of local activation unit.

          - **supports_masking**:If True,the input need to support masking.

        References
          - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]
          Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining.
          ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
      """

    def __init__(self, att_hidden_units=(80, 40), att_activation='sigmoid', weight_normalization=False,
                 return_score=False, supports_masking=False, embedding_dim=4, **kwargs):
        super(AttentionSequencePoolingLayer, self).__init__()
        self.return_score = return_score
        self.weight_normalization = weight_normalization
        self.supports_masking = supports_masking
        self.local_att = LocalActivationUnit(hidden_units=att_hidden_units, embedding_dim=embedding_dim,
                                             activation=att_activation,
                                             dropout_rate=0, use_bn=False)

    def forward(self, query, keys, keys_length, mask=None):
        """
        Input shape
          - A list of three tensor: [query,keys,keys_length]

          - query is a 3D tensor with shape:  ``(batch_size, 1, embedding_size)``

          - keys is a 3D tensor with shape:   ``(batch_size, T, embedding_size)``

          - keys_length is a 2D tensor with shape: ``(batch_size, 1)``

        Output shape
          - 3D tensor with shape: ``(batch_size, 1, embedding_size)``.
        """
        batch_size, max_length, _ = keys.size()

        # Mask
        if self.supports_masking:
            if mask is None:
                raise ValueError("When supports_masking=True,input must support masking")
            keys_masks = mask.unsqueeze(1)
        else:
            keys_masks = torch.arange(max_length, device=keys_length.device,
                                      dtype=keys_length.dtype).repeat(batch_size, 1)  # [B, T]
            keys_masks = keys_masks < keys_length.view(-1, 1)  # 0, 1 mask
            keys_masks = keys_masks.unsqueeze(1)  # [B, 1, T]

        attention_score = self.local_att(query, keys)  # [B, T, 1]

        outputs = torch.transpose(attention_score, 1, 2)  # [B, 1, T]

        if self.weight_normalization:
            paddings = torch.ones_like(outputs) * (-2 ** 32 + 1)
        else:
            paddings = torch.zeros_like(outputs)

        outputs = torch.where(keys_masks, outputs, paddings)  # [B, 1, T]

        # Scale
        # outputs = outputs / (keys.shape[-1] ** 0.05)

        if self.weight_normalization:
            outputs = F.softmax(outputs, dim=-1)  # [B, 1, T]

        if not self.return_score:
            # Weighted sum
            outputs = torch.matmul(outputs, keys)  # [B, 1, E]

        return outputs


class LocalActivationUnit(nn.Module):
    """The LocalActivationUnit used in DIN with which the representation of
        user interests varies adaptively given different candidate items.

    Input shape
        - A list of two 3D tensor with shape:  ``(batch_size, 1, embedding_size)`` and
        ``(batch_size, T, embedding_size)``

    Output shape
        - 3D tensor with shape: ``(batch_size, T, 1)``.

    Arguments
        - **hidden_units**:list of positive integer, the attention net layer number and units in each layer.

        - **activation**: Activation function to use in attention net.

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix of attention net.

        - **dropout_rate**: float in [0,1). Fraction of the units to dropout in attention net.

        - **use_bn**: bool. Whether use BatchNormalization before activation or not in attention net.

        - **seed**: A Python integer to use as random seed.

    References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]
        Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining.
        ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
    """

    def __init__(self, hidden_units=(64, 32), embedding_dim=4, activation='sigmoid', dropout_rate=0, dice_dim=3,
                 l2_reg=0, use_bn=False):
        super(LocalActivationUnit, self).__init__()

        self.dnn = DNN(inputs_dim=4 * embedding_dim,
                       hidden_units=hidden_units,
                       activation=activation,
                       l2_reg=l2_reg,
                       dropout_rate=dropout_rate,
                       dice_dim=dice_dim,
                       use_bn=use_bn)

        self.dense = nn.Linear(hidden_units[-1], 1)

    def forward(self, query, user_behavior):
        # query ad            : size -> batch_size * 1 * embedding_size
        # user behavior       : size -> batch_size * time_seq_len * embedding_size
        user_behavior_len = user_behavior.size(1)

        queries = query.expand(-1, user_behavior_len, -1)

        attention_input = torch.cat([queries, user_behavior, queries - user_behavior, queries * user_behavior],
                                    dim=-1)  # as the source code, subtraction simulates verctors' difference
        attention_output = self.dnn(attention_input)

        attention_score = self.dense(attention_output)  # [B, T, 1]

        return attention_score


class DNN(nn.Module):
    """The Multi Layer Percetron

      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``.
        The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.

      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``.
        For instance, for a 2D input with shape ``(batch_size, input_dim)``,
        the output would have shape ``(batch_size, hidden_size[-1])``.

      Arguments
        - **inputs_dim**: input feature dimension.

        - **hidden_units**:list of positive integer, the layer number and units in each layer.

        - **activation**: Activation function to use.

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.

        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.

        - **use_bn**: bool. Whether use BatchNormalization before activation or not.

        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, inputs_dim, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False,
                 init_std=0.0001, dice_dim=3, seed=1024, device='cpu'):
        super(DNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        hidden_units = [inputs_dim] + list(hidden_units)

        self.linears = nn.ModuleList(
            [nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        if self.use_bn:
            self.bn = nn.ModuleList(
                [nn.BatchNorm1d(hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        self.activation_layers = nn.ModuleList(
            [activation_layer(activation, hidden_units[i + 1], dice_dim) for i in range(len(hidden_units) - 1)])

        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

        self.to(device)

    def forward(self, inputs):
        deep_input = inputs

        for i in range(len(self.linears)):

            fc = self.linears[i](deep_input)

            if self.use_bn:
                fc = self.bn[i](fc)

            fc = self.activation_layers[i](fc)

            fc = self.dropout(fc)
            deep_input = fc
        return deep_input


class AGRUCell(nn.Module):
    """ Attention based GRU (AGRU)

        Reference:
        -  Deep Interest Evolution Network for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1809.03672, 2018.
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(AGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        # (W_ir|W_iz|W_ih)
        self.weight_ih = nn.Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.register_parameter('weight_ih', self.weight_ih)
        # (W_hr|W_hz|W_hh)
        self.weight_hh = nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        self.register_parameter('weight_hh', self.weight_hh)
        if bias:
            # (b_ir|b_iz|b_ih)
            self.bias_ih = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter('bias_ih', self.bias_ih)
            # (b_hr|b_hz|b_hh)
            self.bias_hh = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter('bias_hh', self.bias_hh)
            for tensor in [self.bias_ih, self.bias_hh]:
                nn.init.zeros_(tensor, )
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

    def forward(self, inputs, hx, att_score):
        gi = F.linear(inputs, self.weight_ih, self.bias_ih)
        gh = F.linear(hx, self.weight_hh, self.bias_hh)
        i_r, _, i_n = gi.chunk(3, 1)
        h_r, _, h_n = gh.chunk(3, 1)

        reset_gate = torch.sigmoid(i_r + h_r)
        # update_gate = torch.sigmoid(i_z + h_z)
        new_state = torch.tanh(i_n + reset_gate * h_n)

        att_score = att_score.view(-1, 1)
        hy = (1. - att_score) * hx + att_score * new_state
        return hy


class AUGRUCell(nn.Module):
    """ Effect of GRU with attentional update gate (AUGRU)

        Reference:
        -  Deep Interest Evolution Network for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1809.03672, 2018.
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(AUGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        # (W_ir|W_iz|W_ih)
        self.weight_ih = nn.Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.register_parameter('weight_ih', self.weight_ih)
        # (W_hr|W_hz|W_hh)
        self.weight_hh = nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        self.register_parameter('weight_hh', self.weight_hh)
        if bias:
            # (b_ir|b_iz|b_ih)
            self.bias_ih = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter('bias_ih', self.bias_ih)
            # (b_hr|b_hz|b_hh)
            self.bias_hh = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter('bias_ih', self.bias_hh)
            for tensor in [self.bias_ih, self.bias_hh]:
                nn.init.zeros_(tensor, )
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

    def forward(self, inputs, hx, att_score):
        gi = F.linear(inputs, self.weight_ih, self.bias_ih)
        gh = F.linear(hx, self.weight_hh, self.bias_hh)
        i_r, i_z, i_n = gi.chunk(3, 1)
        h_r, h_z, h_n = gh.chunk(3, 1)

        reset_gate = torch.sigmoid(i_r + h_r)
        update_gate = torch.sigmoid(i_z + h_z)
        new_state = torch.tanh(i_n + reset_gate * h_n)

        att_score = att_score.view(-1, 1)
        update_gate = att_score * update_gate
        hy = (1. - update_gate) * hx + update_gate * new_state
        return hy


class DynamicGRU(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, gru_type='AGRU'):
        super(DynamicGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        if gru_type == 'AGRU':
            self.rnn = AGRUCell(input_size, hidden_size, bias)
        elif gru_type == 'AUGRU':
            self.rnn = AUGRUCell(input_size, hidden_size, bias)

    def forward(self, inputs, att_scores=None, hx=None):
        if not isinstance(inputs, PackedSequence) or not isinstance(att_scores, PackedSequence):
            raise NotImplementedError("DynamicGRU only supports packed input and att_scores")

        inputs, batch_sizes, sorted_indices, unsorted_indices = inputs
        att_scores, _, _, _ = att_scores

        max_batch_size = int(batch_sizes[0])
        if hx is None:
            hx = torch.zeros(max_batch_size, self.hidden_size, dtype=inputs.dtype, device=inputs.device)

        outputs = torch.zeros(inputs.size(0), self.hidden_size, dtype=inputs.dtype, device=inputs.device)

        begin = 0
        for batch in batch_sizes:
            new_hx = self.rnn(inputs[begin:begin + batch], hx[0:batch], att_scores[begin:begin + batch])
            outputs[begin:begin + batch] = new_hx
            hx = new_hx
            begin += batch
        return PackedSequence(outputs, batch_sizes, sorted_indices, unsorted_indices)
