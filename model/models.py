import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils import masked_mae_loss
from model.aug import (
    aug_traffic,
)
from model.layers import (
    STEncoder,
    MLP,
    FeedForward,
    TemporalAttention,
    TemEmbedding,
    GMDSSL,
    TCSSL,
)

class D2STCNet(nn.Module):
    def __init__(self, args):
        super(D2STCNet, self).__init__()
        # spatial temporal encoder
        self.encoder = STEncoder(Kt=3, Ks=3, blocks=[[args.d_model//4, int(args.d_model//2), args.d_model], [args.d_model, int(args.d_model//2), args.d_model]],
                                 input_length=args.input_length, num_nodes=args.num_nodes, droprate=args.dropout)

        # traffic flow prediction branch
        self.mlp = MLP(args.d_model, args.d_output)
        self.thm = TCSSL(args.d_model, args.batch_size, args.num_nodes, args.device)
        self.hm = GMDSSL(args.num_nodes, args.d_model, args.nmb_prototype)
        self.mae = masked_mae_loss(mask_value=5.0)

        self.args = args
        # dynamic graph core tensor
        self.time_of_day = args.time_of_day
        self.nodevec_t = nn.Parameter(torch.randn(args.time_of_day, 32).to(args.device), requires_grad=True).to(args.device)
        self.nodevec_n1 = nn.Parameter(torch.randn(args.num_nodes, 32).to(args.device), requires_grad=True).to(args.device)
        self.nodevec_n2 = nn.Parameter(torch.randn(args.num_nodes, 32).to(args.device), requires_grad=True).to(args.device)
        self.nodevec_core = nn.Parameter(torch.randn(32, 32, 32).to(args.device), requires_grad=True).to(args.device)
        self.aggLayer = nn.Conv2d(in_channels=args.time_of_day, out_channels=1, kernel_size=1, bias=False)

        # attention layer
        self.startEmbedding = FeedForward([args.d_input, args.d_model//4])
        self.endEmbedding = FeedForward([args.d_model//4, args.d_input])
        self.te = TemEmbedding(self.time_of_day+7, args.d_model//4)
        self.ta = TemporalAttention(1, args.d_model//4)

        # x_h channel
        self.ta_h = TemporalAttention(1, args.d_model//4)

        # fusion layer
        self.conv = nn.Conv2d(in_channels=args.d_model * 2, out_channels=args.d_model, kernel_size=(1,1), bias=False)
        self.conv_norm = nn.LayerNorm(args.d_model)
        self.conv_drop = nn.Dropout(args.dropout)

    def dgconstruct(self, t_embedding, source_embedding, target_embedding, core_embedding):
        adp = torch.einsum('ai, ijk->ajk', t_embedding, core_embedding)
        adp = torch.einsum('bj, ajk->abk', source_embedding, adp)
        adp = torch.einsum('ck, abk->abc', target_embedding, adp)
        adp = F.softmax(F.relu(adp), dim=2)
        return adp

    def forward(self, view1, view2, view3, idx):
        """
        idx: [b, t, 2](time, day)
        """

        batch_size, time, node, dim = view1.shape
        te = self.te(idx, self.time_of_day) # temporal embedding
        view1 = self.startEmbedding(view1)
        view1, attn_t = self.ta(view1, te)
        attn_t = attn_t.view(batch_size, node, time, time)  # b n t t
        attn_t = attn_t.diagonal(dim1=-2, dim2=-1)
        attn_t = attn_t.permute(2, 0, 1)

        time = idx[:, 0, 0]
        adp = self.dgconstruct(self.nodevec_t[time], self.nodevec_n1, self.nodevec_n2, self.nodevec_core)

        repr1 = self.encoder(view1, adp) # view1: b t n v,

        graph2 = self.dgconstruct(self.nodevec_t, self.nodevec_n1, self.nodevec_n2, self.nodevec_core)
        graph2 = self.aggLayer(graph2.unsqueeze(0)).squeeze(0)
        graph2 = graph2.expand(batch_size, -1, -1)

        view2 = aug_traffic(attn_t, view2, percent=self.args.percent)
        view2 = self.startEmbedding(view2)
        view2, _ = self.ta(view2, te)

        repr2 = self.encoder(view2, graph2)

        view3 = self.startEmbedding(view3)
        view3, _ = self.ta_h(view3, te)

        repr3 = self.encoder(view3, adp)
        repr3 = torch.concat([repr2, repr3], dim=-1)
        repr3 = repr3.permute(0, 3, 1, 2)
        repr3 = self.conv(repr3)
        repr3 = self.conv_norm(repr3.permute(0, 2, 3, 1))
        repr3 = self.conv_drop(repr3)

        return repr1, repr3

    def my_sim_global(self, v1, v2, sim_type='cos'):
        """
        :param v1: [nodes, hidden]
        :param v2: [nodes, hidden]
        """
        cos_scaling1 = torch.norm(v1, p=2, dim=1) ** -1
        cos_scaling2 = torch.norm(v2, p=2, dim=1) ** -1
        sim = torch.einsum('ni, mi->nm', v1, v2)
        scaling = torch.einsum('i, j->ij', cos_scaling1, cos_scaling2)
        sim = sim * scaling
        return sim.cpu()

    def predict(self, z1, z2):
        '''Predicting future traffic flow.
        :param z1, z2 (tensor): shape nvc
        :return: nlvc, l=1, c=2
        '''
        return self.mlp(z1)  # b t v c

    def loss(self, z1, z2, y_true, scaler, loss_weights):
        l1 = self.pred_loss(z1, z2, y_true, scaler)
        sep_loss = [l1.item()]
        loss = loss_weights[0] * l1

        l2 = self.hm(z1, z2)
        sep_loss.append(l2.item())
        loss += loss_weights[1] * l2

        l3 = self.thm(z1, z2)
        sep_loss.append(l3.item())
        loss += loss_weights[2] * l3
        return loss, sep_loss

    def pred_loss(self, z1, z2, y_true, scaler):
        y_pred = scaler.inverse_transform(self.predict(z1, z2))
        y_true = scaler.inverse_transform(y_true)
        if (self.args.yita != -1):
            loss = self.args.yita * self.mae(y_pred[..., 0], y_true[..., 0]) + \
                   (1 - self.args.yita) * self.mae(y_pred[..., 1], y_true[..., 1])
        else:
            loss = self.mae(y_pred, y_true)
        return loss