import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from model.aug import sim_global

LOG2PI = math.log(2 * math.pi)


class GMDSSL(nn.Module):
    def __init__(self, in_features, channels, num_comp):
        super(GMDSSL, self).__init__()
        self.in_features = in_features
        self.num_comp = num_comp
        self.l2norm = lambda x: F.normalize(x, dim=1, p=2)
        self.gamma = nn.Sequential(
            nn.Linear(in_features * channels, num_comp, bias=False),
            nn.Softmax(dim=-1)
        )
        self.softmax = nn.Softmax(dim=-1)
        self.sigma = nn.Conv1d(in_channels=in_features, out_channels=num_comp, kernel_size=1)
        self.mu = nn.Conv1d(in_channels=in_features, out_channels=num_comp, kernel_size=1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def get_GaussianPara(self, rep):
        '''
        :param rep: representation, [b,c,m,n,t]
        :return gammas: membership score, [b,k]
        :return sigmas, mus: Gaussian parameters (mean and variance), [b,k,c]
        '''
        b, c, n, _ = rep.shape
        gammas = self.gamma(rep.reshape(b, -1))
        mus = self.mu(rep.permute(0, 2, 3, 1).reshape(b, -1, c))
        sigmas = torch.exp(self.sigma(rep.permute(0, 2, 3, 1).reshape(b, -1, c)))
        return gammas.unsqueeze(1), mus.permute(0, 2, 1), sigmas.permute(0, 2, 1)

    def get_logPdf(self, rep, mus, sigmas):
        '''
        :param rep: representation, [b,c,m*n*t]
        :param sigmas, mus: Gaussian parameters, [b,c,k]
        return log_component_prob: log PDF, [b, m*n*t, k]
        '''
        h = rep.unsqueeze(-1)
        mus = mus.unsqueeze(2)
        sigmas = sigmas.unsqueeze(2)
        log_component_prob = -torch.log(sigmas) - 0.5 * LOG2PI - 0.5 * torch.pow((h - mus) / sigmas, 2)
        # torch.prod(log_component_prob, 1) may cause inf
        return self.l2norm(torch.prod(log_component_prob, 1))

    def forward(self, rep, rep_aug):
        """
        rep : [b, t, n, c]
        """

        rep = rep.permute(0, 3, 2, 1)
        rep_aug = rep_aug.permute(0, 3, 2, 1)
        b, c, n, _ = rep_aug.shape

        rep = self.l2norm(rep)
        rep_aug = self.l2norm(rep_aug)
        gammas_aug, mus_aug, sigmas_aug = self.get_GaussianPara(rep_aug)
        # get log Pdf with the original representation H as a self-supervised signal
        log_component_prob_aug = self.get_logPdf(rep.reshape(b, c, -1), mus_aug, sigmas_aug)
        log_prob_aug = log_component_prob_aug + torch.log(gammas_aug)
        # calculate loss
        loss = -torch.mean(torch.log(torch.sum(log_prob_aug.exp(), dim=-1)))
        return loss

class TCSSL(nn.Module):
    '''Temporal heterogeneity modeling in a contrastive manner.
    '''

    def __init__(self, c_in, batch_size, num_nodes, device):
        super(TCSSL, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_nodes, c_in))  # representation weights
        self.W2 = nn.Parameter(torch.FloatTensor(num_nodes, c_in))
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))

        self.read = AvgReadout()
        self.disc = Discriminator(c_in)
        self.b_xent = nn.BCEWithLogitsLoss()

        lbl_rl = torch.ones(batch_size, num_nodes)
        lbl_fk = torch.zeros(batch_size, num_nodes)
        lbl = torch.cat((lbl_rl, lbl_fk), dim=1)
        if device == 'cuda':
            self.lbl = lbl.cuda()

        self.n = batch_size

    def forward(self, z1, z2):
        '''
        :param z1, z2 (tensor): shape nlvc, i.e., (batch_size, seq_len, num_nodes, feat_dim)
        :return loss: loss of generative branch. nclv
        '''
        h = (z1 * self.W1 + z2 * self.W2).squeeze(1)  # nlvc->nvc
        s = self.read(h)  # s: summary of h, nc

        # select another region in batch
        idx = torch.randperm(self.n)
        shuf_h = h[idx]

        logits = self.disc(s, h, shuf_h)
        loss = self.b_xent(logits, self.lbl)
        return loss


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()
        self.sigm = nn.Sigmoid()

    def forward(self, h):
        '''Apply an average on graph.
        :param h: hidden representation, (batch_size, num_nodes, feat_dim)
        :return s: summary, (batch_size, feat_dim)
        '''
        s = torch.mean(h, dim=1)
        s = self.sigm(s)
        return s


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.net = nn.Bilinear(n_h, n_h, 1)  # similar to score of CPC

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, summary, h_rl, h_fk):
        '''
        :param s: summary, (batch_size, feat_dim)
        :param h_rl: real hidden representation (w.r.t summary),
            (batch_size, num_nodes, feat_dim)
        :param h_fk: fake hidden representation
        :return logits: prediction scores, (batch_size, num_nodes, 2)
        '''
        s = torch.unsqueeze(summary, dim=1)
        s = s.expand_as(h_rl).contiguous()

        # score of real and fake, (batch_size, num_nodes)
        sc_rl = torch.squeeze(self.net(h_rl, s), dim=2)
        sc_fk = torch.squeeze(self.net(h_fk, s), dim=2)

        logits = torch.cat((sc_rl, sc_fk), dim=1)

        return logits

class STEncoder(nn.Module):
    def __init__(self, Kt, Ks, blocks, input_length, num_nodes, droprate=0.1):
        super(STEncoder, self).__init__()
        self.Ks = Ks
        c = blocks[0]
        self.tconv11 = TemporalConvLayer(Kt, c[0], c[1], "GLU")
        self.pooler = Pooler(input_length - (Kt - 1), c[1])

        self.sconv12 = SpatioConvLayer(Ks, c[1], c[1])
        self.tconv13 = TemporalConvLayer(Kt, c[1], c[2])
        self.ln1 = nn.LayerNorm([num_nodes, c[2]])
        self.dropout1 = nn.Dropout(droprate)

        c = blocks[1]
        self.tconv21 = TemporalConvLayer(Kt, c[0], c[1], "GLU")

        self.sconv22 = SpatioConvLayer(Ks, c[1], c[1])
        self.tconv23 = TemporalConvLayer(Kt, c[1], c[2])
        self.ln2 = nn.LayerNorm([num_nodes, c[2]])
        self.dropout2 = nn.Dropout(droprate)

        self.s_sim_mx = None
        self.t_sim_mx = None

        out_len = input_length - 2 * (Kt - 1) * len(blocks)
        self.out_conv = TemporalConvLayer(out_len, c[2], c[2], "GLU")
        self.ln3 = nn.LayerNorm([num_nodes, c[2]])
        self.dropout3 = nn.Dropout(droprate)
        self.receptive_field = input_length + Kt - 1

    def forward(self, x0, graph):
        in_len = x0.size(1)  # x0, nlvc
        if in_len < self.receptive_field:
            x = F.pad(x0, (0, 0, 0, 0, self.receptive_field - in_len, 0))
        else:
            x = x0
        x = x.permute(0, 3, 1, 2)  # b c t v

        ## ST block 1
        x = self.tconv11(x)  # nclv
        x, x_agg, self.t_sim_mx = self.pooler(x)
        self.s_sim_mx = sim_global(x_agg, sim_type='cos')

        x = self.sconv12(x, graph)  # nclv
        x = self.tconv13(x)
        x = self.dropout1(self.ln1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))

        ## ST block 2
        x = self.tconv21(x)
        x = self.sconv22(x, graph)
        x = self.tconv23(x)
        x = self.dropout2(self.ln2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))

        ## out block
        x = self.out_conv(x)  # ncl(=1)v
        x = self.dropout3(self.ln3(x.permute(0, 2, 3, 1)))  # nlvc
        return x  # nl(=1)vc

class Align(nn.Module):
    def __init__(self, c_in, c_out):
        '''Align the input and output.
        '''
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)  # filter=(1,1), similar to fc

    def forward(self, x):  # x: (n,c,l,v)
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            x_ = F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
            return x_
        return x


class TemporalConvLayer(nn.Module):
    def __init__(self, kt, c_in, c_out, act="relu"):
        super(TemporalConvLayer, self).__init__()
        self.kt = kt
        self.act = act
        self.c_out = c_out
        self.align = Align(c_in, c_out)
        if self.act == "GLU":
            self.conv = nn.Conv2d(c_in, c_out * 2, (kt, 1), 1)
        else:
            self.conv = nn.Conv2d(c_in, c_out, (kt, 1), 1)

    def forward(self, x):
        """
        :param x: (n,c,l,v)
        :return: (n,c,l-kt+1,v)
        """
        x_in = self.align(x)[:, :, self.kt - 1:, :]
        if self.act == "GLU":
            x_conv = self.conv(x)
            return (x_conv[:, :self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[:, self.c_out:, :, :])
        if self.act == "sigmoid":
            return torch.sigmoid(self.conv(x) + x_in)
        return torch.relu(self.conv(x) + x_in)

    class TemporalConvLayer(nn.Module):
        def __init__(self, kt, c_in, c_out, d_pred, act="relu"):
            super(TemporalConvLayer, self).__init__()
            self.kt = kt
            self.act = act
            self.c_out = c_out
            self.align = Align(c_in, c_out)
            if self.act == "GLU":
                self.conv = nn.Conv2d(c_in, c_out * 2, (kt, 1), 1)
            else:
                self.conv = nn.Conv2d(c_in, c_out, (kt, 1), 1)

        def forward(self, x):
            """
            :param x: (n,c,l,v)
            :return: (n,c,l-kt+1,v)
            """
            x_in = self.align(x)[:, :, self.kt - 1:, :]
            if self.act == "GLU":
                x_conv = self.conv(x)
                return (x_conv[:, :self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[:, self.c_out:, :, :])
            if self.act == "sigmoid":
                return torch.sigmoid(self.conv(x) + x_in)
            return torch.relu(self.conv(x) + x_in)


class Pooler(nn.Module):
    '''Pool token representations of region time series to the region level.'''

    def __init__(self, n_query, d_model, agg='avg'):
        """
        :param n_query: Number of queries.
        :param d_model: Dimensionality of the model.
        """
        super(Pooler, self).__init__()

        # Attention matrix
        self.att = FCLayer(d_model, n_query)
        self.align = Align(d_model, d_model)
        self.softmax = nn.Softmax(dim=2)  # Apply softmax over the sequence length dimension (nclv)

        self.d_model = d_model
        self.n_query = n_query
        if agg == 'avg':
            self.agg = nn.AvgPool2d(kernel_size=(n_query, 1), stride=1)
        elif agg == 'max':
            self.agg = nn.MaxPool2d(kernel_size=(n_query, 1), stride=1)
        else:
            raise ValueError('Pooler supports only "avg" and "max" aggregation methods.')

    def forward(self, x):
        """
        Compute the overall traffic pattern embedding u and temporal similarity matrix p.

        :param x: Output from the first TC of the spatiotemporal encoder, with shape (batch_size, feature_dim, time_steps, num_nodes).
        :return x: Latent features for the next input into SC, with shape (batch_size, feature_dim, n_query, num_nodes).
        :return x_agg: Overall traffic change pattern embedding u, with shape (batch_size, num_nodes, feature_dim).
        :return A: Correlation between specific time step traffic patterns and overall traffic change pattern, with shape (time_steps, batch_size, num_nodes).
        """
        x_in = self.align(x)[:, :, -self.n_query:, :]  # ncqv
        # Calculate the attention matrix A using key x
        A = self.att(x)  # x: nclv, A: nqlv
        A = F.softmax(A, dim=2)  # nqlv

        # Calculate region embedding using the attention matrix A
        x = torch.einsum('nclv,nqlv->ncqv', x, A)
        x_agg = self.agg(x).squeeze(2)  # ncqv -> ncv
        x_agg = torch.einsum('ncv->nvc', x_agg)  # ncv -> nvc

        # Calculate the temporal similarity (prob)
        A = torch.einsum('nqlv->lnqv', A)
        A = self.softmax(self.agg(A).squeeze(2))  # A: lnqv -> lnv
        return torch.relu(x + x_in), x_agg.detach(), A.detach()

class MLP(nn.Module):
    def __init__(self, c_in, c_out):
        super(MLP, self).__init__()
        self.fc1 = FCLayer(c_in, int(c_in // 2))
        self.fc2 = FCLayer(int(c_in // 2), c_out)

    def forward(self, x):
        x = torch.tanh(self.fc1(x.permute(0, 3, 1, 2)))  # btnc->bctn
        x = self.fc2(x).permute(0, 2, 3, 1)  # bctn->btnc
        return x


class FCLayer(nn.Module):
    def __init__(self, c_in, c_out):
        super(FCLayer, self).__init__()
        self.linear = nn.Conv2d(c_in, c_out, (1, 1))

    def forward(self, x):
        return self.linear(x)


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,nwv->ncwl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class SpatioConvLayer(nn.Module):
    def __init__(self, K, c_in, c_out, dropout=0.2):
        super(SpatioConvLayer, self).__init__()
        self.nconv = nconv()
        c_in = (K + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.K = K

    def forward(self, x, support):
        x0 = x.permute(0, 1, 3, 2)
        out = [x0]

        x1 = self.nconv(x0, support)
        out.append(x1)
        for k in range(2, self.K + 1):
            x2 = self.nconv(x1, support)
            out.append(x2)
            x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h.permute(0, 1, 3, 2)


class FeedForward(nn.Module):
    def __init__(self, fea, res_ln=False):
        super(FeedForward, self).__init__()
        self.res_ln = res_ln
        self.L = len(fea) - 1
        self.linear = nn.ModuleList([nn.Linear(fea[i], fea[i + 1]) for i in range(self.L)])
        self.ln = nn.LayerNorm(fea[self.L], elementwise_affine=False)

    def forward(self, inputs):
        x = inputs
        for i in range(self.L):
            x = self.linear[i](x)
            if i != self.L - 1:
                x = F.relu(x)
        if self.res_ln:
            x += inputs
            x = self.ln(x)
        return x


class TemporalAttention(nn.Module):
    def __init__(self, heads, dims):
        super(TemporalAttention, self).__init__()
        features = heads * dims
        self.h = heads
        self.d = dims

        self.qfc = FeedForward([features, features])
        self.kfc = FeedForward([features, features])
        self.vfc = FeedForward([features, features])
        self.ofc = FeedForward([features, features])

        self.ln = nn.LayerNorm(features, elementwise_affine=False)
        self.ff = FeedForward([features, features, features], True)

    def forward(self, x, te, Mask=True):
        '''
        x: [B,T,N,F]
        te: [B,1,N,F]
        return: [B,T,N,F]
        '''
        x += te

        query = self.qfc(x)  # [B,T,N,F]
        key = self.kfc(x)  # [B,T,N,F]
        value = self.vfc(x)  # [B,T,N,F]

        query = torch.cat(torch.split(query, self.d, -1), 0).permute(0, 2, 1, 3)  # [k*B,T,N,d]
        key = torch.cat(torch.split(key, self.d, -1), 0).permute(0, 2, 3, 1)  # [k*B,N,d,T]
        value = torch.cat(torch.split(value, self.d, -1), 0).permute(0, 2, 1, 3)  # [k*B,N,T,d]

        attention = torch.matmul(query, key)  # [k*B,N,T,T]
        attention /= (self.d ** 0.5)  # scaled

        if Mask:
            batch_size = x.shape[0]
            num_steps = x.shape[1]
            num_vertexs = x.shape[2]
            mask = torch.ones(num_steps, num_steps).to(x.device)  # [T,T]
            mask = torch.tril(mask)  # [T,T]
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)  # [1,1,T,T]
            mask = mask.repeat(self.h * batch_size, num_vertexs, 1, 1)  # [k*B,N,T,T]
            mask = mask.to(torch.bool)
            zero_vec = (-2 ** 15 + 1) * torch.ones_like(attention).to(x.device)  # [k*B,N,T,T]
            attention = torch.where(mask, attention, zero_vec)

        attention = F.softmax(attention, -1)  # [k*B,N,T,T]

        value = torch.matmul(attention, value)  # [k*B,N,T,d]

        value = torch.cat(torch.split(value, value.shape[0] // self.h, 0), -1).permute(0, 2, 1, 3)  # [B,T,N,F]
        value = self.ofc(value)
        value += x

        value = self.ln(value)

        return self.ff(value), attention
class TemEmbedding(nn.Module):
    def __init__(self, I, D):
        super(TemEmbedding, self).__init__()
        self.ff_te = FeedForward([I, D, D])

    def forward(self, TE, T):
        '''
        TE: [B,T,2]
        return: [B,T,N,D]
        '''
        # temporal embedding
        dayofweek = torch.empty(TE.shape[0], TE.shape[1], 7).to(TE.device)  # [B,T,7]
        timeofday = torch.empty(TE.shape[0], TE.shape[1], T).to(TE.device)  # [B,T,288]
        for i in range(TE.shape[0]):
            dayofweek[i] = F.one_hot(TE[..., 0][i].to(torch.int64) % 7, 7)
        for j in range(TE.shape[0]):
            timeofday[j] = F.one_hot(TE[..., 1][j].to(torch.int64) % T, T)
        TE = torch.cat((dayofweek, timeofday), dim=-1)  # [B,T,295]
        TE = TE.unsqueeze(dim=2)  # [B,T,1,295]
        TE = self.ff_te(TE)  # [B,T,1,F]

        return TE  # ][B,T,N,F