import torch
import torch.nn as nn
import contrastive_loss as CL


class fea_self_expression(nn.Module):
    def __init__(self, n_samples):
        super(fea_self_expression, self).__init__()
        self.coef = nn.Parameter(1e-4 * torch.ones(n_samples, n_samples, dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        coef = self.coef - torch.diag(torch.diag(self.coef))
        x_rec = torch.mm(coef, x)
        return x_rec, coef


class single_view_contrastive_clustering(nn.Module):
    def __init__(self, n_samples, temperature, q=0.7):
        super(single_view_contrastive_clustering, self).__init__()
        self.fea_self_expression = fea_self_expression(n_samples)
        self.tau = temperature
        self.q = q

    def forward(self, x_v, coef_fusion):
        x_v_rec, coef_v = self.fea_self_expression(x_v)
        coef_v_rec = torch.matmul(coef_fusion, coef_v)
        return x_v_rec, coef_v, coef_v_rec

    def loss(self, x_v, x_v_rec, coef_v, coef_v_rec, coef_fusion, adj_graph, alpha, beta):
        feature_rec_loss = CL.robust_info_loss_embed_fea_rec(x_v_rec, x_v, adj_graph, self.tau, self.q)
        coef_v_reg_loss  = CL.robust_info_loss_coef_rec(coef_v_rec, coef_v, adj_graph, self.tau, self.q)
        coef_v_fusion_loss = CL.robust_info_loss_coef_rec(coef_fusion, coef_v, adj_graph, self.tau, self.q)
        total_loss = feature_rec_loss + alpha * coef_v_reg_loss + beta * coef_v_fusion_loss
        return total_loss, coef_v_fusion_loss


class multi_view_contrastive_clustering(nn.Module):
    def __init__(self, n_samples, n_views, temperature, adj_graph, q=0.7):
        super(multi_view_contrastive_clustering, self).__init__()
        self.n_views = n_views
        self.adj_graph = adj_graph
        self.coef_fusion = nn.Parameter(1e-4 * torch.ones(n_samples, n_samples, dtype=torch.float32),
                                        requires_grad=True)
        self.multi_module = nn.ModuleList(
            [single_view_contrastive_clustering(n_samples, temperature, q=q) for _ in range(n_views)]
        )

    def forward(self, x):
        x_v_rec = []
        coef_v = []
        coef_v_rec = []
        coef_fusion = self.coef_fusion - torch.diag(torch.diag(self.coef_fusion))
        for i in range(self.n_views):
            x_v_rec_i, coef_v_i, coef_v_rec_i = self.multi_module[i](x[i], coef_fusion)
            x_v_rec.append(x_v_rec_i)
            coef_v.append(coef_v_i)
            coef_v_rec.append(coef_v_rec_i)
        return x_v_rec, coef_v, coef_v_rec, coef_fusion

    def loss(self, x, x_v_rec, coef_v, coef_v_rec, coef_fusion, alpha, beta):
        total_loss = 0
        for i in range(self.n_views):
            loss_i, coef_i_fusion_loss = self.multi_module[i].loss(x[i], x_v_rec[i], coef_v[i],
                                                                   coef_v_rec[i], coef_fusion, self.adj_graph,
                                                                   alpha, beta)
            total_loss = total_loss + loss_i
        return total_loss
