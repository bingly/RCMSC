import torch


def cos_sim(A, B):
    # 计算 A 和 B 之间的余弦相似度矩阵
    A_norm = torch.norm(A, dim=1, keepdim=True)  # 计算 A 中每个向量的 L2 范数
    B_norm = torch.norm(B, dim=1, keepdim=True)  # 计算 B 中每个向量的 L2 范数
    similarity_matrix = torch.matmul(A, B.T) / (torch.matmul(A_norm, B_norm.T) + 1e-8)  # 计算余弦相似度
    return similarity_matrix


def info_nec_loss_embed_fea_rec(A, B, adj_graph, temperature):
    similarity_matrix1 = torch.exp(cos_sim(A, B) / temperature)
    positive_score_array1 = torch.sum(similarity_matrix1 * adj_graph[0], dim=1)
    negative_score_array1 = torch.sum(similarity_matrix1 * adj_graph[1], dim=1)

    similarity_matrix2 = torch.exp(cos_sim(A, A) / temperature)
    similarity_matrix2 = similarity_matrix2 - torch.diag(torch.diag(similarity_matrix2))
    positive_score_array2 = torch.sum(similarity_matrix2 * adj_graph[0], dim=1)
    negative_score_array2 = torch.sum(similarity_matrix2 * adj_graph[1], dim=1)

    positive_score_array = positive_score_array1 + positive_score_array2
    negative_score_array = negative_score_array1 + negative_score_array2
    positive_score_array = positive_score_array[positive_score_array > 0]
    negative_score_array = negative_score_array[negative_score_array > 0]

    positive_score = torch.log(positive_score_array).sum()
    negative_score = torch.log(negative_score_array).sum()
    fea_loss = -(positive_score - negative_score)
    return fea_loss


def info_nec_loss_coef_rec(A, B, adj_graph, temperature):
    similarity_matrix1 = torch.exp(cos_sim(A, B) / temperature)
    positive_score_array1 = torch.sum(similarity_matrix1 * adj_graph[0], dim=1)
    negative_score_array1 = torch.sum(similarity_matrix1 * adj_graph[1], dim=1)

    similarity_matrix2 = torch.exp(cos_sim(B, B) / temperature)
    similarity_matrix2 = similarity_matrix2 - torch.diag(torch.diag(similarity_matrix2))
    positive_score_array2 = torch.sum(similarity_matrix2 * adj_graph[0], dim=1)
    negative_score_array2 = torch.sum(similarity_matrix2 * adj_graph[1], dim=1)

    similarity_matrix3 = torch.exp(cos_sim(A, A) / temperature)
    similarity_matrix3 = similarity_matrix3 - torch.diag(torch.diag(similarity_matrix3))
    positive_score_array3 = torch.sum(similarity_matrix3 * adj_graph[0], dim=1)
    negative_score_array3 = torch.sum(similarity_matrix3 * adj_graph[1], dim=1)

    positive_score_array = positive_score_array1 + positive_score_array2 + positive_score_array3
    negative_score_array = negative_score_array1 + negative_score_array2 + negative_score_array3
    positive_score_array = positive_score_array[positive_score_array > 0]
    negative_score_array = negative_score_array[negative_score_array > 0]
    positive_score = torch.log(positive_score_array).sum()
    negative_score = torch.log(negative_score_array).sum()
    coef_loss = -(positive_score - negative_score)

    return coef_loss


def info_nec_loss_fusion(A, B, adj_graph, temperature):
    similarity_matrix1 = torch.exp(cos_sim(A, B) / temperature)
    positive_score_array1 = torch.sum(similarity_matrix1 * adj_graph[0], dim=1)
    negative_score_array1 = torch.sum(similarity_matrix1 * adj_graph[1], dim=1)

    similarity_matrix2 = torch.exp(cos_sim(A, A) / temperature)
    similarity_matrix2 = similarity_matrix2 - torch.diag(torch.diag(similarity_matrix2))
    positive_score_array2 = torch.sum(similarity_matrix2 * adj_graph[0], dim=1)
    negative_score_array2 = torch.sum(similarity_matrix2 * adj_graph[1], dim=1)

    positive_score_array = positive_score_array1 + positive_score_array2
    negative_score_array = negative_score_array1 + negative_score_array2
    positive_score_array = positive_score_array[positive_score_array > 0]
    negative_score_array = negative_score_array[negative_score_array > 0]
    positive_score = torch.log(positive_score_array).sum()
    negative_score = torch.log(negative_score_array).sum()
    fusion_loss = -(positive_score - negative_score)
    return fusion_loss

# === Robust contrastive variants (unified P+ + q-loss) =======================

def _robust_loss_from_pos_neg(positive_score_array, negative_score_array, q: float, eps: float=1e-12):
    # positive_score_array / negative_score_array: (n,) >= 0
    positive_score_array = positive_score_array[positive_score_array > 0]
    negative_score_array = negative_score_array[negative_score_array > 0]
    Gp = positive_score_array
    Gn = negative_score_array + eps
    Pp = Gp / (Gp + Gn)                     # element-wise unified positive probability
    # robust q-loss: mean over samples (scale-friendly)
    loss = (1.0 - torch.pow(Pp + eps, q)) / q
    return loss.mean()

def robust_info_loss_embed_fea_rec(A, B, adj_graph, temperature, q: float):
    # A~B channel
    similarity_matrix1 = torch.exp(cos_sim(A, B) / temperature)
    positive_score_array1 = torch.sum(similarity_matrix1 * adj_graph[0], dim=1)
    negative_score_array1 = torch.sum(similarity_matrix1 * adj_graph[1], dim=1)
    # A~A channel (drop diag)
    similarity_matrix2 = torch.exp(cos_sim(A, A) / temperature)
    similarity_matrix2 = similarity_matrix2 - torch.diag(torch.diag(similarity_matrix2))
    positive_score_array2 = torch.sum(similarity_matrix2 * adj_graph[0], dim=1)
    negative_score_array2 = torch.sum(similarity_matrix2 * adj_graph[1], dim=1)

    positive_score_array = positive_score_array1 + positive_score_array2
    negative_score_array = negative_score_array1 + negative_score_array2
    return _robust_loss_from_pos_neg(positive_score_array, negative_score_array, q)

def robust_info_loss_coef_rec(A, B, adj_graph, temperature, q: float):
    # A~B
    similarity_matrix1 = torch.exp(cos_sim(A, B) / temperature)
    positive_score_array1 = torch.sum(similarity_matrix1 * adj_graph[0], dim=1)
    negative_score_array1 = torch.sum(similarity_matrix1 * adj_graph[1], dim=1)
    # B~B
    similarity_matrix2 = torch.exp(cos_sim(B, B) / temperature)
    similarity_matrix2 = similarity_matrix2 - torch.diag(torch.diag(similarity_matrix2))
    positive_score_array2 = torch.sum(similarity_matrix2 * adj_graph[0], dim=1)
    negative_score_array2 = torch.sum(similarity_matrix2 * adj_graph[1], dim=1)
    # A~A
    similarity_matrix3 = torch.exp(cos_sim(A, A) / temperature)
    similarity_matrix3 = similarity_matrix3 - torch.diag(torch.diag(similarity_matrix3))
    positive_score_array3 = torch.sum(similarity_matrix3 * adj_graph[0], dim=1)
    negative_score_array3 = torch.sum(similarity_matrix3 * adj_graph[1], dim=1)

    positive_score_array = positive_score_array1 + positive_score_array2 + positive_score_array3
    negative_score_array = negative_score_array1 + negative_score_array2 + negative_score_array3
    return _robust_loss_from_pos_neg(positive_score_array, negative_score_array, q)
