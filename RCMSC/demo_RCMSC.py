import os
import time
import yaml
import argparse

os.environ["OMP_NUM_THREADS"] = '1'

import numpy as np
import torch
import torch.optim as optim

import rcmsc as M
import utils as U

def get_args_parser():
    parser = argparse.ArgumentParser(description='RCMSC')

    parser.add_argument('--data_name', type=str, default='3Sources',
                        choices=['3Sources','COIL_20','EYaleB10','MNIST_USPS','BBCSport','100leaves','Hdigit','NUS_WIDE'],
                        help='dataset name')
    parser.add_argument('--seed', type=int, default=10, help='Initializing random seed.')
    parser.add_argument("--epochs", default=100, help='Number of epochs to fine-tuning.')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0005, help='Initializing learning rate.')
    parser.add_argument("--temperature_l", type=float, default=1.0)
    parser.add_argument('--normalized', type=bool, default=False)
    parser.add_argument('--gpu', default='1', type=str, help='GPU device idx.')

    return parser


def train_model(X, model, optimizer, alpha, beta, epochs):
    l = []
    acc_array = []
    nmi_array = []
    f1_array = []
    ari_array = []
    C_best = []
    for epoch in range(epochs+1):
        x_v_rec, coef_v, coef_v_rec, coef_fusion = model(X)
        loss = model.loss(X, x_v_rec, coef_v, coef_v_rec, coef_fusion, alpha, beta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 or epoch == epochs:
            C = coef_fusion.detach().to('cpu').numpy()

            acc, nmi, f1_macro, ari = U.get_cluster_results(C, labels.squeeze(1), n_classes)
            if len(acc_array) > 1 and acc > max(acc_array):
                C_best = C
            acc_array.append(acc)
            nmi_array.append(nmi)
            f1_array.append(f1_macro)
            ari_array.append(ari)
            print("epoch = %d, acc = %f, nmi = %f, f1 = %f, ari = %f, loss = %f" % (
                epoch, acc, nmi, f1_macro, ari, loss.item()))

        l.append(loss.item())

    # U.plot_tsne(coef_fusion.detach().cpu().numpy(), [x[0] for x in labels], title='PCMVSC', db_name='COIL-20')
    best_index = np.argmax(np.array(acc_array))
    acc_max = acc_array[best_index] * 100
    nmi_max = nmi_array[best_index] * 100
    f1_max = f1_array[best_index] * 100
    ari_max = ari_array[best_index] * 100
    return l, acc_max, nmi_max, f1_max, ari_max, C_best


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    # load config file
    config_file = f'./config/{args.data_name}.yaml'
    with open(config_file) as f:
        if hasattr(yaml, "FullLoader"):
            configs = yaml.load(f.read(), Loader=yaml.FullLoader)
        else:
            configs = yaml.load(f.read())

    args = vars(args)
    args.update(configs)
    args = argparse.Namespace(**args)

    print("==========\nArgs:{}\n==========".format(args))
    # torch.cuda.setting
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    U.set_seed(args.seed)

    # load data
    current_directory = os.getcwd()

    X, labels, n_views, n_samples = U.load_data(args.data_name, current_directory)
    X = U.data_normalize_l2(X, n_views)

    n_classes = np.max(np.unique(labels))
    if np.min(np.unique(labels)) == 0:
        n_classes = n_classes + 1

    temperature = 1
    U.write_splitter(args.data_name)
    if isinstance(X[0], torch.Tensor):
        for i in range(n_views):
            X[i] = X[i].detach().to('cpu').numpy()

    positive_adj_graphs = U.adj_graphs(X, n_samples, args.k, 'cosine')  #'euclidean' 'cosine'
    fused_adj_graph = U.fused_adj_graph(positive_adj_graphs, n_samples, n_views)
    adj_graph = torch.tensor(fused_adj_graph, dtype=torch.float32, device=device)

    if not isinstance(X, torch.Tensor):
        for i in range(n_views):
            X[i] = torch.tensor(X[i], dtype=torch.float32, device=device)

    t = time.time()

    model = M.multi_view_contrastive_clustering(n_samples, n_views, temperature, adj_graph, q=args.q).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    l, acc_max, nmi_max, f1_max, ari_max, C_best = train_model(X, model, optimizer, args.alpha, args.beta,
                                                                       args.epochs)
    U.write_best_results(args.data_name, temperature, args.k, args.q, args.alpha, args.beta, 0, acc_max, nmi_max, f1_max,
                                             ari_max, (time.time() - t), False)
