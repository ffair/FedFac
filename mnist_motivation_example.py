

import numpy as np
import pandas as pd
import random
from torch import nn
import torch.nn.functional as F
import torch
import torch.utils.data as Data
from img_utils_linux import Data_Process
import os
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--result_path", type= str, default= '/home/THY/fl_fac')
parser.add_argument("--mnist_cached_file_pfl", type= str, default= '/home/THY/mnist/torchdata')
parser.add_argument("--mnist_rawdata", type= str, default= '/home/THY/mnist')

parser.add_argument("--dataset", type=str, default = 'mnist', choices = ['mnist', 'cifar', 'femnist', 'shakespeare', 'cifar100'])
parser.add_argument("--devide_by_label", action = 'store_true', help="if true, clients are devided by labels of 0-9 (10 clients)")
parser.add_argument("--sampling", type= str, default = 'dirichlet', choices= ['dirichlet', 'sample_mode2', 'mix_dirichlet', 'pathological_split'],
                    help="抽样方案")
parser.add_argument("--dirichlet_parameter", type=float, default = 0.01, help="dirichlet parameter for sampling")
parser.add_argument("--num_classes", type=int, default=10, help="number of classes")
parser.add_argument("--method", type = str, default = 'local')
parser.add_argument("--experi_type", type = str, default = 'non-iid', choices = ['non-iid', 'iid'],
                    help = "if non-iid, each client just have one class; otherwise, each client has the same data distribution")

parser.add_argument("--device", type=str, default='gpu', help="")
parser.add_argument("--batch_size", type=int, default = 16, help="local batch size")
parser.add_argument("--local_ep", type=int, default = 5, help="local epoches")
parser.add_argument("--num_clients", type=int, default = 100, help="number of total clients")
parser.add_argument("--weight_decay", type = float, default = 0, help="weight_decay of adam, 0.0005, 0.00001")
parser.add_argument("--lr", type = float, default= 0.001, help="learning rate")

parser.add_argument("--seed", type = int, default = 319, help="random seed")
args = parser.parse_args()


# path_0 = self.cached_file_pfl + '_clients{}'.format(self.args.num_classes)
# path = path_0 + '/mnist_torchdata'

class Mnist_logi(nn.Module):
    def __init__(self, ):
        super(Mnist_logi, self).__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = torch.nn.Sequential(
            nn.Linear(200, 10),
            torch.nn.Softmax(dim=1)
        )
    def forward(self, x):
        x = torch.flatten(x, start_dim= 1)
        x1 = F.relu(self.fc1(x))
        x2 = self.fc2(x1)
        return x1, x2

model = Mnist_logi()
for n,p in model.named_parameters():
    print(n)
    print(p.size())

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    device = torch.device('cuda' if (torch.cuda.is_available()) & (args.device == 'gpu') else 'cpu')
    setup_seed(args.seed)  # 随机初始化种子

    data_Process = Data_Process(args)
    train_clients, test_clients = data_Process.fetch_data()
    client_list = list(train_clients.keys())

    model = Mnist_logi()

    client_model = {}
    for client_i in client_list:
        client_model[client_i] = model.state_dict()

    result_path = os.path.join(args.result_path, args.method, args.experi_type, args.dataset)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    client_df = pd.DataFrame(columns=['loss', 'acc'])
    client_emb = pd.DataFrame(columns=list(range(200)))

    model.to(device)

    loss_func = nn.CrossEntropyLoss(reduction='sum').to(device)
    for client in client_list:
        model.load_state_dict(client_model[client])
        model.train()
        model.zero_grad()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
                                     weight_decay=args.weight_decay,amsgrad=False)
        train_data = train_clients[client]
        epoch_loss, epoch_acc = [], []
        for epoch in range(args.local_ep):
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
            correct = 0
            batch_loss = []
            emb_i = np.empty(shape=(0, 200))
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                model.zero_grad()
                emb, log_probs = model(images)
                loss = loss_func(log_probs, labels)
                y_pred = log_probs.data.max(1, keepdim=True)[1]
                loss.backward()
                optimizer.step()

                correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()
                batch_loss.append(loss.item())
                emb = emb.cpu().detach().numpy()
                emb_i = np.concatenate([emb_i, emb], axis = 0)

            print(emb_i.shape)
            emb_i = emb_i.mean(axis=0)

            epoch_loss.append(sum(batch_loss) / len(train_loader.dataset))
            accuracy = correct / len(train_loader.dataset)
            epoch_acc.append(accuracy)

        client_emb.loc[client] = emb_i
        a2 = sum(epoch_acc) / len(epoch_acc)
        a1 = sum(epoch_loss) / len(epoch_loss)
        client_df.loc[client] = [a1, a2.item()]

        w = model.state_dict()['fc1.weight'].cpu().detach().numpy()
        pd.DataFrame(w).to_csv(result_path + "/weight_{}.csv".format(client))
        client_df.to_excel(result_path + "/client_metrics.xlsx")
        client_emb.to_excel(result_path + "/client_emb.xlsx")


if __name__ == '__main__':
    main()
