

import matplotlib
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import random
import copy
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision.models as models
import os
import cv2
from PIL import Image

def get_resnet18(n_classes):
    """
    creates Resnet model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, n_classes)

    return model

dataset = 'cifar'
mnist_rawdata = '/home/THY/mnist'
cifar_rawdata = '/home/THY/cifar/CIFAR/cifar-10-python'
crop224 = False
if dataset == 'mnist':
    raw_train = datasets.MNIST(root=mnist_rawdata, train=True, transform=transforms.ToTensor(), download=False)
    raw_test = datasets.MNIST(root=mnist_rawdata, train=False, transform=transforms.ToTensor(), download=False)
if dataset == 'cifar':
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=mean),
    ])
    raw_train = datasets.CIFAR10(root=cifar_rawdata, train=True, transform=transform_train, download=False)
    raw_test = datasets.CIFAR10(root=cifar_rawdata, train=False, transform=transform_test, download=False)

if dataset == 'cifar':
    train_x, train_y = [],[]
    indx = [8, 69, 12, 37, 43, 52, 68, 15, 13, 1, 31, 53, 18, 54, 19, 22, 25, 20, 28, 34, 26, 36, 39, 30, 40]
    for i in indx:
        a, b = raw_train[i]
        train_x.append(a.numpy())
        train_y.append(b)
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    train_data = Data.TensorDataset(torch.from_numpy(train_x), torch.tensor(train_y, dtype=torch.long))

# read image
if dataset == 'image':
    train_x_1, train_y_1 = [],[]
    tf = transforms.Compose([transforms.Resize((224, 224)),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])
    image_list = {'bird':7, 'cat':7, 'deer':8, 'dog':5, 'horse':7, 'truck':5}
    for m,n in image_list.items():
        for i in range(n):
            image_path = '/home/THY/images/' + m + str(i+1) +'.jpg'
            image = Image.open(image_path)
            img = tf(image)
            train_x_1.append(img.numpy())
            train_y_1.append(raw_train.class_to_idx[m])
    train_x_1 = np.array(train_x_1)
    train_y_1 = np.array(train_y_1)
    train_data = Data.TensorDataset(torch.from_numpy(train_x_1), torch.tensor(train_y_1, dtype=torch.long))


import matplotlib.pyplot as plt
img_plt = plt.imread('/home/THY/images/bird1.jpg')
plt.imshow(img_plt, cmap=plt.cm.binary)
plt.show()

initial_imgae = []
for m,n in image_list.items():
    for i in range(n):
        image_path = '/home/THY/images/' + m + str(i+1) +'.jpg'
        img = cv2.imread(image_path)
        initial_imgae.append(img)


#===================================================================
device = 'cpu'
dataset = 'cifar'
cached_file_pfl = '/home/THY/cifar/torchdata'
sampling = 'dirichlet'
dirichlet_parameter = 0.5
partial_cnn_layer = '5'
cov_thrhd = 0.85
rounds = 268
num_classes = 10
partial_fed = False

threshold = 'mean'
result_path = '/home/THY/fl_fac'
method = 'fed' # 'fed' ，fed_fac


if dataset == 'cifar':
    file_path = os.path.join(result_path, method, dataset, sampling)
    sample_p = dirichlet_parameter

if partial_fed:
    result_path = os.path.join(file_path,'threshold{}_p{}_l{}_covthrhd{}'.format(threshold, sample_p, partial_cnn_layer, cov_thrhd))
if method == 'fed':
    result_path = os.path.join(file_path, 'p{}_r{}'.format(sample_p, rounds))

# load model
if os.path.exists(result_path):
    shared_layers_param = torch.load(os.path.join(result_path, "s_param" + ".pth"))
    partial_conv_shared = torch.load(os.path.join(result_path, "p_param" + ".pth"))['partial_conv_shared']
    client_model_param = torch.load(os.path.join(result_path, 'client_model' + '.pth'))

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if (torch.cuda.is_available()) & (device == 'gpu') else 'cpu')
setup_seed(1012)

# get data
path = cached_file_pfl + '/dirichlet_parameter_{}'.format(dirichlet_parameter) + '/torchdata'
if os.path.exists(path):
    data_load = torch.load(path)
    train_clients = data_load['traindataset']
    test_clients = data_load['testdataset']

client_list = [str(i) for i in range(100)]
if dataset == 'cifar':
    model = get_resnet18(num_classes)
client_model = client_model_param

layer_name, layer_size = [], []
for n, p in model.named_parameters():
    layer_name.append(n)
    layer_size.append(p.size())

l_n, fac_type = {}, {}
if dataset == 'cifar':
    for i in range(len(layer_name) // 3):
        l_n[str(i + 1)] = layer_name[3 * i:3 * (i + 1)]
    l_n[str(21)] = layer_name[-2:]
    for k, v in l_n.items():
        if ('conv' in v[0]) or ('downsample' in v[0]):
            fac_type[v[0]] = 'conv'
        else:
            fac_type[v[0]] = 'dense'

pcl = partial_cnn_layer.split('+')
partial_shared_layer_key = []
nsk = []
for i in pcl:
    partial_shared_layer_key.append(l_n[i][0])
    nsk = nsk + l_n[i]

if partial_fed:
    shared_layer_keys = [item for item in layer_name if item not in nsk]
if method == 'fed':
    shared_layer_keys = layer_name

print('-'*20 + 'layer_name' + '-'*20)
print(layer_name)
print('-'*20 + 'partial_shared_layer_key' + '-'*20)
print(partial_shared_layer_key)
print('-'*20 + 'shared_layer_keys' + '-'*20)
print(shared_layer_keys)


def partial_param_update(model, local_model, partial_conv_shared, partial_shared_layer_key):
    model_dict = copy.deepcopy(local_model)
    p = model_dict[partial_shared_layer_key]
    if device == 'gpu':
        p = p.cpu().detach().numpy()
    else:
        p = p.numpy()
    for indx in partial_conv_shared.keys():
        p[indx,] = partial_conv_shared[indx]
    p = torch.from_numpy(p).to(device)
    model_dict[partial_shared_layer_key] = p
    model.load_state_dict(model_dict)

def load_welltrained_model(client):
    """
    Args:
        client_model_i: model.state_dict of client_i

    Returns:

    """
    model.load_state_dict(client_model[client])
    if partial_fed:
        for key in partial_shared_layer_key:
            partial_param_update(model, client_model[client], partial_conv_shared[key], key)
        model_dict = model.state_dict()
        model_dict.update(copy.deepcopy(shared_layers_param))
        model.load_state_dict(model_dict)
    else:
        model_dict = client_model[client]
        model_dict.update(copy.deepcopy(shared_layers_param))
        model.load_state_dict(model_dict)



data_loader = torch.utils.data.DataLoader(train_data, batch_size = len(train_data), shuffle=False)
loss_test = pd.DataFrame(columns=client_list)
acc_test = pd.DataFrame(columns=client_list)
pred = pd.DataFrame(columns=client_list)
def save_metrics():
    loss_test.to_excel(result_path + '/losstest_single.xlsx')
    acc_test.to_excel(result_path + '/acctest_single.xlsx')

def new_pred(client):
    load_welltrained_model(client)
    model.eval()
    test_loss = 0
    correct = 0
    labels = []
    with torch.no_grad():
        for idx, (data, target) in enumerate(data_loader):
            labels += list(target.numpy())
            data, target = data.to(device), target.to(device)
            log_probs = model(data)
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
            pred_i = y_pred.eq(target.data.view_as(y_pred)).long().cpu().numpy()
            test_loss /= len(data_loader.dataset)
            accuracy = correct / len(data_loader.dataset)
    loss_test.loc[0, client] = test_loss
    acc_test.loc[0, client] = accuracy.item()
    pred.loc[0, client] = pred_i.reshape((1, -1))
    return test_loss, accuracy.item(), pred_i.reshape((1, -1))

for client in client_list:
    loss_i, acc_i, pred_i = new_pred(client)

acc_test = acc_test.sort_values(by = 0, axis = 1)

# loss_14, acc_14, pred_14 = new_pred('90') #'14'
# loss_40, acc_40, pred_40 = new_pred('85') # '40'
#
# loss_14, acc_14, pred_14 = new_pred('78') # 4: deer
# loss_40, acc_40, pred_40 = new_pred('87') # 1, 2, 9
loss_8, acc_8, pred_8 = new_pred('8') # 4: deer
loss_72, acc_72, pred_72 = new_pred('72') # 1, 2, 9

# 所选图片
photo = pred_8 * pred_72
photo = photo.tolist()
#photo = np.where(photo == 1)
#photo = [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,0, 1, 1]


share_indx = [1, 3, 7, 8, 9, 11, 13, 16, 17, 18, 20, 22, 23, 26, 29, 31, 38, 39, 40, 41, 43, 44, 46, 48, 49, 50, 53, 57, 62, 63]
private_indx = [i for i in range(64) if i not in share_indx]


class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = {}
        for name, module in self.submodule._modules.items():
            if "fc" in name:
                x = x.view(x.size(0), -1)

            x = module(x)
            print(name)
            if self.extracted_layers is None or name in self.extracted_layers and 'fc' not in name:
                outputs[name] = x

        return outputs

def make_dirs(path):
    if os.path.exists(path) is False:
        os.makedirs(path)


def get_feature(img_indx, client, dataset, exact_list=None, pred_list = None):
    if exact_list is None:
        exact_list = ['layer4']
    if dataset == 'cifar':
        img = raw_train[indx[img_indx]][0]
        dst_s = '/home/THY/fl_fac/feature_plot/' + dataset + '/' + method + '/' + 'client{}'.format(
            client) + '/' + 'fig{}'.format(str(indx[img_indx])) + '/' + 'share'
        dst_p = '/home/THY/fl_fac/feature_plot/' + dataset + '/' + method + '/' + 'client{}'.format(
            client) + '/' + 'fig{}'.format(str(indx[img_indx])) + '/' + 'private'
    if pred_list[img_indx] == 1:
        pred = 'correct'
    else:
        pred = 'incorrect'

    if dataset == 'image':
        img = torch.from_numpy(train_x_1)[img_indx]
        dst_s = '/home/THY/fl_fac/feature_plot/' + dataset + '/' + method + '/' + 'client{}'.format(
            client) + '/' + pred + '/' + 'fig{}'.format(str(img_indx)) + '/' + 'share'
        dst_p = '/home/THY/fl_fac/feature_plot/' + dataset + '/' + method + '/' + 'client{}'.format(
            client) + '/' + pred + '/' + 'fig{}'.format(str(img_indx)) + '/' + 'private'

    img = img.unsqueeze(0)
    therd_size = 256
    myexactor = FeatureExtractor(model, exact_list)
    outs = myexactor(img)
    for k, v in outs.items():
        print(k)
        features = v[0]
        iter_range = features.shape[0]
        for i in range(iter_range):
            if 'fc' in k:
                continue

            feature = features.data.numpy()
            feature_img = feature[i, :, :]
            feature_img = np.asarray(feature_img * 255, dtype=np.uint8)

            dst_path_s = os.path.join(dst_s, k)
            dst_path_p = os.path.join(dst_p, k)
            make_dirs(dst_path_s)
            make_dirs(dst_path_p)

            feature_img = cv2.applyColorMap(feature_img, cv2.COLORMAP_PINK)
            if feature_img.shape[0] < therd_size:
                if i in share_indx:
                    tmp_file = os.path.join(dst_path_s, str(i) + '_' + str(therd_size) + '.png')
                else:
                    tmp_file = os.path.join(dst_path_p, str(i) + '_' + str(therd_size) + '.png')

                tmp_img = feature_img.copy()
                tmp_img = cv2.resize(tmp_img, (therd_size, therd_size), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(tmp_file, tmp_img)
            if i in share_indx:
                dst_file = os.path.join(dst_path_s, str(i) + '.png')
            else:
                dst_file = os.path.join(dst_path_p, str(i) + '.png')

            cv2.imwrite(dst_file, feature_img)

    if dataset == 'cifar':
        path = '/home/THY/fl_fac/feature_plot/' + dataset + '/' + method + '/' + 'client{}'.format(client) + '/' + \
               'fig{}'.format(str(indx[img_indx])) +'/'+ '{}.jpg'.format(str(indx[img_indx]))
        fig = raw_train.data[indx[img_indx]]
        if fig.shape[0] < therd_size:
            tmp_img = cv2.resize(fig, (therd_size, therd_size), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(path, tmp_img)
    if dataset == 'image':
        path = '/home/THY/fl_fac/feature_plot/' + dataset + '/' + method + '/' + 'client{}'.format(client) + '/' + pred + '/' + \
               'fig{}'.format(str(img_indx)) +'/'+ '{}.jpg'.format(str(img_indx))
        initial_img = cv2.resize(initial_imgae[img_indx], (therd_size, therd_size))
        cv2.imwrite(path, initial_img)

#--------------------------------------------------

client = '87' # '85'
load_welltrained_model(client)
photo_8 = pred_8.tolist()
photo_72 = pred_72.tolist()
for m in range(len(photo_8[0])):
    if photo_8[0][m] != 0:
        get_feature(m, client, 'image', exact_list = ['layer1']) # exact_list = ['layer4'], exact_list = ['conv1']

get_feature(19, '8', 'image', exact_list = ['conv1'])

p_union = pred_8 + pred_72
p_union = p_union.tolist()

a=[7,8,9,10,11,12,13,14,16,17,18,19,20,21,36, 22, 23, 24, 25, 34, 35, 38]
for item in a:
    p_union[0][item] = 1
client = '8' # '85'
load_welltrained_model(client)
for m in range(len(p_union[0])):
    if p_union[0][m] != 0:
        get_feature(m, client, 'image', exact_list = ['layer1'], pred_list = pred_8.tolist()[0]) # exact_list = ['layer4'], exact_list = ['conv1']

