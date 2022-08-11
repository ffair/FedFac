# Factor-Assisted Federated Learning for Personalized Optimization with Heterogeneous Data

This repository contains the official code for our proposed method, FedFac, and the experiments in our paper [Factor-Assisted Federated Learning for Personalized Optimization with Heterogeneous Data].

Federated learning is an emerging distributed machine learning approach, which can simultaneously train a global model from decentralized datasets while preserve data privacy.
However, data heterogeneity is one of the core challenges in federated learning. The heterogeneity issue may severely
degrade the convergence rate and prediction performance of the model trained in federated learning. To address this issue,
we develop a novel personalized federated learning method for heterogeneous data, which is called FedFac. The proposed method is motivated by a common finding that, data in different clients contain both common knowledge and
personalized knowledge. Therefore, the two types of knowledge should be decomposed and taken advantages of separately. We introduce the idea of factor analysis to distinguish
the client-shared information and client-specific information. With this decomposition, a new objective function is established and optimized. Both theoretical and empirical analysis
demonstrate that FedFac has higher computational efficiency against the classical federated learning approaches. The superior prediction performance of FedFac is also verified empirically by comparison with various state-of-the-art federated
learning methods on several real datasets.

## Dependencies

The code requires Python >= 3.8 and PyTorch >= 1.8.0. To install the other dependencies: `pip3 install -r requirements.txt`.

## Data

This code uses the MNIST, CIFAR10, CIFAR100, Federated Extended MNIST (FEMNIST), and Shakespeare datasets.

The CIFAR10, and MNIST datasets are downloaded automatically by the torchvision package. We download CIFAR100, FEMNIST, and Shakespeare 
raw datasets from the [LEAF repository](https://github.com/TalwalkarLab/leaf/), and then generate the versions of these datasets we use in the paper
before running experiments. See the `README.md` files of respective dataset, for instructions on generating data. 

## Usage

FedFac is run using a command of the following form:

`python mnist_fac2_v2.py --method fed_fac --partial_fed --direct_eigDecom --partial_cnn_layer[--partial_cnn_layer] --given_threshold --given_threshold_v[given_threshold_v] --threshold_p --threshold[threshold] --cov_thrhd[--cov_thrhd] --dataset [dataset] --sampling[sampling] --dirichlet_parameter [dirichlet_parameter] --n_shards[n_shards] --num_clients[num_clients] --num_clients_frac[num_clients_frac] --batch_size[batch_size] --local_ep[local_ep] --rounds[rounds] --device[device]`

Explanation of parameters:

- `method` : algorithm to run, may be `fedfac`, `fedavg`, `prox` (FedProx), `fedper` (FedPer), or `lg` (LG-FedAvg)
-`partial_fed`: set True when exacuating algorithm `fedfac`
-`direct_eigDecom`: set True when exacuating algorithm `fedfac`
-`partial_cnn_layer`: the index of layers of the DNN to decompose, setting `1` means the first layer is chosen, setting `1+2` means the first two layers are chosen
-`given_threshold`, `given_threshold_v`: for manually setting the portion of the personalized parameters of the layer to be decomposed 
-`threshold_p`, `threshold`: setting the portion of the personalized parameters of the layer to be decomposed by the percentile of $\nu$
-`cov_thrhd`: for setting the threshold of cumulative variability contribution rate when exacuting factor analysis
- `dataset` : dataset, may be `cifar`(CIFAR10), `cifar100`, `femnist`, `mnist`, `shakespeare`
-`sampling`: the way of setting heterogeneity degree, be either `dirichlet` or `pathological_split`
-`dirichlet_parameter`: setting dirichlet parameter $\pi$ when `sampling` is `dirichlet`
-`n_shards`: setting $S$ when `sampling` is `pathological_split`
-`num_clients`: number of clients
-`num_clients_frac`: fraction of participating clients in each round (for all experiments we use 0.1)
-`batch_size`: batch size used locally by each client
-`local_ep`: total number of local epochs
-`rounds`: total number of communication rounds
-`device`: be either `cpu` or `gpu`

The complete scripts of FedFac as well as the FL baselines we compare against are provided in `papers_experiments/ `

## Generalization to unseen clients

You need to run the same script as in the previous section. Make sure that `--new_test` is added and `--test-clients-frac` is non-zero. For
generalization test of FedFac, if you want to choose `LocalTrain` method, set `fac_newtest_priv` as `train-fix`; if `Ensemble` is preferred, add `--general_ensemble `

## Other experiments

-To simulate our motivation example you need to respectively specify `--experi_type` as `iid` and `non-iid`  when you run `mnist_motivation_example.py`, and get the plot with `fig1.r`

-The plots for `Figure 4` in our paper is built through `fig4_plot.py`, where the results of experiments can be abtained by running the according scripts in `papers_experiments/ `

-Through running `feature extract.py`, you could visualize the features extracted by Resnet-18 on any image provided in `images/` 

-The `Figure B.1` in our paper is built using `fedfac_clients.rmd`, `Figure B.2` and `Figure B.3` could be abtained by `fedfac_plot.rmd`

# Acknowledgements

The code for data generation and `FedEM` algorithm in this repository was adapted from code in repository of [Federated Multi-Task Learning under a Mixture of Distributions](https://github.com/omarfoq/FedEM)




