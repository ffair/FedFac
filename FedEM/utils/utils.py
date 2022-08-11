import time
from torch import nn
from models import *
from datasets import *
from learners.learner import *
from learners.learners_ensemble import *
from client import *
from aggregator import *
import os
from .optim import *
from .metrics import *
from .constants import *
# from .decentralized import *

from torch.utils.data import DataLoader

from tqdm import tqdm


def get_data_dir(experiment_name):
    """
    returns a string representing the path where to find the datafile corresponding to the experiment

    :param experiment_name: name of the experiment
    :return: str

    """
    data_dir = os.path.join("data", experiment_name, "all_data")

    return data_dir


def get_learner(
        model_state,
        model_dim,
        name,
        device,
        seed,
        input_dim=None,
        output_dim=None
):
    """
    constructs the learner corresponding to an experiment for a given seed

    :param name: name of the experiment to be used; possible are
                 {`synthetic`, `cifar10`, `emnist`, `shakespeare`}
    :param device: used device; possible `cpu` and `cuda`
    :param optimizer_name: passed as argument to utils.optim.get_optimizer
    :param scheduler_name: passed as argument to utils.optim.get_lr_scheduler
    :param initial_lr: initial value of the learning rate
    :param mu: proximal term weight, only used when `optimizer_name=="prox_sgd"`
    :param input_dim: input dimension, only used for synthetic dataset
    :param output_dim: output_dimension; only used for synthetic dataset
    :param n_rounds: number of training rounds, only used if `scheduler_name == multi_step`, default is None;
    :param seed:
    :return: Learner

    """
    torch.manual_seed(seed)
    if name == "cifar10":
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        is_binary_classification = False
    elif name == "cifar100":
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        is_binary_classification = False
    elif name == "femnist":
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        is_binary_classification = False
    elif name == "shakespeare":
        all_characters = string.printable
        labels_weight = torch.ones(len(all_characters), device=device)
        for character in CHARACTERS_WEIGHTS:
            labels_weight[all_characters.index(character)] = CHARACTERS_WEIGHTS[character]
        labels_weight = labels_weight * 8
        criterion = nn.CrossEntropyLoss(reduction="none", weight=labels_weight).to(device)
        metric = accuracy
        is_binary_classification = False
    else:
        raise NotImplementedError


    if name == "shakespeare":
        return LanguageModelingLearner(
            model_state=model_state,
            model_dim=model_dim,
            criterion=criterion,
            metric=metric,
            device=device,
            is_binary_classification=is_binary_classification
        )
    else:
        return Learner(
            model_state = model_state,
            model_dim=model_dim,
            criterion=criterion,
            metric=metric,
            device=device,
            is_binary_classification=is_binary_classification
        )


def get_learners_ensemble(
        n_learners,
        clients_model,
        model_dim,
        name,
        device,
        seed,
        input_dim=None,
        output_dim=None
):
    """
    constructs the learner corresponding to an experiment for a given seed

    :param n_learners: number of learners in the ensemble
    :param name: name of the experiment to be used; possible are
                 {`synthetic`, `cifar10`, `emnist`, `shakespeare`}
    :param device: used device; possible `cpu` and `cuda`
    :param optimizer_name: passed as argument to utils.optim.get_optimizer
    :param scheduler_name: passed as argument to utils.optim.get_lr_scheduler
    :param initial_lr: initial value of the learning rate
    :param mu: proximal term weight, only used when `optimizer_name=="prox_sgd"`
    :param input_dim: input dimension, only used for synthetic dataset
    :param output_dim: output_dimension; only used for synthetic dataset
    :param n_rounds: number of training rounds, only used if `scheduler_name == multi_step`, default is None;
    :param seed:
    :return: LearnersEnsemble

    """

    learners = []
    for learner_id in range(n_learners):
        learners.append(
            get_learner(
                model_state = clients_model[learner_id],
                model_dim = model_dim,
                name=name,
                device=device,
                input_dim=input_dim,
                output_dim=output_dim,
                seed=seed + learner_id,
            )
        )

    learners_weights = torch.ones(n_learners) / n_learners
    if name == "shakespeare":
        return LanguageModelingLearnersEnsemble(learners=learners, learners_weights=learners_weights)
    else:
        return LearnersEnsemble(learners=learners, learners_weights=learners_weights)


def get_client(
        client_name,
        client_type,
        learners_ensemble,
        q,
        train_iterator,
        val_iterator,
        test_iterator,
        logger,
        local_steps,
        tune_locally,
):
    """

    :param client_type:
    :param learners_ensemble:
    :param q: fairness hyper-parameter, ony used for FFL client
    :param train_iterator:
    :param val_iterator:
    :param test_iterator:
    :param logger:
    :param local_steps:
    :param tune_locally

    :return:

    """
    if client_type == "mixture":
        return MixtureClient(
            client_name = client_name,
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally
        )
    else:
        return Client(
            client_name=client_name,
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally
        )


def get_aggregator(
        args,
        aggregator_type,
        clients,
        global_learners_ensemble,
        lr,
        lr_lambda,
        mu,
        communication_probability,
        q,
        sampling_rate,
        log_freq,
        global_train_logger,
        global_test_logger,
        test_clients,
        verbose,
        seed=None,
        file_path = None,
        loss_train_df = None,
        loss_test_df = None,
        acc_train_df = None,
        acc_test_df = None,
        fed_metrics_df = None,
        loss_train_new_df=None,
        loss_test_new_df=None,
        acc_train_new_df=None,
        acc_test_new_df=None,
        fed_metrics_new_df = None
):
    """
    `personalized` corresponds to pFedMe

    :param aggregator_type:
    :param clients:
    :param global_learners_ensemble:
    :param lr: oly used with FLL aggregator
    :param lr_lambda: only used with Agnostic aggregator
    :param mu: penalization term, only used with L2SGD
    :param communication_probability: communication probability, only used with L2SGD
    :param q: fairness hyper-parameter, ony used for FFL client
    :param sampling_rate:
    :param log_freq:
    :param global_train_logger:
    :param global_test_logger:
    :param test_clients
    :param verbose: level of verbosity
    :param seed: default is None
    :return:

    """
    seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    if aggregator_type == "no_communication":
        return NoCommunicationAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed,
            file_path=file_path,
            loss_train_df=loss_train_df,
            loss_test_df=loss_test_df,
            acc_train_df=acc_train_df,
            acc_test_df=acc_test_df,
            fed_metrics_df=fed_metrics_df
        )
    elif aggregator_type == "centralized":
        return CentralizedAggregator(
            args=args,
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed,
            file_path=file_path,
            loss_train_df=loss_train_df,
            loss_test_df=loss_test_df,
            acc_train_df=acc_train_df,
            acc_test_df=acc_test_df,
            fed_metrics_df=fed_metrics_df,
            loss_train_new_df=loss_train_new_df,
            loss_test_new_df=loss_test_new_df,
            acc_train_new_df=acc_train_new_df,
            acc_test_new_df=acc_test_new_df,
            fed_metrics_new_df = fed_metrics_new_df
        )
    else:
        raise NotImplementedError(
            "{aggregator_type} is not a possible aggregator type."
            " Available are: `no_communication`, `centralized`,"
            " `personalized`, `clustered`, `fednova`, `AFL`,"
            " `FFL` and `decentralized`."
        )
