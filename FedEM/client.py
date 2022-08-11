import torch.nn.functional as F

from copy import deepcopy
from utils.torch_utils import *


class Client(object):
    r"""Implements one clients

    Attributes
    ----------
    learners_ensemble
    n_learners

    train_iterator

    val_iterator

    test_iterator

    train_loader

    n_train_samples

    n_test_samples

    samples_weights

    local_steps

    logger

    tune_locally:

    Methods
    ----------
    __init__
    step
    write_logs
    update_sample_weights
    update_learners_weights

    """
    def __init__(
            self,
            client_name,
            learners_ensemble,
            train_iterator,
            val_iterator,
            test_iterator,
            logger,
            local_steps,
            tune_locally=False
    ):

        self.client_name = client_name,
        self.learners_ensemble = learners_ensemble
        self.n_learners = len(self.learners_ensemble)
        self.tune_locally = tune_locally

        if self.tune_locally:
            self.tuned_learners_ensemble = deepcopy(self.learners_ensemble)
        else:
            self.tuned_learners_ensemble = None

        self.binary_classification_flag = self.learners_ensemble.is_binary_classification

        self.train_iterator = train_iterator
        self.val_iterator = val_iterator
        self.test_iterator = test_iterator

        self.train_loader = iter(self.train_iterator)

        self.n_train_samples = len(self.train_iterator.dataset)
        self.n_test_samples = len(self.test_iterator.dataset)

        self.samples_weights = torch.ones(self.n_learners, self.n_train_samples) / self.n_learners

        self.local_steps = local_steps

        self.counter = 0
        self.logger = logger

    def get_next_batch(self):
        try:
            batch = next(self.train_loader)
        except StopIteration:
            self.train_loader = iter(self.train_iterator)
            batch = next(self.train_loader)

        return batch

    def step(self, learners_model, learners_optim, learners_optim_lrsched, single_batch_flag=False, *args, **kwargs):
        """
        perform one step for the client
        : client_models: [learner_model.state_dict,...]
        :param single_batch_flag: if true, the client only uses one batch to perform the update
        :return
            clients_updates: ()
        """
        self.counter += 1
        self.learners_ensemble.load_client_models(learners_model, learners_optim)
        self.update_sample_weights(learners_model)
        self.update_learners_weights()

        if single_batch_flag:
            batch = self.get_next_batch()
            client_updates = \
                self.learners_ensemble.fit_batch(
                    learners_model = learners_model,
                    learners_optim=learners_optim,
                    batch=batch,
                    weights=self.samples_weights,
                    learners_optim_lrsched = learners_optim_lrsched
                )
        else:
            client_updates = \
                self.learners_ensemble.fit_epochs(
                    learners_model = learners_model,
                    learners_optim = learners_optim,
                    iterator=self.train_iterator,
                    n_epochs=self.local_steps,
                    weights=self.samples_weights,
                    learners_optim_lrsched = learners_optim_lrsched
                )

        # TODO: add flag arguments to use `free_gradients`
        # self.learners_ensemble.free_gradients()

        return client_updates

    def write_logs(self, learners_model):
        self.learners_ensemble.load_models_without_optim(learners_model)

        if self.tune_locally:
            self.update_tuned_learners()

        if self.tune_locally:
            train_loss, train_acc = self.tuned_learners_ensemble.evaluate_iterator(self.val_iterator)
            test_loss, test_acc = self.tuned_learners_ensemble.evaluate_iterator(self.test_iterator)
        else:
            train_loss, train_acc = self.learners_ensemble.evaluate_iterator(learners_model, self.val_iterator)
            test_loss, test_acc = self.learners_ensemble.evaluate_iterator(learners_model, self.test_iterator)

        # self.logger.add_scalar("Train/Loss", train_loss, self.counter)
        # self.logger.add_scalar("Train/Metric", train_acc, self.counter)
        # self.logger.add_scalar("Test/Loss", test_loss, self.counter)
        # self.logger.add_scalar("Test/Metric", test_acc, self.counter)

        return train_loss, train_acc, test_loss, test_acc

    def update_sample_weights(self, learners_model):
        pass

    def update_learners_weights(self):
        pass

    def update_tuned_learners(self):
        if not self.tune_locally:
            return

        for learner_id, learner in enumerate(self.tuned_learners_ensemble):
            copy_model(source=self.learners_ensemble[learner_id].model, target=learner.model)
            learner.fit_epochs(self.train_iterator, self.local_steps, weights=self.samples_weights[learner_id])


class MixtureClient(Client):
    def update_sample_weights(self, learners_model):
        all_losses = self.learners_ensemble.gather_losses(learners_model, self.val_iterator)
        self.samples_weights = F.softmax((torch.log(self.learners_ensemble.learners_weights) - all_losses.T), dim=1).T

    def update_learners_weights(self):
        self.learners_ensemble.learners_weights = self.samples_weights.mean(dim=1)


