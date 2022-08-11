import torch


class Learner:
    """
    Responsible of training and evaluating a (deep-)learning model

    Attributes
    ----------
    model (nn.Module): the model trained by the learner

    criterion (torch.nn.modules.loss): loss function used to train the `model`, should have reduction="none"

    metric (fn): function to compute the metric, should accept as input two vectors and return a scalar

    device (str or torch.device):

    optimizer (torch.optim.Optimizer):

    lr_scheduler (torch.optim.lr_scheduler):

    is_binary_classification (bool): whether to cast labels to float or not, if `BCELoss`
    is used as criterion this should be set to True

    Methods
    ------
    compute_gradients_and_loss:

    optimizer_step: perform one optimizer step, requires the gradients to be already computed.

    fit_batch: perform an optimizer step over one batch

    fit_epoch:

    fit_batches: perform successive optimizer steps over successive batches

    fit_epochs:

    evaluate_iterator: evaluate `model` on an iterator

    gather_losses:

    get_param_tensor: get `model` parameters as a unique flattened tensor

    free_memory: free the memory allocated by the model weights

    free_gradients:
    """

    def __init__(
            self,
            model_state, # model.state_dict()
            model_dim,
            criterion,
            metric,
            device,
            is_binary_classification=False
    ):

        self.model_state = model_state
        self.criterion = criterion.to(device)
        self.metric = metric
        self.device = device
        # self.optimizer = optimizer
        # self.lr_scheduler = lr_scheduler
        self.is_binary_classification = is_binary_classification
        self.model_dim = model_dim
        # self.model_dim = int(self.get_param_tensor().shape[0])

    def load_model_without_optim(self, model):
        model.load_state_dict(self.model_state)

    def load_client_model(self, model, optimizer):
        model.load_state_dict(self.model_state)
        model.zero_grad()
        if callable(getattr(optimizer, "set_initial_params", None)):
            optimizer.set_initial_params(
                model.parameters()
            )

    def optimizer_step(self, optimizer, lr_scheduler = None):
        """
         perform one optimizer step, requires the gradients to be already computed.
        """
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

    def compute_gradients_and_loss(self, model, optimizer, batch, weights=None):
        """
        compute the gradients and loss over one batch.

        :param batch: tuple of (x, y, indices)
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            loss

        """
        model.train()

        x, y, indices = batch
        x = x.to(self.device).type(torch.float32)
        y = y.to(self.device)

        if self.is_binary_classification:
            y = y.type(torch.float32).unsqueeze(1)

        optimizer.zero_grad()

        y_pred = model(x)
        loss_vec = self.criterion(y_pred, y)

        if weights is not None:
            weights = weights.to(self.device)
            loss = (loss_vec.T @ weights[indices]) / loss_vec.size(0)
        else:
            loss = loss_vec.mean()

        loss.backward()

        return loss.detach()

    def fit_batch(self, model, optimizer, batch, weights=None, lr_scheduler = None):
        """
        perform an optimizer step over one batch drawn from `iterator`

        :param batch: tuple of (x, y, indices)
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            loss.detach()
            metric.detach()

        """
        model.train()

        x, y, indices = batch
        x = x.to(self.device).type(torch.float32)
        y = y.to(self.device)

        if self.is_binary_classification:
            y = y.type(torch.float32).unsqueeze(1)

        optimizer.zero_grad()

        y_pred = model(x)
        loss_vec = self.criterion(y_pred, y)
        metric = self.metric(y_pred, y) / len(y)

        if weights is not None:
            weights = weights.to(self.device)
            loss = (loss_vec.T @ weights[indices]) / loss_vec.size(0)
        else:
            loss = loss_vec.mean()

        loss.backward()

        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        return loss.detach(), metric.detach()

    def fit_epoch(self, model, optimizer, iterator, weights=None):
        """
        perform several optimizer steps on all batches drawn from `iterator`

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            loss.detach()
            metric.detach()

        """
        model.train()

        global_loss = 0.
        global_metric = 0.
        n_samples = 0
        for indices, (x, y) in enumerate(iterator):
            x = x.to(self.device).type(torch.float32)
            y = y.to(self.device)

            if self.is_binary_classification:
                y = y.type(torch.float32).unsqueeze(1)

            optimizer.zero_grad()

            y_pred = model(x)

            loss_vec = self.criterion(y_pred, y)

            if weights is not None:
                weights = weights.to(self.device)
                loss = (loss_vec.T @ weights[n_samples:n_samples+y.size(0)]) / loss_vec.size(0)
            else:
                loss = loss_vec.mean()
            loss.backward()

            optimizer.step()

            n_samples += y.size(0)

            global_loss += loss.detach() * loss_vec.size(0)
            global_metric += self.metric(y_pred, y).detach()

        return global_loss / n_samples, global_metric / n_samples

    def fit_epochs(self, model, optimizer, iterator, n_epochs, weights=None, lr_scheduler = None):
        """
        perform multiple training epochs

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :param n_epochs: number of successive batches
        :type n_epochs: int
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            None

        """
        for step in range(n_epochs):
            self.fit_epoch(model, optimizer, iterator, weights)

            if lr_scheduler is not None:
                lr_scheduler.step()
        net_weight = model.state_dict()
        net_weight = {k: v.cpu() for k, v in net_weight.items()}
        self.model_state = net_weight

    def gather_losses(self, model, iterator):
        """
        gathers losses for all elements of iterator

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :return
            tensor with losses of all elements of the iterator.dataset

        """
        # self.load_client_model(model, client_model_statedict)
        model.eval()
        n_samples = len(iterator.dataset)
        # print('n_samples{}'.format(n_samples))
        all_losses = torch.zeros(n_samples, device=self.device)

        with torch.no_grad():
            for indices, (x, y) in enumerate(iterator):
                x = x.to(self.device).type(torch.float32)
                y = y.to(self.device)

                if self.is_binary_classification:
                    y = y.type(torch.float32).unsqueeze(1)

                y_pred = model(x)
                all_losses = self.criterion(y_pred, y)

        return all_losses

    def evaluate_iterator(self, model, iterator):
        """
        evaluate learner on `iterator`

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :return
            global_loss and  global_metric accumulated over the iterator

        """
        model.eval()

        global_loss = 0.
        global_metric = 0.
        n_samples = 0
        for indices, (x, y) in enumerate(iterator):
            x = x.to(self.device).type(torch.float32)
            y = y.to(self.device)

            if self.is_binary_classification:
                y = y.type(torch.float32).unsqueeze(1)

            with torch.no_grad():
                y_pred = model(x)

                global_loss += self.criterion(y_pred, y).sum().detach()
                global_metric += self.metric(y_pred, y).detach()

            n_samples += y.size(0)

        return global_loss / n_samples, global_metric / n_samples

    def get_param_tensor(self, model):
        """
        get `model` parameters as a unique flattened tensor

        :return: torch.tensor

        """

        param_list = []
        for param in model.parameters():
            param_list.append(param.data.view(-1, ))

        return torch.cat(param_list)

    def get_grad_tensor(self, model):
        """
        get `model` gradients as a unique flattened tensor

        :return: torch.tensor

        """
        grad_list = []

        for param in model.parameters():
            if param.grad is not None:
                grad_list.append(param.grad.data.view(-1, ))

        return torch.cat(grad_list)



class LanguageModelingLearner(Learner):
    def fit_epoch(self, model, optimizer, iterator, weights=None):

        model.train()

        global_loss = 0.
        global_metric = 0.
        n_samples = 0
        for indices, (x, y) in enumerate(iterator):
            x = x.to(self.device)
            y = y.to(self.device)

            chunk_len = y.size(1)

            optimizer.zero_grad()

            y_pred = model(x)
            loss_vec = self.criterion(y_pred, y)

            if weights is not None:
                weights = weights.to(self.device)
                loss = (loss_vec.T @ weights[n_samples:n_samples+y.size(0)]).mean() / loss_vec.size(0)
            else:
                loss = loss_vec.mean()

            n_samples += y.size(0)
            loss.backward()
            optimizer.step()

            global_loss += loss.detach() * loss_vec.size(0) / chunk_len
            global_metric += self.metric(y_pred, y).detach() / chunk_len

        return global_loss / n_samples, global_metric / n_samples

    def fit_batch(self, model, optimizer, batch, weights=None, lr_scheduler = None):

        model.train()

        x, y, indices = batch
        x = x.to(self.device)
        y = y.to(self.device)

        n_samples = y.size(0)
        chunk_len = y.size(1)

        optimizer.zero_grad()

        y_pred = model(x)
        loss_vec = self.criterion(y_pred, y)

        if weights is not None:
            weights = weights.to(self.device)
            loss = (loss_vec.T @ weights[indices]).mean() / loss_vec.size(0)
        else:
            loss = loss_vec.mean()

        loss.backward()

        optimizer.step()

        global_loss = loss.detach() * loss_vec.size(0) / chunk_len
        global_metric = self.metric(y_pred, y).detach() / chunk_len

        return global_loss / n_samples, global_metric / n_samples

    def compute_gradients_and_loss(self, model, optimizer, batch, weights=None):
        """
        compute the gradients and loss over one batch.

        :param batch: tuple of (x, y, indices)
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            loss

        """
        raise NotImplementedError

    def gather_losses(self, model, iterator):
        """
        gathers losses for all elements of iterator

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :return
            tensor with losses of all elements of the iterator.dataset

        """
        model.eval()
        n_samples = len(iterator.dataset)
        predictions = torch.zeros(n_samples, device=self.device)

        for n, p in model.named_parameters():
            print(p.device)

        with torch.no_grad():
            for indices, (x, y) in enumerate(iterator):
                x = x.to(self.device)
                y = y.to(self.device)

                print(x)
                print(y)
                y_pred = model(x)
                print("y{}".format(y.size()))
                print("y_pred{}".format(y_pred.size()))
                predictions = self.criterion(y_pred, y).mean(axis=1)

        return predictions

    def evaluate_iterator(self, model, iterator):
        """
        evaluate learner on `iterator`

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :return
            global_loss and  global_metric accumulated over the iterator

        """
        model.eval()

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        with torch.no_grad():
            for indices, (x, y) in enumerate(iterator):
                x = x.to(self.device)
                y = y.to(self.device)
                n_samples += y.size(0)

                chunk_len = y.size(1)

                y_pred = model(x)
                global_loss += self.criterion(y_pred, y).sum().detach() / chunk_len
                global_metric += self.metric(y_pred, y).detach() / chunk_len

        return global_loss / n_samples, global_metric / n_samples
