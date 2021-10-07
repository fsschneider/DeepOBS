# -*- coding: utf-8 -*-
"""Base class for DeepOBS test problems."""
import abc

import torch

from .. import config


class TestProblem(abc.ABC):
    """Base class for DeepOBS test problems.

    Args:
        batch_size (int): Batch size to use.
        l2_reg (float): L2-Regularization (weight decay) factor to use. If
            not specified, the test problems revert to their respective defaults.
            Note: Some test problems do not use regularization and this value will
            be ignored in such a case.

    Attributes:
        _batch_size: Batch_size for the data of this test problem.
        _l2_reg: The regularization factor for this test problem
        data: The dataset used by the test problem (datasets.DataSet instance).
        loss_function: The loss function for this test problem.
        net: The torch module (the neural network) that is trained.

    Methods:
        train_init_op: Initializes the test problem for the
            training phase.
        train_eval_init_op: Initializes the test problem for
            evaluating on training data.
        test_init_op: Initializes the test problem for
            evaluating on test data.
        _get_next_batch: Returns the next batch of data of the current phase.
        get_batch_loss_and_accuracy: Calculates the loss and accuracy of net on
            the next batch of the current phase.
        set_up: Sets all public attributes.
    """

    def __init__(self, batch_size, l2_reg=None):
        """Create a new test problem instance.

        Args:
            batch_size (int): Batch size to use.
            l2_reg (float): L2-Regularization (weight decay) factor to use. If
                not specified, the test problems revert to their respective defaults.
                Note: Some test problems do not use regularization and this value will
                be ignored in such a case.
        """
        self._batch_size = batch_size
        self._l2_reg = l2_reg
        self._device = torch.device(config.get_default_device())

        self._batch_count = 0

        # Public attributes by which to interact with test problems. These have to
        # be created by the set_up function of sub-classes.
        self.data = None
        self.loss_function = None
        self.net = None
        self.regularization_groups = None

        self._batch_count = 0

    def train_init_op(self):
        """Initialize the testproblem instance to train mode.

        I.e. sets the iterator to the training set and sets the model to train mode.
        """
        self._iterator = iter(self.data._train_dataloader)
        self.phase = "train"
        self.net.train()

    def train_eval_init_op(self):
        """Initialize the testproblem instance to train eval mode.

        I.e. sets the iterator to the train evaluation set and sets the model to eval mode.
        """
        self._iterator = iter(self.data._train_eval_dataloader)
        self.phase = "train_eval"
        self.net.eval()

    def valid_init_op(self):
        """Initialize the testproblem instance to validation mode.

        I.e. sets the iterator to the validation set and sets the model to eval mode.
        """
        self._iterator = iter(self.data._valid_dataloader)
        self.phase = "valid"
        self.net.eval()

    def test_init_op(self):
        """Initialize the testproblem instance to test mode.

        I.e. sets the iterator to the test set and sets the model to eval mode.
        """
        self._iterator = iter(self.data._test_dataloader)
        self.phase = "test"
        self.net.eval()

    def _get_next_batch(self):
        """Return the next batch from the iterator."""
        batch = next(self._iterator)
        self._batch_count += 1
        return batch

    def get_batch_loss_and_accuracy_func(
        self, reduction="mean", add_regularization_if_available=True
    ):
        """Get new batch and create forward function.

        Creates the forward function that calculates loss and accuracy (if available)
        on that batch. This is a default implementation for image classification.
        Testproblems with different calculation routines (e.g. RNNs) overwrite
        this method accordingly.

        Args:
            reduction (str): The reduction that is used for returning the loss.
                Can be 'mean', 'sum' or 'none' in which case each indivual loss
                in the mini-batch is returned as a tensor.
            add_regularization_if_available (bool): If true, regularization is
                added to the loss.

        Returns:
            callable:  The function that calculates the loss/accuracy on the
                current batch.
        """
        inputs, labels = self._get_next_batch()
        inputs = inputs.to(self._device)
        labels = labels.to(self._device)

        def forward_func():
            correct = 0.0
            total = 0.0

            # in evaluation phase is no gradient needed
            if self.phase in ["train_eval", "test", "valid"]:
                with torch.no_grad():
                    outputs = self.net(inputs)
                    loss = self.loss_function(reduction=reduction)(outputs, labels)
            else:
                outputs = self.net(inputs)
                loss = self.loss_function(reduction=reduction)(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.numel()
            correct += (predicted == labels).sum().item()

            accuracy = correct / total

            if add_regularization_if_available:
                regularizer_loss = self.get_regularization_loss()
            else:
                regularizer_loss = torch.tensor(0.0, device=torch.device(self._device))

            return loss + regularizer_loss, accuracy

        return forward_func

    def get_batch_loss_and_accuracy(
        self, reduction="mean", add_regularization_if_available=True
    ):
        """Get a new batch and calculates the loss and accuracy on that batch.

        Args:
            reduction (str): The reduction that is used for returning the loss.
                Can be 'mean', 'sum' or 'none' in which case each indivual loss
                in the mini-batch is returned as a tensor.
            add_regularization_if_available (bool): If true, regularization is
                added to the loss.

        Returns:
            float/torch.tensor, float: loss and accuracy of the model on the
                current batch.
        """
        forward_func = self.get_batch_loss_and_accuracy_func(
            reduction=reduction,
            add_regularization_if_available=add_regularization_if_available,
        )

        return forward_func()

    def get_regularization_loss(self):
        """Return the current regularization loss of the network based on the parameter groups.

        Returns:
            int or torch.tensor: If no regularzations is applied, it returns the integer 0. Else a torch.tensor \
            that holds the regularization loss.
        """
        # iterate through all layers
        layer_norms = []
        for regularization, parameter_group in self.regularization_groups.items():
            if regularization > 0.0:
                # L2 regularization
                for parameters in parameter_group:
                    layer_norms.append(regularization * parameters.pow(2).sum())

        regularization_loss = 0.5 * sum(layer_norms)
        return regularization_loss

    @abc.abstractmethod
    def get_regularization_groups(self):
        """Create regularization groups for the parameters.

        Returns:
            dict: A dictionary where the key is the regularization factor and the value is a list of parameters.
        """
        return

    @abc.abstractmethod
    # TODO get rid of setup structure by parsing individual loss func, network and dataset
    def set_up(self):
        """Set up the test problem."""
        pass


class WeightRegularizedTestproblem(TestProblem):
    """Test problem with l2 regularization on weights, none on bias."""

    def get_regularization_groups(self):
        """Create regularization groups for the parameters.

        Returns:
            dict: A dictionary where the key is the regularization factor and the value is a list of parameters.
        """
        return self._make_groups_regularize_only_weights(self.net, self._l2_reg)

    @staticmethod
    def _make_groups_regularize_only_weights(net, l2_reg):
        """Create regularization groups with l2 regularization for the weights.

        Args:
            net (nn.Module) :
                Network to be regularized
            l2_reg (float) :
                l2 regularization strength
        Returns:
            dict:
                Keys correspond to l2 regularization strengths, values contain
                a list parameters that that are regularized accordingly.
        """
        no_reg = 0.0
        group_dict = {no_reg: [], l2_reg: []}

        def is_bias(name):
            return "bias" in name

        def is_weight(name):
            return "weight" in name

        def reg_strength(name):
            if is_bias(name):
                return no_reg
            elif is_weight(name):
                return l2_reg
            else:
                raise ValueError(name + "cannot be classified as weight or bias")

        for param_name, param in net.named_parameters():
            append_key = reg_strength(param_name)
            group_dict[append_key].append(param)

        return group_dict


class UnregularizedTestproblem(TestProblem):
    """Test problem with no regularization."""

    def __init__(self, batch_size, l2_reg=None):
        """Create a new test problem instance.

        Args:
            batch_size (int): Batch size to use.
            l2_reg (float): L2-Regularization (weight decay) factor to use.
        """
        super(UnregularizedTestproblem, self).__init__(batch_size, l2_reg)

    def get_regularization_groups(self):
        """Create regularization groups for the parameters.

        Returns:
            dict: A dictionary where the key is the regularization factor and the value is a list of parameters.
        """
        no = 0.0
        group_dict = {no: []}

        for parameters_name, parameters in self.net.named_parameters():
            # penalize no parameters
            group_dict[no].append(parameters)
        return group_dict

    @abc.abstractmethod
    def set_up(self):
        """Set up the test problem."""
        pass
