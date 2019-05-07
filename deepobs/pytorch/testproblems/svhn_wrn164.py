import torch
from torch import nn
from .testproblems_modules import net_wrn
from ..datasets.svhn import svhn
from .testproblem import TestProblem

class svhn_wrn164(TestProblem):
    def __init__(self, batch_size, weight_decay=0.0005):
        super(svhn_wrn164, self).__init__(batch_size, weight_decay)

    def set_up(self):
        """Set up the vanilla CNN test problem on Cifar-10."""
        self.data = svhn(self._batch_size, data_augmentation=True)
        self.loss_function = nn.CrossEntropyLoss()
        self.net = net_wrn(num_outputs=10, num_residual_blocks=2, widening_factor=4)
#        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = torch.device('cpu')
        self.net.to(self._device)

    def get_regularization_loss(self):
        # iterate through all layers
        layer_norms = []
        for parameters_name, parameters in self.net.named_parameters():
            # penalize only the non bias layer parameters
            if ('weight' in parameters_name) and (('dense' in parameters_name) or ('conv' in parameters_name)):
                # L2 regularization
                layer_norms.append(parameters.pow(2).sum())

        regularization_loss = 0.5 * sum(layer_norms)

        return self._weight_decay * regularization_loss