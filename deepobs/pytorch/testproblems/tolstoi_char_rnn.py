import torch
from torch import nn
from .testproblems_modules import net_char_rnn
from ..datasets.tolstoi import tolstoi
from .testproblem import TestProblem

class tolstoi_char_rnn(TestProblem):
    def __init__(self, batch_size, weight_decay=None):
        super(tolstoi_char_rnn, self).__init__(batch_size, weight_decay)

    def set_up(self):
        """Set up the vanilla CNN test problem on Cifar-10."""
        self.data = tolstoi(self._batch_size)
        self.loss_function = nn.CrossEntropyLoss()
        self.net = net_char_rnn(seq_len=50, hidden_dim=128, vocab_size=83, num_layers=2)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#        self._device = torch.device('cpu')
        self.net.to(self._device)

#    def get_regularization_loss(self):
#        # iterate through all layers
#        layer_norms = []
#        for parameters_name, parameters in self.net.named_parameters():
#            # penalize only the non bias layer parameters
#            if ('weight' in parameters_name) and (('dense' in parameters_name) or ('conv' in parameters_name)):
#                # L2 regularization
#                layer_norms.append(parameters.norm(2)**2)
#
#        regularization_loss = 0.5 * sum(layer_norms)
#
#        return self._weight_decay * regularization_loss

    def get_batch_loss_and_accuracy(self):
        inputs, labels = self._get_next_batch()
#        inputs.unsqueeze_(2)
#        labels.unsqueeze_(2)
        inputs = inputs.to(self._device)
        labels = labels.to(self._device)
        correct = 0.0
        total = 0.0

        # in evaluation phase is no gradient needed
        if self.phase in ["train_eval", "test"]:
            with torch.no_grad():
                outputs = self.net(inputs)
                # reshape for loss
                outputs = outputs.view(-1, outputs.size(2))
                labels = labels.view(-1)
                loss = self.loss_function(outputs, labels)
        else:
            outputs = self.net(inputs)
            outputs = outputs.view(-1, outputs.size(2))
            labels = labels.view(-1)
            loss = self.loss_function(outputs, labels)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        accuracy = correct/total
        return loss, accuracy