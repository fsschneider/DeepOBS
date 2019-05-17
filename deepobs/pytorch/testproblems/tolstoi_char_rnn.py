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
        self.net.to(self._device)

    # override the init operation for training to reset the hidden states and cell states after every epoch
    def train_init_op(self):
        self._iterator = iter(self.data._train_dataloader)
        self.phase = "train"
        self.net.train()
        self._reset_state()

    def _reset_state(self):
        hidden_state = torch.zeros((2, self._batch_size, 128)).to(self._device)
        cell_state = torch.zeros((2, self._batch_size, 128)).to(self._device)
        self.state = (hidden_state, cell_state)

    def get_batch_loss_and_accuracy(self):
        inputs, labels = self._get_next_batch()
        inputs = inputs.to(self._device)
        labels = labels.to(self._device)
        correct = 0.0
        total = 0.0

        # in evaluation phase is no gradient needed. cell states default to zero if not provided
        if self.phase in ["train_eval", "test"]:
            with torch.no_grad():
                outputs, _ = self.net(inputs)
                # reshape for loss
                outputs = outputs.view(-1, outputs.size(2))
                labels = labels.view(-1)
                loss = self.loss_function(outputs, labels)
        else:
            outputs, (hidden_state, cell_state) = self.net(inputs, self.state)
            # detach state from backpropagation
            self.state = (hidden_state.detach(), cell_state.detach())
            outputs = outputs.view(-1, outputs.size(2))
            labels = labels.view(-1)
            loss = self.loss_function(outputs, labels)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        accuracy = correct/total
        return loss, accuracy