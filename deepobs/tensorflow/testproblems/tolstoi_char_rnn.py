# -*- coding: utf-8 -*-
"""A two-layer LSTM for character-level language modelling on Tolstoi's War and Peace."""

import tensorflow as tf

from ..datasets.tolstoi import tolstoi
from .testproblem import TestProblem


class tolstoi_char_rnn(TestProblem):
    """DeepOBS test problem class for a two-layer LSTM for character-level language
  modelling (Char RNN) on Tolstoi's War and Peace.

  Some network characteristics:

  - ``128`` hidden units per LSTM cell
  - sequence length ``50``
  - cell state is automatically stored in variables between subsequent steps
  - when the phase placeholder swithches its value from one step to the next,
    the cell state is set to its zero value (meaning that we set to zero state
    after each round of evaluation, it is therefore important to set the
    evaluation interval such that we evaluate after a full epoch.)

  Working training parameters are:

  - batch size ``50``
  - ``200`` epochs
  - SGD with a learning rate of :math:`\\approx 0.1` works

  Args:
    batch_size (int): Batch size to use.
    weight_decay (float): No weight decay (L2-regularization) is used in this
        test problem. Defaults to ``None`` and any input here is ignored.

  Attributes:
    dataset: The DeepOBS data set class for Tolstoi.
    train_init_op: A tensorflow operation initializing the test problem for the
        training phase.
    train_eval_init_op: A tensorflow operation initializing the test problem for
        evaluating on training data.
    test_init_op: A tensorflow operation initializing the test problem for
        evaluating on test data.
    losses: A tf.Tensor of shape (batch_size, ) containing the per-example loss
        values.
    regularizer: A scalar tf.Tensor containing a regularization term.
    accuracy: A scalar tf.Tensor containing the mini-batch mean accuracy.
  """

    def __init__(self, batch_size, weight_decay=None):
        """Create a new Char RNN test problem instance on Tolstoi.

        Args:
          batch_size (int): Batch size to use.
          weight_decay (float): No weight decay (L2-regularization) is used in this
              test problem. Defaults to ``None`` and any input here is ignored.
        """
        super(tolstoi_char_rnn, self).__init__(batch_size, weight_decay)

        if weight_decay is not None:
            print(
                "WARNING: Weight decay is non-zero but no weight decay is used",
                "for this model."
            )

    def set_up(self):
        """Set up the Char RNN test problem instance on Tolstoi."""
        self.dataset = tolstoi(self._batch_size)

        seq_length = 50
        vocab_size = 83  # For War and Peace

        x, y = self.dataset.batch

        num_layers = 2
        rnn_size = 128

        input_keep_prob = tf.cond(
            tf.equal(self.dataset.phase, tf.constant("train")),
            lambda: tf.constant(0.8), lambda: tf.constant(1.0))
        output_keep_prob = tf.cond(
            tf.equal(self.dataset.phase, tf.constant("train")),
            lambda: tf.constant(0.8), lambda: tf.constant(1.0))

        # Create an embedding matrix, look up embedding of input
        embedding = tf.get_variable("embedding", [vocab_size, rnn_size])
        inputs = tf.nn.embedding_lookup(embedding, x)

        # Split batch of input sequences along time, such that inputs[i] is a
        # batch_size x embedding_size representation of the batch of characters
        # at position i of this batch of sequences
        inputs = tf.split(inputs, seq_length, axis=1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        # Make Multi LSTM cell
        cells = []
        for _ in range(num_layers):
            cell = tf.contrib.rnn.LSTMCell(rnn_size)
            cell = tf.contrib.rnn.DropoutWrapper(
                cell,
                input_keep_prob=input_keep_prob,
                output_keep_prob=output_keep_prob)
            cells.append(cell)
        cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        # Create RNN using the cell defined above, (including operations that store)
        # the state in variables
        self.state_variables, self.zero_states = self._get_state_variables(
            self._batch_size, cell)

        outputs, new_states = tf.nn.static_rnn(
            cell, inputs, initial_state=self.state_variables)
        with tf.control_dependencies(outputs):
            state_update_op = self._get_state_update_op(self.state_variables,
                                                       new_states)

        # Reshape RNN output for multiplication with softmax layer
        # print "Shape of outputs", [output.get_shape() for output in outputs]
        with tf.control_dependencies(state_update_op):
            output = tf.reshape(tf.concat(outputs, 1), [-1, rnn_size])
        # print "Shape of output", output.get_shape()

        # Apply softmax layer
        with tf.variable_scope("rnnlm"):
            softmax_w = tf.get_variable("softmax_w", [rnn_size, vocab_size])
            softmax_b = tf.get_variable("softmax_b", [vocab_size])
        logits = tf.matmul(output, softmax_w) + softmax_b
        # print logits.get_shape()

        # Reshape logits to batch_size x seq_length x vocab size
        reshaped_logits = tf.reshape(
            logits, [self._batch_size, seq_length, vocab_size])
        # print "Shape of reshaped logits", reshaped_logits.get_shape()

        # Create vector of losses
        self.losses = tf.contrib.seq2seq.sequence_loss(
            reshaped_logits,
            y,
            weights=tf.ones([self._batch_size, seq_length], dtype=tf.float32),
            average_across_timesteps=True,
            average_across_batch=False)

        predictions = tf.argmax(reshaped_logits, 2)
        correct_prediction = tf.equal(predictions, y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.regularizer = tf.losses.get_regularization_loss()

        self.train_init_op = tf.group([
            self.dataset.train_init_op,
            self._get_state_update_op(self.state_variables, self.zero_states)
        ])
        self.train_eval_init_op = tf.group([
            self.dataset.train_eval_init_op,
            self._get_state_update_op(self.state_variables, self.zero_states)
        ])
        self.test_init_op = tf.group([
            self.dataset.test_init_op,
            self._get_state_update_op(self.state_variables, self.zero_states)
        ])

    def _get_state_variables(self, batch_size, cell):
        """For each layer, get the initial state and make a variable out of it
        to enable updating its value.

        Args:
            batch_size (int): Batch size.
            cell (tf.BasicLSTMCell): LSTM cell to get the initial state for.

        Returns:
            tupel: Tupel of the state variables and there zero states.

        """
        # For each layer, get the initial state and make a variable out of it
        # to enable updating its value.
        zero_state = cell.zero_state(batch_size, tf.float32)
        state_variables = []
        for state_c, state_h in zero_state:
            state_variables.append(
                tf.contrib.rnn.LSTMStateTuple(
                    tf.Variable(state_c, trainable=False),
                    tf.Variable(state_h, trainable=False)))
        # Return as a tuple, so that it can be fed to dynamic_rnn as an initial state
        return tuple(state_variables), zero_state

    def _get_state_update_op(self, state_variables, new_states):
        """Add an operation to update the train states with the last state tensors

        Args:
            state_variables (tf.Variable): State variables to be updated
            new_states (tf.Variable): New state of the state variable.

        Returns:
            tf.Operation: Return a tuple in order to combine all update_ops into a
            single operation. The tuple's actual value should not be used.

        """
        # Add an operation to update the train states with the last state tensors
        update_ops = []
        for state_variable, new_state in zip(state_variables, new_states):
            # Assign the new state to the state variables on this layer
            update_ops.extend([
                state_variable[0].assign(new_state[0]),
                state_variable[1].assign(new_state[1])
            ])
        # Return a tuple in order to combine all update_ops into a single operation.
        # The tuple's actual value should not be used.
        return tf.tuple(update_ops)
