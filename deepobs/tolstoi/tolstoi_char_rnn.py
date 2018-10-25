# -*- coding: utf-8 -*-
"""A two-layer LSTM for character-level language modelling

Large parts of this code are adapted from github.com/sherjilozair/char-rnn-tensorflow.

Some characteristics:
- 128 hidden units per LSTM cell
- sequence length 50
- cell state is automatically stored in variables between subsequent steps
- when the phase placeholder swithches its value from one step to the next,
  the cell state is set to its zero value (meaning that we set to zero state
  after each round of evaluation, it is therefore important to set the evaluation interval such that we evaluate after a full epoch.)

Training parameters:
- batch size 50 (yields 1045 batches in train set and 261 in test set)
- 209000 total steps
- evaluate after 2090 steps (2 epochs)
- eval_size 13050 (i.e. 261 batches, exactly the size of the test set)
- plain SGD with roughly lr=0.1 works
"""

import tensorflow as tf

import tolstoi_input


class set_up:
    """Class providing the functionality for recurrent neural network providing character-level language modelling on the data set `Tolstoi`.

    The network has two LSTM layers, each with ``128`` hidden units. We use a sequence length of ``50`` for the data set. The cell state is automatically stored in variables between subsequent steps and reset after each epoch (or when switching the ``phase``).

    Large parts of this code are adapted from `here`_.

    Suggested training settings are: Batch size of ``50`` and training for a total of ``200`` epochs.

    Args:
        batch_size (int): Batch size of the data points. No default value is specified.
        weight_decay (float): Weight decay factor. In this model there is no weight decay implemented. Defaults to ``None``.

    Attributes:
        seq_length (int): Sequence lenght, defined as the number of characters. Defaults to ``50``.
        data_loading (deepobs.data_loading): Data loading class for tolstoi data set, :class:`.tolstoi_input.data_loading`.
        losses (tf.Tensor): Tensor of size ``batch_size`` containing the individual losses per data point.
        accuracy (tf.Tensor): Tensor containing the accuracy of the model.
        train_init_op (tf.Operation): A TensorFlow operation to be performed before starting every training epoch. Among other things, it sets the state variables to the zero state.
        train_eval_init_op (tf.Operation): A TensorFlow operation to be performed before starting every training eval epoch. Among other things, it sets the state variables to the zero state.
        test_init_op (tf.Operation): A TensorFlow operation to be performed before starting every test evaluation phase. Among other things, it sets the state variables to the zero state.

    .. _here:  github.com/sherjilozair/char-rnn-tensorflow.
    """
    def __init__(self, batch_size, weight_decay=None):
        """Initializes the problem set_up class.

        Args:
            batch_size (int): Batch size of the data points. No default value is specified.
            weight_decay (float): Weight decay factor. In this model there is no weight decay implemented. Defaults to ``None``.

        """
        self.seq_length = 50
        self.batch_size = batch_size
        self.data_loading = tolstoi_input.data_loading(batch_size=self.batch_size, seq_length=self.seq_length)
        self.losses, self.accuracy = self.set_up(weight_decay)

        # Operations to do when switching the phase (the one defined in data_loading initializes the iterator and assigns the phase variable, here you can add more operations)
        self.train_init_op = tf.group([self.data_loading.train_init_op, self.get_state_update_op(self.state_variables, self.zero_states)])
        self.train_eval_init_op = tf.group([self.data_loading.train_eval_init_op, self.get_state_update_op(self.state_variables, self.zero_states)])
        self.test_init_op = tf.group([self.data_loading.test_init_op, self.get_state_update_op(self.state_variables, self.zero_states)])

    def get(self):
        """Returns the losses and the accuray of the model.

        Returns:
            tupel: Tupel consisting of the losses and the accuracy.

        """
        return self.losses, self.accuracy

    def set_up(self, weight_decay):
        """Sets up the test problem.

        Args:
            weight_decay (float): Weight decay factor. In this model there is no weight decay implemented.

        Returns:
            tupel: Tupel consisting of the losses and the accuracy.

        """
        if weight_decay is not None:
            print "WARNING: Weight decay is non-zero but no weight decay is used for this model."

        vocab_size = 83  # For War and Peace

        input_data, targets, phase = self.data_loading.load()
        # print "Shape of input_data", input_data.get_shape()

        num_layers = 2
        rnn_size = 128

        input_keep_prob = tf.cond(tf.equal(phase, tf.constant("train")),
                                  lambda: tf.constant(0.8),
                                  lambda: tf.constant(1.0))
        output_keep_prob = tf.cond(tf.equal(phase, tf.constant("train")),
                                   lambda: tf.constant(0.8),
                                   lambda: tf.constant(1.0))

        # Create an embedding matrix, look up embedding of input
        embedding = tf.get_variable("embedding", [vocab_size, rnn_size])
        inputs = tf.nn.embedding_lookup(embedding, input_data)
        # print "Shape of inputs", inputs.get_shape()

        # Split batch of input sequences along time, such that inputs[i] is a
        # batch_size x embedding_size representation of the batch of characters
        # at position i of this batch of sequences
        inputs = tf.split(inputs, self.seq_length, axis=1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
        # print "Shape of inputs", [inp.get_shape() for inp in inputs]

        # Make Multi LSTM cell
        cells = []
        for _ in range(num_layers):
            cell = tf.contrib.rnn.BasicLSTMCell(rnn_size)
            cell = tf.contrib.rnn.DropoutWrapper(cell,
                                                 input_keep_prob=input_keep_prob,
                                                 output_keep_prob=output_keep_prob)
            cells.append(cell)
        cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        # Create RNN using the cell defined above, (including operations that store)
        # the state in variables
        self.state_variables, self.zero_states = self.get_state_variables(self.batch_size, cell)

        outputs, new_states = tf.nn.static_rnn(cell, inputs, initial_state=self.state_variables)
        with tf.control_dependencies(outputs):
            state_update_op = self.get_state_update_op(self.state_variables, new_states)

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
        reshaped_logits = tf.reshape(logits, [self.batch_size, self.seq_length, vocab_size])
        print "Shape of reshaped logits", reshaped_logits.get_shape()

        # Create vector of losses
        losses = tf.contrib.seq2seq.sequence_loss(
            reshaped_logits,
            targets,
            weights=tf.ones([self.batch_size, self.seq_length], dtype=tf.float32),
            average_across_timesteps=True,
            average_across_batch=False)
        print "Shape of losses", losses.get_shape()

        predictions = tf.argmax(reshaped_logits, 2)
        correct_prediction = tf.equal(predictions, targets)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return losses, accuracy

    def get_state_variables(self, batch_size, cell):
        """For each layer, get the initial state and make a variable out of it to enable updating its value.

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
            state_variables.append(tf.contrib.rnn.LSTMStateTuple(
                tf.Variable(state_c, trainable=False),
                tf.Variable(state_h, trainable=False)))
        # Return as a tuple, so that it can be fed to dynamic_rnn as an initial state
        return tuple(state_variables), zero_state

    def get_state_update_op(self, state_variables, new_states):
        """Add an operation to update the train states with the last state tensors

        Args:
            state_variables (tf.Variable): State variables to be updated
            new_states (tf.Variable): New state of the state variable.

        Returns:
            tf.Operation: Return a tuple in order to combine all update_ops into a single operation. The tuple's actual value should not be used.

        """
        # Add an operation to update the train states with the last state tensors
        update_ops = []
        for state_variable, new_state in zip(state_variables, new_states):
            # Assign the new state to the state variables on this layer
            update_ops.extend([state_variable[0].assign(new_state[0]),
                               state_variable[1].assign(new_state[1])])
        # Return a tuple in order to combine all update_ops into a single operation.
        # The tuple's actual value should not be used.
        return tf.tuple(update_ops)
