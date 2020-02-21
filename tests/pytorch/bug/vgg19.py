from deepobs.pytorch.testproblems import cifar100_vgg19, cifar10_vgg19


def perform_train_forward_pass(tproblem):
    tproblem.set_up()
    tproblem.train_init_op()
    loss, acc = tproblem.get_batch_loss_and_accuracy()
    return loss, acc


def test_train_forward_pass_cifar100_vgg19():
    tproblem = cifar100_vgg19(batch_size = 1)
    perform_train_forward_pass(tproblem)


def test_train_forward_pass_cifar10_vgg19():
    tproblem = cifar10_vgg19(batch_size = 1)
    perform_train_forward_pass(tproblem)
