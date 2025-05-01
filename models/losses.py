import torch
import torch.nn.functional as F


def LabelSmoothingLoss(
    predictions,
    ground_truth,
):
    """
    Compute the label smoothing loss.
    input:
    - predictions: a tensor of shape (batch_size * seq_len, vocab_size)
    - ground_truth: a tensor of shape (batch_size * seq_len,)

    output:
    - loss: a tensor of shape (batch_size, seq_len)

    formula:
    loss = (\sum_{i=1}^{batch_size * seq_len} -\sum_{j=1}^{vocab_size} p_{ij} \log q_{ij}) / (batch_size * seq_len)
    """
    # get the number of classes
    num_classes = predictions.shape[1]

    # get the one-hot vector
    # shape: (batch_size * seq_len, vocab_size)
    one_hot = F.one_hot(ground_truth, num_classes=num_classes)

    # apply label smoothing
    # make the probability of the correct class to be 0.9
    # and the probability of the other classes to be 0.1 / (num_classes - 1)
    smoothed_one_hot = one_hot * (1 - 0.1) + 0.1 / (num_classes - 1)

    # compute the loss
    loss = -torch.mean(
        torch.sum(smoothed_one_hot * torch.log_softmax(predictions, dim=-1), dim=-1)
    )

    return loss


def CrossEntropyLoss(
    predictions,
    ground_truth,
):
    """
    Compute the cross entropy loss.
    input:
    - predictions: a tensor of shape (batch_size * seq_len, vocab_size)
    - ground_truth: a tensor of shape (batch_size * seq_len,)
    output:
    - loss: a tensor of shape (batch_size, seq_len)

    formula:
    loss = (-\sum_{i=1}^{batch_size * seq_len} \sum_{j=1}^{vocab_size} p_{ij} \log q_{ij}) / (batch_size * seq_len)
    """
    return F.cross_entropy(predictions, ground_truth, reduction="mean")
