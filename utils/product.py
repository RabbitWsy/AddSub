import torch


def scaled_dot_product(query, key, value, mask=None):
    """
    Compute the dot product between query and key, and then apply the softmax function to it.
    The result is then multiplied by the value.

    input:
    - query: a tensor of shape (batch_size, seq_len_q, d_model)
    - key: a tensor of shape (batch_size, seq_len, d_model)
    - value: a tensor of shape (batch_size, seq_len, d_model_v)
    - mask: a tensor of shape (batch_size, seq_len, seq_len)

    output:
    - output: a tensor of shape (batch_size, seq_len_q, d_model)
    """
    # get the dimension of the last axis
    d_k = query.size(-1)

    # compute the dot product between query and key, then divide it by the square root of d_k
    # shape of scores: (batch_size, seq_len_q, seq_len)
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
        torch.tensor(d_k, dtype=torch.float32)
    )

    # apply the mask to the scores
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)

    # apply the softmax function to the scores
    # shape of attention_weights: (batch_size, seq_len_q, seq_len)
    attention_weights = torch.softmax(scores, dim=-1)

    # compute the output
    # shape of output: (batch_size, seq_len_q, d_model_v)
    output = torch.matmul(attention_weights, value)
    return output, attention_weights
