import torch


def positional_encoding_sinusoidal(seq_len, emb_dim):
    """
    Compute the positional encoding using the sinusoidal function.
    input:
    - seq_len(int): length of the sequence
    - emb_dim(int): dimension of the embedding

    output:
    - positional_encoding: a tensor of shape (seq_len, emb_dim)

    the formula is:
    PE(pos, 2i) = sin(pos / 10000^(2i / emb_dim))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / emb_dim))
    """
    # create a tensor of shape (1, seq_len, emb_dim)
    positional_encoding = torch.zeros(seq_len, emb_dim)

    # create the terms of the position and the terms of the exponent
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(
        1
    )  # shape: (seq_len, 1)
    div_term = torch.exp(
        torch.arange(0, emb_dim, 2).float()
        * (-torch.log(torch.tensor(10000.0)))
        / emb_dim
    )  # shape: (emb_dim / 2)

    # compute the positional encoding
    positional_encoding[:, 0::2] = torch.sin(position * div_term)
    positional_encoding[:, 1::2] = torch.cos(position * div_term)
    return positional_encoding
