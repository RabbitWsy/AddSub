import torch


def create_padding_mask(seq):
    """
    Create a padding mask for the input sequence.

    input:
    - seq: a tensor of shape (batch_size, seq_len)

    output:
    - mask: a tensor of shape (batch_size, seq_len, seq_len)
    """
    # get the shape of the input sequence
    batch_size, seq_len = seq.shape

    # create a tensor of shape (batch_size, seq_len, seq_len) with all ones
    mask = torch.ones(
        (batch_size, seq_len, seq_len), device=seq.device, dtype=torch.bool
    )

    # fill the lower triangle with zeros
    mask = torch.triu(mask, diagonal=1)
    return mask
