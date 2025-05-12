import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.product import scaled_dot_product


class SelfAttention(nn.Module):
    def __init__(self, d_in, d_qk, d_v):
        """
        project the input into three different vectors: query, key, and value.
        compute the attention weights and use them to compute the weighted sum of the value vectors.

        init_input:
        - d_in(int): dimension of the input
        - d_qk(int): dimension of the query and key
        - d_v(int): dimension of the value

        forward_input:
        - query: a tensor of shape (batch_size, seq_len_q, d_in)
        - key: a tensor of shape (batch_size, seq_len_kv, d_in)
        - value: a tensor of shape (batch_size, seq_len_kv, d_in)

        forward_output:
        - output: a tensor of shape (batch_size, seq_len_q, d_v)
        - attention_weights: a tensor of shape (batch_size, seq_len_q, seq_len_kv)
        """
        super().__init__()

        # define three linear layers
        self.q = nn.Linear(d_in, d_qk)
        self.k = nn.Linear(d_in, d_qk)
        self.v = nn.Linear(d_in, d_v)

        # xavier initialization
        nn.init.xavier_uniform_(self.q.weight)
        nn.init.xavier_uniform_(self.k.weight)
        nn.init.xavier_uniform_(self.v.weight)

    def forward(self, query, key, value, mask=None):
        # compute the query, key, and value
        query = self.q(query)
        key = self.k(key)
        value = self.v(value)

        # compute the weighted sum of the value vectors
        output, self.attention_weights = scaled_dot_product(query, key, value, mask)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads):
        """
        use multiple heads to compute the attention in parallel
        and concatenate the output of each head to get the final output.

        init_input:
        - d_in(int): dimension of the input
        - d_out(int): dimension of the output
        - num_heads(int): number of heads

        forward_input:
        - query: a tensor of shape (batch_size, seq_len, d_in)
        - key: a tensor of shape (batch_size, seq_len, d_in)
        - value: a tensor of shape (batch_size, seq_len, d_in)
        - mask: a tensor of shape (batch_size, seq_len, seq_len)

        forward_output:
        - output: a tensor of shape (batch_size, seq_len, d_out)
        """
        super().__init__()
        # define the number of heads
        self.heads = nn.ModuleList(
            [SelfAttention(d_in, d_out, d_out) for _ in range(num_heads)]
        )

        # project the output of each head to the original dimension
        self.linear = nn.Linear(num_heads * d_out, d_in)

        # xavier initialization
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, query, key, value, mask=None):
        # compute the output of each head
        output = [head(query, key, value, mask) for head in self.heads]

        # concatenate the output of each head
        output = torch.cat(output, dim=-1)

        # transfer the output to the original dimension
        output = self.linear(output)
        return output


class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-12):
        """
        normalize the embeddings of the input to have a mean of 0 and a variance of 1.

        init_input:
        - emb_dim(int): dimension of the input
        - eps(float): a small number to avoid division by zero

        forward_input:
        - x: a tensor of shape (batch_size, seq_len, emb_dim)

        forward_output:
        - output: a tensor of shape (batch_size, seq_len, emb_dim)
        """
        super().__init__()

        self.epsilon = eps

        # the parameters are learnable
        self.beta = nn.Parameter(torch.zeros(emb_dim))
        self.gamma = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x):
        # compute the mean and variance of the input
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)

        # normalize the input
        x = (x - mean) / torch.sqrt(var + self.epsilon)

        # scale and shift the input
        x = self.gamma * x + self.beta
        return x


class FeedForward(nn.Module):
    def __init__(self, d_in_out, d_hidden):
        """
        a simple feedforward network with two linear layers and a ReLU activation function.

        init_input:
        - d_in_out(int): dimension of the input and output
        - d_hidden(int): dimension of the hidden layer

        forward_input:
        - x: a tensor of shape (batch_size, seq_len, emb_dim)

        forward_output:
        - output: a tensor of shape (batch_size, seq_len, emb_dim)

        structure: linear1 -> relu -> linear2
        """
        super().__init__()

        # define the linear layers
        # kaiming initialization for relu is default
        self.linear1 = nn.Linear(d_in_out, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_in_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.linear1(x)
        output = self.relu(output)
        output = self.linear2(output)
        return output


class EncoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, ff_dim, dropout=0.1):
        """
        init_input:
        - emb_dim(int): dimension of the input
        - num_heads(int): number of heads
        - ff_dim(int): dimension of the hidden layer
        - dropout(float): dropout rate

        forward_input:
        - x: a tensor of shape (batch_size, seq_len, emb_dim)

        forward_output:
        - output: a tensor of shape (batch_size, seq_len, emb_dim)

        structure: multiheads -> lay_norm1(output + x) -> dropout -> feedforward -> lay_norm2(output + output1) -> dropout
        """
        super().__init__()

        self.multiheads = MultiHeadAttention(
            emb_dim, int(emb_dim // num_heads), num_heads
        )
        self.lay_norm1 = LayerNorm(emb_dim)

        self.feedforward = FeedForward(emb_dim, ff_dim)
        self.lay_norm2 = LayerNorm(emb_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # shape: (batch_size, seq_len, emb_dim)
        output = self.multiheads(x, x, x)
        output = self.lay_norm1(output + x)
        output1 = self.dropout(output)

        output = self.feedforward(output1)
        output = self.lay_norm2(output + output1)
        output = self.dropout(output)
        return output


class DecoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, ff_dim, dropout=0.1):
        """
        init_input:
        - emb_dim(int): dimension of the input
        - num_heads(int): number of heads
        - ff_dim(int): dimension of the hidden layer
        - dropout(float): dropout rate

        forward_input:
        - decoder_input: a tensor of shape (batch_size, seq_len_de, emb_dim)
        - encoder_output: a tensor of shape (batch_size, seq_len_en, emb_dim)
        - mask: a tensor of shape (batch_size, seq_len_de, seq_len_de)

        forward_output:
        - output: a tensor of shape (batch_size, seq_len_de, emb_dim)

        structure: masked_self_attention -> lay_norm1 -> dropout -> cross_attention -> lay_norm2 -> dropout
                    -> feedforward -> lay_norm3 -> dropout
        """
        super().__init__()

        if emb_dim % num_heads != 0:
            raise ValueError("emb_dim must be divisible by num_heads")

        self.masked_self_attention = MultiHeadAttention(
            emb_dim, int(emb_dim // num_heads), num_heads
        )
        self.lay_norm1 = LayerNorm(emb_dim)

        self.cross_attention = MultiHeadAttention(
            emb_dim, int(emb_dim // num_heads), num_heads
        )
        self.lay_norm2 = LayerNorm(emb_dim)

        self.feedforward = FeedForward(emb_dim, ff_dim)
        self.lay_norm3 = LayerNorm(emb_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, decoder_input, encoder_output, mask=None):
        # shape: (batch_size, seq_len_de, emb_dim)
        out1 = self.masked_self_attention(
            decoder_input, decoder_input, decoder_input, mask
        )
        out1 = self.lay_norm1(out1 + decoder_input)
        out1 = self.dropout(out1)

        # shape: (batch_size, seq_len_de, emb_dim)
        out2 = self.cross_attention(out1, encoder_output, encoder_output)
        out2 = self.lay_norm2(out2 + out1)
        out2 = self.dropout(out2)

        out3 = self.feedforward(out2)
        out3 = self.lay_norm3(out3 + out2)
        out3 = self.dropout(out3)
        return out3
