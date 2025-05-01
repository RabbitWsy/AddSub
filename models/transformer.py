import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import EncoderBlock, DecoderBlock
from utils.mask import create_padding_mask


class Encoder(nn.Module):
    def __init__(self, emb_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        """
        implement the output through multiple EncoderBlock

        init_input:
        - emb_dim(int): dimension of the input
        - num_heads(int): number of heads in the multi-head attention
        - ff_dim(int): dimension of the hidden layer
        - num_layers(int): number of layers
        - dropout(float): dropout rate

        forward_input:
        - x: a tensor of shape (batch_size, seq_len, emb_dim)

        forward_output:
        - output: a tensor of shape (batch_size, seq_len, emb_dim)

        structure: EncoderBlock -> EncoderBlock -> ... -> EncoderBlock
        """
        super().__init__()

        self.layers = nn.ModuleList(
            [
                EncoderBlock(emb_dim, num_heads, ff_dim, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        # iterate through each layer
        for layer in self.layers:
            x = layer(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self, emb_dim, num_heads, ff_dim, num_layers, dropout=0.1, vocab_size=None
    ):
        """
        implement the output through multiple DecoderBlock

        init_input:
        - emb_dim(int): dimension of the input
        - num_heads(int): number of heads in the multi-head attention
        - ff_dim(int): dimension of the hidden layer
        - num_layers(int): number of layers
        - dropout(float): dropout rate
        - vocab_size(int): size of the vocabulary

        forward_input:
        - x: a tensor of shape (batch_size, seq_len, emb_dim)
        - encoder_output: a tensor of shape (batch_size, seq_len, emb_dim)
        - mask: a tensor of shape (batch_size, seq_len, seq_len)

        forward_output:
        - output: a tensor of shape (batch_size, seq_len, vocab_size)

        structure: DecoderBlock -> DecoderBlock ->... -> DecoderBlock
        """
        super().__init__()

        self.layers = nn.ModuleList(
            [
                DecoderBlock(emb_dim, num_heads, ff_dim, dropout)
                for _ in range(num_layers)
            ]
        )

        # project the embedding dimension of the output back to the vocabulary size
        self.linear = nn.Linear(emb_dim, vocab_size)

        # xavier initialization
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x, encoder_output, mask=None):
        # iterate through each layer
        for layer in self.layers:
            x = layer(x, encoder_output, mask)

        # project the embedding dimension of the output back to the vocabulary size
        output = self.linear(x)
        return output


class Transformer(nn.Module):
    def __init__(
        self,
        emb_dim,
        num_heads,
        ff_dim,
        num_enc_layers,
        num_dec_layers,
        dropout=0.1,
        vocab_size=None,
    ):
        """
        init_input:
        - emb_dim(int): dimension of the input
        - num_heads(int): number of heads in the multi-head attention
        - ff_dim(int): dimension of the hidden layer
        - num_enc_layers(int): number of layers in the encoder
        - num_dec_layers(int): number of layers in the decoder
        - dropout(float): dropout rate
        - vocab_size(int): size of the vocabulary

        forward_input:
        - question(LongTensor): a tensor of shape (batch_size, seq_len_question)
        - question_pos(FloatTensor): a tensor of shape (seq_len_question, emb_dim)
        - answer(LongTensor): a tensor of shape (batch_size, seq_len_answer)
        - answer_pos(FloatTensor): a tensor of shape (seq_len_answer, emb_dim)

        forward_output:
        - output: a tensor of shape (batch_size * (seq_len_answer - 1), vocab_size)
        """
        super().__init__()

        # making the index of the input to an embedding vector by adding a new dimension
        # the embedding vector is a learnable parameter
        self.emb_layer = nn.Embedding(vocab_size, emb_dim)

        # define the encoder and decoder
        self.encoder = Encoder(emb_dim, num_heads, ff_dim, num_enc_layers, dropout)
        self.decoder = Decoder(
            emb_dim, num_heads, ff_dim, num_dec_layers, dropout, vocab_size
        )

        # xavier initialization
        nn.init.xavier_uniform_(self.emb_layer.weight)

    def forward(self, question, question_pos, answer, answer_pos):
        # shape: (batch_size, seq_len_q_a, emb_dim)
        question_emb = self.emb_layer(question)
        answer_emb = self.emb_layer(answer)

        # add the position embedding to the embedding vector
        question_emb += question_pos
        answer_emb += answer_pos

        # shift the answer embedding to the left by one position
        # so that the decoder can predict the next word
        # shape: (batch_size, seq_len_answer - 1, emb_dim)
        answer_emb = answer_emb[:, :-1]

        # create the mask for the decoder
        mask = create_padding_mask(answer[:, :-1])

        # pass the question and answer through the encoder and decoder
        # shape: (batch_size, seq_len_question, vocab_size)
        encoder_output = self.encoder(question_emb)
        # shape: (batch_size, seq_len_answer - 1, vocab_size)
        output = self.decoder(answer_emb, encoder_output, mask)

        # reshape the output to (batch_size * (seq_len_answer - 1), vocab_size)
        output = output.view(-1, output.size(-1))
        return output
