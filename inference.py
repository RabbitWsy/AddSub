import torch
import os
import argparse
import tomllib

from models.transformer import Transformer
from utils.position_encoding import positional_encoding_sinusoidal
from utils.preprocessing_data import preprocessing_data
from data.data_loader import convert_str_to_tokens, SPECIAL_TOKENS


def parse_args():
    """
    Parse the arguments.
    output:
    - args(Namespace): the arguments
    """
    parser = argparse.ArgumentParser()

    # add the arguments
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="the path of the config file",
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="the question to be answered",
    )

    return parser.parse_args()


def inference(
    model,
    input_seq,
    input_pos,
    output_pos,
    output_len,
    device,
):
    """
    input:
    - model(nn.Module): the trained model
    - input_seq(torch.Tensor): the input sequence of shape (seq_len_input)
    - input_pos(torch.Tensor): the input position of shape (seq_len_input, emb_dim)
    - output_pos(torch.Tensor): the output position of shape (seq_len_output, emb_dim)
    - output_len(int): the length of the output
    - device(torch.device): the device to run the model

    output:
    - output(torch.Tensor): the output sequence
    """
    # set the model to evaluation mode
    model.eval()

    # initial output is 14 meaning "BOS"
    # shape: (1,)
    output = torch.tensor([14], dtype=torch.long, device=device)

    # shape: (seq_len_input, emb_dim)
    question_emb = model.emb_layer(input_seq)
    question_emb = question_emb + input_pos

    # get the output of the encoder
    # shape: (seq_len_input, emb_dim)
    enc_out = model.encoder(question_emb)

    for i in range(output_len - 1):
        # get the output embedding
        # shape: (1 + i, emb_dim)
        ans_emb = model.emb_layer(output)
        ans_emb = ans_emb + output_pos[: i + 1]

        # get the output of the decoder
        # shape: (1 + i, vocab_size)
        dec_out = model.decoder(ans_emb, enc_out)

        # get the last token
        # shape: (1 + i,)
        dec_out_max = dec_out.argmax(dim=-1)
        # shape: (1,)
        last_token = dec_out_max[-1:]

        # append the last token to the output
        # shape: (2 + i,)
        output = torch.cat([output, last_token], dim=0)

    return output


if __name__ == "__main__":
    args = parse_args()

    # load the config file
    with open(args.config, "rb") as f:  # must use binary mode
        config = tomllib.load(f)

    # set the hyperparameters
    emb_dim = config["model"]["emb_dim"]
    num_heads = config["model"]["num_heads"]
    ff_dim = config["model"]["ff_dim"]
    num_enc_layers = config["model"]["num_enc_layers"]
    num_dec_layers = config["model"]["num_dec_layers"]
    dropout = config["model"]["dropout"]
    vocab_size = config["model"]["vocab_size"]
    # data config
    question_len = config["data"]["question_len"]
    answer_len = config["data"]["answer_len"]
    # set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create the model
    model = Transformer(
        emb_dim,
        num_heads,
        ff_dim,
        num_enc_layers,
        num_dec_layers,
        dropout,
        vocab_size,
    )
    # load the model
    model.load_state_dict(
        torch.load(os.path.join(config["model"]["pretrained_model"]), weights_only=True)
    )

    # input the question
    input_seq = args.question
    # preprocess the input sequence
    # shape: (seq_len,)
    input_seq = preprocessing_data(input_seq, convert_str_to_tokens, SPECIAL_TOKENS)
    input_seq = torch.tensor(input_seq, dtype=torch.long)
    # create the input position
    # shape: (seq_len, emb_dim)
    input_pos = positional_encoding_sinusoidal(question_len, emb_dim)
    output_pos = positional_encoding_sinusoidal(answer_len, emb_dim)
    # set the model to device
    model = model.to(device)
    input_seq = input_seq.to(device)
    input_pos = input_pos.to(device)
    output_pos = output_pos.to(device)

    # inference
    # shape: (answer_len,)
    output = inference(model, input_seq, input_pos, output_pos, answer_len, device)

    # convert the output to string
    output = output.tolist()
    convert_tokens_to_str = {v: k for k, v in convert_str_to_tokens.items()}
    output = [convert_tokens_to_str[i] for i in output]
    output = " ".join(output)
    # print the output
    print(output)
