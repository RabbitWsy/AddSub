import torch
import json
from torch.utils.data import Dataset, DataLoader
from utils.preprocessing_data import generate_token_dict, preprocessing_data
from utils.position_encoding import positional_encoding_sinusoidal

# train_data_path = "./data/raw/train_data.csv"
# val_data_path = "./data/raw/val_data.csv"
# test_data_path = "./data/raw/test_data.csv"

SPECIAL_TOKENS = ["POSITIVE", "NEGATIVE", "add", "subtract", "BOS", "EOS"]
vocab = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"] + SPECIAL_TOKENS

# generate a dictionary of tokens
convert_str_to_tokens = generate_token_dict(vocab)


class AddSubDataset(Dataset):
    def __init__(
        self,
        input_seqs,
        target_seqs,
        convert_str_to_tokens,
        special_tokens,
        emb_dim,
        positional_encoding,
    ):
        """
        init_input:
        - input_seqs(list): a list of input sequences
        - target_seqs(list): a list of target sequences
        - convert_str_to_tokens(dict): a dictionary of tokens
        - special_tokens(list): a list of special tokens
        - emb_dim(int): dimension of the embedding
        - positional_encoding(function): a function of positional encoding

        forward_output:
        - input_seq(torch.LongTensor): a tensor of shape (seq_len)
        - input_pos(torch.FloatTensor): a tensor of shape (seq_len, emb_dim)
        - target_seq(torch.LongTensor): a tensor of shape (seq_len)
        - target_pos(torch.FloatTensor): a tensor of shape (seq_len, emb_dim)
        """
        self.input_seqs = input_seqs
        self.target_seqs = target_seqs

        self.convert_str_to_tokens = convert_str_to_tokens
        self.special_tokens = special_tokens
        self.emb_dim = emb_dim

        self.positional_encoding = positional_encoding

    def preprocessing(self, seq):
        """
        preprocess the input sequence and target sequence
        input:
        - seq(list): a list of sequences

        output:
        - output(list): a list of tokens
        """
        return preprocessing_data(seq, self.convert_str_to_tokens, self.special_tokens)

    def __len__(self):
        return len(self.input_seqs)

    def __getitem__(self, idx):
        # get the input and target sequence
        input_seq = self.input_seqs[idx]
        target_seq = self.target_seqs[idx]

        # preprocess the input and target sequence
        input_seq = self.preprocessing(input_seq)
        target_seq = self.preprocessing(target_seq)

        # make the input and target sequence into a integer tensor
        input_seq = torch.tensor(input_seq, dtype=torch.long)
        target_seq = torch.tensor(target_seq, dtype=torch.long)

        # create the positional encoding
        input_len = input_seq.shape[0]
        target_len = target_seq.shape[0]
        input_pos = self.positional_encoding(input_len, self.emb_dim)
        target_pos = self.positional_encoding(target_len, self.emb_dim)

        return input_seq, input_pos, target_seq, target_pos


def create_dataloader(
    train_data_path,
    val_data_path,
    test_data_path,
    batch_size,
    emb_dim,
    convert_str_to_tokens=convert_str_to_tokens,
    special_tokens=SPECIAL_TOKENS,
    positional_encoding=positional_encoding_sinusoidal,
):
    """
    create the dataloader for training, validation and testing
    input:
    - train_data_path(str): the path of the training data
    - val_data_path(str): the path of the validation data
    - test_data_path(str): the path of the testing data
    - batch_size(int): the batch size
    - convert_str_to_tokens(dict): a dictionary of tokens
    - special_tokens(list): a list of special tokens
    - emb_dim(int): dimension of the embedding
    - positional_encoding(function): a function of positional encoding

    output:
    - train_dataloader(DataLoader): the dataloader for training
    - val_dataloader(DataLoader): the dataloader for validation
    - test_dataloader(DataLoader): the dataloader for testing
    """
    # load the data
    train_data = json.load(open(train_data_path, "r"))
    val_data = json.load(open(val_data_path, "r"))
    test_data = json.load(open(test_data_path, "r"))

    # get the input and target sequence
    x_train, y_train = train_data["inp_expression"], train_data["out_expression"]
    x_val, y_val = val_data["inp_expression"], val_data["out_expression"]
    x_test, y_test = test_data["inp_expression"], test_data["out_expression"]

    # create the dataset
    train_dataset = AddSubDataset(
        x_train,
        y_train,
        convert_str_to_tokens,
        special_tokens,
        emb_dim,
        positional_encoding,
    )
    val_dataset = AddSubDataset(
        x_val,
        y_val,
        convert_str_to_tokens,
        special_tokens,
        emb_dim,
        positional_encoding,
    )
    test_dataset = AddSubDataset(
        x_test,
        y_test,
        convert_str_to_tokens,
        special_tokens,
        emb_dim,
        positional_encoding,
    )

    # create the dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader
