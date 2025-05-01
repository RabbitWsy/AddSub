import torch
import argparse
import os
from tqdm import tqdm
import tomllib

from models.transformer import Transformer
from models.losses import LabelSmoothingLoss, CrossEntropyLoss
from data.data_loader import create_dataloader


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True, help="config file path")
    parser.add_argument(
        "--test", action="store_true", default=False, help="test the model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ckpts",
        help="output directory for saving the model",
    )

    return parser.parse_args()


def train(
    model,
    train_dataloader,
    val_dataloader,
    loss_fn,
    epochs,
    warmup_steps=10000,
    warmup_lr=1e-6,
    lr=1e-4,
    device=torch.device("cpu"),
):
    """
    train the model.
    input:
    - model(nn.Module): the model to be trained
    - train_dataloader(DataLoader): the dataloader for training
    - val_dataloader(DataLoader): the dataloader for validation
    - loss_fn(nn.Module): the loss function
    - epochs(int): the number of epochs
    - batch_size(int): the batch size
    - warmup_steps(int): the number of warmup steps
    - warmup_lr(float): the learning rate for warmup
    - lr(float): the learning rate
    - device(torch.device): the device to be used

    output:
    - model(nn.Module): the trained model
    """
    print("Training...")
    if warmup_steps == 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=warmup_lr, weight_decay=1e-4
        )

    iterations = 0
    for epoch in tqdm(range(epochs), desc="Total Epoch"):
        epoch_loss = []
        epoch_correct = 0
        epoch_total = 0
        model.train()
        process_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}", leave=False)
        for batch in process_bar:
            # get the input and target sequence
            input_seq, input_pos, target_seq, target_pos = batch

            # move the data to the device
            model = model.to(device)
            input_seq = input_seq.to(device)
            input_pos = input_pos.to(device)
            target_seq = target_seq.to(device)
            target_pos = target_pos.to(device)

            # prepare the ground truth
            # shape: (batch_size * (seq_len - 1),)
            gnd_truth = target_seq[:, 1:].contiguous().view(-1)

            # get the prediction
            # shape: (batch_size * (seq_len - 1), vocab_size)
            prediction = model(input_seq, input_pos, target_seq, target_pos)

            # training correctness
            pred_max = prediction.argmax(dim=-1)
            epoch_correct += (pred_max == gnd_truth).sum().item()
            epoch_total += gnd_truth.numel()

            # calculate the loss
            loss = loss_fn(prediction, gnd_truth)
            epoch_loss.append(loss.item())

            # adjust the learning rate
            if warmup_steps > 0:
                iterations += 1
                if iterations < warmup_steps:
                    lr = warmup_lr + (lr - warmup_lr) * iterations / warmup_steps
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            process_bar.set_postfix(
                loss=f"{loss.item():.4f}", accuracy=f"{epoch_correct / epoch_total:.4f}"
            )

        # calculate the average loss and accuracy of training
        train_loss = sum(epoch_loss) / len(epoch_loss)
        # train_loss = train_loss / batch_size
        train_acc = epoch_correct / epoch_total

        # validation
        val_loss, val_acc = validate(model, val_dataloader, loss_fn, device)

        # print the result
        # print(
        #     f"Epoch {epoch+1}/{epochs}, train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, val loss: {val_loss:.4f}, val acc: {val_acc:.4f}"
        # )
        tqdm.write(
            f"Epoch {epoch+1}/{epochs}, train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, val loss: {val_loss:.4f}, val acc: {val_acc:.4f}"
        )

    return model


def validate(model, val_dataloader, loss_fn, device):
    """
    validate the model.
    input:
    - model(nn.Module): the model to be validated
    - val_dataloader(DataLoader): the dataloader for validation
    - loss_fn(nn.Module): the loss function
    - device(torch.device): the device to be used
    output:
    - val_loss(float): the average loss of validation
    - val_acc(float): the average accuracy of validation
    """
    model.eval()
    with torch.no_grad():
        val_loss = []
        val_correct = 0
        val_total = 0
        for batch in val_dataloader:
            # get the input and target sequence
            input_seq, input_pos, target_seq, target_pos = batch

            # move the data to the device
            model = model.to(device)
            input_seq = input_seq.to(device)
            input_pos = input_pos.to(device)
            target_seq = target_seq.to(device)
            target_pos = target_pos.to(device)

            # prepare the ground truth
            # shape: (batch_size * (seq_len - 1),)
            gnd_truth = target_seq[:, 1:].contiguous().view(-1)

            # get the prediction
            # shape: (batch_size * (seq_len - 1), vocab_size)
            prediction = model(input_seq, input_pos, target_seq, target_pos)

            # validation correctness
            pred_max = prediction.argmax(dim=-1)
            val_correct += (pred_max == gnd_truth).sum().item()
            val_total += gnd_truth.numel()

            # calculate the loss
            loss = loss_fn(prediction, gnd_truth)
            val_loss.append(loss.item())

        # calculate the average loss and accuracy of validation
        val_loss = sum(val_loss) / len(val_loss)
        val_acc = val_correct / val_total
        return val_loss, val_acc


if __name__ == "__main__":
    args = parse_args()

    # create the output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

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

    train_data_path = config["data"]["train_data_path"]
    val_data_path = config["data"]["val_data_path"]
    test_data_path = config["data"]["test_data_path"]

    if config["train"]["loss_fn"] == "LabelSmoothingLoss":
        loss_fn = LabelSmoothingLoss
    elif config["train"]["loss_fn"] == "CrossEntropyLoss":
        loss_fn = CrossEntropyLoss

    epochs = config["train"]["epochs"]
    batch_size = config["train"]["batch_size"]
    warmup_steps = config["train"]["warmup_steps"]
    warmup_lr = config["train"]["warmup_lr"]
    lr = config["train"]["lr"]
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

    # create the dataloader
    train_dataloader, val_dataloader, test_dataloader = create_dataloader(
        train_data_path,
        val_data_path,
        test_data_path,
        batch_size,
        emb_dim,
    )

    # train the model
    if args.test == False:
        # train from the pretrained model
        if config["model"]["pretrained_model"] != "":
            model.load_state_dict(
                torch.load(
                    os.path.join(config["model"]["pretrained_model"]), weights_only=True
                )
            )
            print("Load pretrained model")
        # train the model
        train_model = train(
            model,
            train_dataloader,
            val_dataloader,
            loss_fn,
            epochs,
            warmup_steps,
            warmup_lr,
            lr,
            device,
        )

        # save the model
        torch.save(
            train_model.state_dict(),
            os.path.join(args.output_dir, "addshub_transformer.pt"),
        )
    else:
        # load the model
        model.load_state_dict(
            torch.load(os.path.join(config["model"]["test_model"]), weights_only=True)
        )
        # test the model
        test_loss, test_acc = validate(model, train_dataloader, loss_fn, device)
        print(f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")
