"""
Code for training.
"""

import argparse
import os

import bitsandbytes as bnb
import torch
from torch.nn import MSELoss
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from audioenhancer.model.audio_ae.model import model_xtransformer as model
from einops import rearrange

from audioenhancer.constants import (
    BATCH_SIZE,
    EPOCH,
    GRADIENT_ACCUMULATION_STEPS,
    INPUT_FREQ,
    LOGGING_STEPS,
    MAX_AUDIO_LENGTH,
    OUTPUT_FREQ,
    SAVE_STEPS,
)
from audioenhancer.dataset.loader import SynthDataset

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_dir",
    type=str,
    default="../media/works/dataset",
    required=False,
    help="The directory containing the dataset",
)

parser.add_argument(
    "--model_path",
    type=str,
    required=False,
    default="data/model/",
    help="The path to save the model",
)

parser.add_argument(
    "--mono",
    action="store_true",
    help="Use mono audio",
)

args = parser.parse_args()

dtype = torch.float32

# Load the dataset
dataset = SynthDataset(
    args.dataset_dir,
    max_duration=MAX_AUDIO_LENGTH,
    mono=args.mono,
    input_freq=INPUT_FREQ,
    output_freq=OUTPUT_FREQ,
)
writer = SummaryWriter()

# mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
#     sample_rate=16000,
#     n_mels=128,
#     n_fft=1024,
#     normalized=True,
# )
#
#
# def mel_loss(logits, target):
#     """
#     Compute the MSE loss between the mel spectrogram of the logits and the target.
#     """
#     logits = logits.float()
#     target = target.float()
#     logits = mel_spectrogram_transform(logits)
#     target = mel_spectrogram_transform(target)
#     return MSELoss()(logits, target) / 10

loss_fn = [MSELoss()]
disc_loss_fn = MSELoss()

# split test and train
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size]
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")
model.to(device, dtype=dtype)

# discriminator.to(device, dtype=dtype)
# mel_spectrogram_transform.to(device)

# add both models to optimizer
optimizer = torch.optim.AdamW(
    params=list(model.parameters()),
    lr=1e-4,
    betas=(0.95, 0.999),
    eps=1e-6,
    weight_decay=1e-3,
)

# disc_optimizer = bnb.optim.AdamW8bit(
#     [
#         {"params": discriminator.parameters()},
#     ],
#     lr=8e-5,
#     weight_decay=5e-5,
# )

scheduler = lr_scheduler.LinearLR(
    optimizer,
    start_factor=1,
    end_factor=1e-6,
    total_iters=train_size * EPOCH // (GRADIENT_ACCUMULATION_STEPS * BATCH_SIZE),
)

# disc_scheduler = lr_scheduler.LinearLR(
#     disc_optimizer,
#     start_factor=1,
#     end_factor=1e-6,
#     total_iters=train_size * EPOCH // (GRADIENT_ACCUMULATION_STEPS * BATCH_SIZE),
# )


# print number of parameters
print(f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6}M")


step = 0
logging_loss = 0
# logging_desc_loss = 0
for epoch in range(EPOCH):
    model.train()
    for batch in train_loader:
        step += 1

        x = batch[0].to(
            device, dtype=dtype
        )  # [8, 2, 9, 1152] so [batch, channel, codebook, time]
        y = batch[1].to(device, dtype=dtype)

        # rearrange x and y
        x = rearrange(x, "b c d t -> b t (c d)")

        y_hat = model(x, mask=None)

        y_hat = rearrange(y_hat, "b t (c d) -> b c d t", c=2, d=1024)

        loss = sum([loss_fn[i](y_hat, y) for i in range(len(loss_fn))])
        loss.backward()

        # batch_disc = torch.cat([y, y_hat], dim=0)
        # disc_pred = discriminator(batch_disc)
        # disc_pred = torch.sigmoid(disc_pred).squeeze()
        # labels = [0] * y.shape[0] + [1] * y.shape[0]
        # disc_loss = disc_loss_fn(
        #     disc_pred, torch.Tensor(labels).to(device, dtype=dtype)
        # )
        # logging_desc_loss += disc_loss.detach().cpu().float().numpy()

        logging_loss += loss.detach().cpu().float().numpy()
        # loss += disc_pred[: -y.shape[0]].mean().squeeze()

        if (step % GRADIENT_ACCUMULATION_STEPS) == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        if (step % LOGGING_STEPS) == 0:
            writer.add_scalar("Loss", logging_loss / LOGGING_STEPS, step)
            # writer.add_scalar("Desc Loss", logging_desc_loss / LOGGING_STEPS, step)
            writer.add_scalar("LR", scheduler.get_last_lr()[0], step)
            print(
                f"EPOCH {round(step/len(train_loader), 2)} "
                f"({round((step % len(train_loader)) / (len(train_loader) * EPOCH) * 100, 2)}%) - "
                f"STEP {step}:"
                f" Loss {round(logging_loss / LOGGING_STEPS, 4)}"
                # f" Desc Loss {round(logging_desc_loss / LOGGING_STEPS, 4)}"
                f" LR {round(scheduler.get_last_lr()[0], 8)}"
            )
            logging_loss = 0
            logging_desc_loss = 0

        if step % SAVE_STEPS == 0:
            if not os.path.exists(args.model_path):
                os.makedirs(args.model_path)
            torch.save(model.state_dict(), args.model_path + f"model_{step}.pt")
            # torch.save(discriminator.state_dict(), args.model_path + f"disc_{step}.pt")

    model.eval()

    loss_test = 0
    # loss_desc_test = 0
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device, dtype=dtype)
            y = batch[1].to(device, dtype=dtype)

            loss = model(x=x, y=y)

            loss_test += loss.item()
            # loss_desc_test += disc_loss.item()

    print(
        f"Avg test Loss: {loss_test / len(test_loader)}"
        # f" Desc Loss {loss_desc_test / len(test_loader)}"
    )

print("Training done")
torch.save(model.state_dict(), args.model_path + "model.pt")
# torch.save(discriminator.state_dict(), args.model_path + "disc.pt")

writer.flush()
writer.close()
