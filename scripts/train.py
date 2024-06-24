"""
Code for training.
"""

import argparse
import os
from pathlib import Path

import auraloss
import torch
from torch.nn import MSELoss
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from audioenhancer.constants import BATCH_SIZE, EPOCH, MAX_AUDIO_LENGTH, LOGGING_STEPS, GRADIENT_ACCUMULATION_STEPS, \
    SAVE_STEPS
from audioenhancer.dataset.loader import SynthDataset
from audioenhancer.model.soundstream import SoundStream
import bitsandbytes as bnb

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
    default="data/model.pth",
    help="The path to save the model",
)

parser.add_argument(
    "--mono",
    action="store_true",
    help="Use mono audio",
)

args = parser.parse_args()


# Load the dataset
dataset = SynthDataset(args.dataset_dir, max_duration=MAX_AUDIO_LENGTH, mono=args.mono)
writer = SummaryWriter()

loss_fn = [auraloss.time.LogCoshLoss()]

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

model = SoundStream(
    D=512,
    C=64,
    strides=(2, 4, 4, 5),
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")
model.to(device)

optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=5e-4)
scheduler = lr_scheduler.LinearLR(
    optimizer,
    start_factor=1,
    end_factor=1e-6,
    total_iters=train_size * EPOCH // (GRADIENT_ACCUMULATION_STEPS * BATCH_SIZE)
)

# print number of parameters

print(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1e6}M")

step = 0
logging_loss = 0
for epoch in range(EPOCH):
    model.train()
    for batch in train_loader:
        step += 1

        x = batch[0].to(device)
        y = batch[1].to(device)

        y_hat = model(y)

        loss = sum([loss(y_hat, y) for loss in loss_fn])
        logging_loss += loss.detach().cpu().numpy()
        loss.backward()
        if (step % LOGGING_STEPS) == 0:
            writer.add_scalar("Loss", logging_loss/LOGGING_STEPS, step)
            writer.add_scalar("LR", scheduler.get_last_lr()[0], step)
            print(
                f"EPOCH {epoch} ({round((step % len(train_loader)) / len(train_loader) * 100, 3)}%) - "
                f"STEP {step}:"
                f" Loss {round(logging_loss/LOGGING_STEPS, 4)}"
                f" LR {round(scheduler.get_last_lr()[0], 7)}"
            )
            logging_loss = 0

        if (step % GRADIENT_ACCUMULATION_STEPS) == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)

            for name, param in model.named_parameters():
                if param.grad is not None:
                    param.grad /= GRADIENT_ACCUMULATION_STEPS

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()



        if step % SAVE_STEPS == 0:
            if not os.path.exists(args.model_path):
                Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)

            torch.save(model.state_dict(), args.model_path)

    model.eval()

    loss_test = 0
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device)
            y = batch[1].to(device)

            y_hat = model(y)
            loss = sum([loss(y_hat, y) for loss in loss_fn])

            loss_test += loss.item()

    print(f"Avg test Loss: {loss_test / len(test_loader)}")

torch.save(model.state_dict(), args.model_path)

writer.flush()
writer.close()
