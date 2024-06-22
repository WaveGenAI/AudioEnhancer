"""
Code for training.
"""

import auraloss
import torch
from torch.optim import lr_scheduler

from constants import BATCH_SIZE, EPOCH, SAMPLING_RATE
from model.dataset import SynthDataset
from model.soundstream import SoundStream

# Load the dataset
dataset = SynthDataset("/media/works/dataset/", mono=False)

loss_fn = [auraloss.time.LogCoshLoss(), auraloss.freq.STFTLoss()]

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
    D=256,
    C=58,
    strides=(2, 4, 5, 5),
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = lr_scheduler.LinearLR(
    optimizer, start_factor=1.0, end_factor=0.5, total_iters=30
)

step = 0
for epoch in range(EPOCH):
    model.train()
    for batch in train_loader:
        step += 1

        optimizer.zero_grad()

        x = batch[0].to(device)
        y = batch[1].to(device)

        y_hat = model(y)

        loss = sum([loss(y_hat, y) for loss in loss_fn])
        loss.backward()

        optimizer.step()

        print(
            f"EPOCH {epoch} ({round(step / len(train_loader) * 100, 3)}%) - STEP {step}: Loss {round(loss.item(), 3)}"
        )

        if step % 100 == 0:
            torch.save(model.state_dict(), "data/model.pth")

    scheduler.step()

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device)
            y = batch[1].to(device)

            y_hat = model(y)
            loss = sum([loss(y_hat, y) for loss in loss_fn])

            print("Test loss:", loss.item())

    step = 0
