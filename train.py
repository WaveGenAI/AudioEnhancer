"""
Code for training.
"""

import auraloss
import torch

from constants import BATCH_SIZE, EPOCH, SAMPLING_RATE
from model.dataset import SynthDataset
from model.soundstream import SoundStream

# Load the dataset
dataset = SynthDataset("/media/works/dataset/")

loss_fn = auraloss.time.LogCoshLoss()

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
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

step = 0
for epoch in range(EPOCH):
    model.train()
    for batch in train_loader:
        step += 1

        optimizer.zero_grad()

        x = batch[0].to(device)
        y = model(x)

        loss = loss_fn(y, x)
        loss.backward()
        optimizer.step()

        print(
            f"EPOCH {epoch}, STEP {step}: Loss {loss.item()}, % of epoch {step / len(train_loader) * 100}"
        )
        
        if step % 100 == 0:
            torch.save(model.state_dict(), "data/model.pth")

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device)
            y = model(x)
            loss = loss_fn(y, x)
            print("Test loss:", loss.item())

