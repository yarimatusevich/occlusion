import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import lightning as L

from data_transforms import transform_training_data

fabric = L.Fabric(accelerator='mps')

BATCH_SIZE = 32
NUM_EPOCHS = 50

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        def block(in_channels: int, out_channels: int):
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU()
            )

        self.conv_layers = nn.Sequential(
            # 128 x 128 x 3 input
            block(in_channels=3, out_channels=32),
            # 64 x 64 x 32
            block(in_channels=32, out_channels=64),
            # 32 x 32 x 64
            block(in_channels=64, out_channels=128)
        )

        # 16 x 16 x 128 matrix needs to be flattened
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=16 * 16 * 128, out_features=2)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x


def create_train_val_data_loaders(filepath: str, batch_size: int):
    dataset = transform_training_data(filepath)

    train_size = int(0.8 * len(dataset))
    validation_size = len(dataset) - train_size

    train_data, validation_data = random_split(
        dataset, lengths=[train_size, validation_size]
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_data, batch_size=batch_size)

    return train_loader, val_loader


def train(model, num_epochs, dataloaders: tuple[DataLoader, DataLoader]):
    optimizer = torch.optim.AdamW(model.parameters())
    criterion = nn.CrossEntropyLoss()
    model, optimizer = fabric.setup(model, optimizer)
    train_loader, val_loader = fabric.setup_dataloaders(*dataloaders)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            features, targets = batch
            optimizer.zero_grad()

            logits = model(features)
            loss = criterion(logits, targets)
            train_loss += loss

            fabric.backward(loss)
            optimizer.step()

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for features, targets in val_loader:
                logits = model(features)
                val_loss += criterion(logits, targets).item()
                correct += (logits.argmax(dim=1) == targets).sum().item()
                total += targets.size(0)

        print(f"epoch {epoch + 1}: train_loss={train_loss / len(train_loader):.4f} "
              f"val_loss={val_loss / len(val_loader):.4f} val_acc={correct / total:.3f}")


if __name__ == '__main__':
    fabric.seed_everything(42)

    model = Model()
    dataloaders = create_train_val_data_loaders('../data/', BATCH_SIZE)
    train(model, NUM_EPOCHS, dataloaders)