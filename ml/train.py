import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import lightning as L

from data_transforms import transform_training_data
from ml.model import OcclusionModel

fabric = L.Fabric(accelerator='mps')

BATCH_SIZE = 32
NUM_EPOCHS = 50

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

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            features, targets = batch
            optimizer.zero_grad()

            logits = model(features)
            loss = criterion(logits, targets)
            train_loss += loss.item()

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

        val_loss /= len(val_loader)

        print(f"epoch {epoch + 1}: train_loss={train_loss / len(train_loader):.4f} "
              f"val_loss={val_loss:.4f} val_acc={correct / total:.3f}")

        state = {
            'model': model,
            'optimizer': optimizer,
        }
        fabric.save(CHECKPOINT_DIR / 'last.ckpt', state)


if __name__ == '__main__':
    fabric.seed_everything(42)

    model = OcclusionModel()
    dataloaders = create_train_val_data_loaders('../data/', BATCH_SIZE)
    train(model, NUM_EPOCHS, dataloaders)