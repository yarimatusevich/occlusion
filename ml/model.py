import torch
from torch import optim
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import coremltools as ct

'''
Overfitting: training loss going down almost zero, val loss not decreasing (Overfitting)
- Solution: added regularization dropout 0.5 now underfitting val loss low training loss higher
- Solution: dropout = 0.25
'''

inference_transform = transforms.Compose([
        transforms.ToTensor()
])

BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 50

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # images are rgb, so in_channels=3
        # conv layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # pooling, relu, fully-connected layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=16*16*128, out_features=2)
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        # pool into 64x64x32
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        # pool into 32x32x64
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        # pool into 16x16x128
        x = self.relu(self.conv3(x))
        x = self.pool(x)

        # flatten and pass to fully-connected layer
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)

        return x

def load_data(filepath: str, batchsize: int):
    # todo: might add some normalization here
    # for now just converts to tensors

    dataset = datasets.ImageFolder(root=filepath, transform=inference_transform)
    generator = torch.default_generator.manual_seed(42)

    # calculating split sizes from data
    train_size = int(0.8 * len(dataset))
    validation_size = len(dataset) - train_size

    # splitting datasets
    train_data, validation_data = random_split(
        dataset=dataset, lengths=[train_size, validation_size], generator=generator
    )

    # creating loaders for mini-batch gradient descent
    train_loader = DataLoader(dataset=train_data, batch_size=batchsize, shuffle=True)
    validation_loader = DataLoader(dataset=validation_data, batch_size=batchsize, shuffle=True)

    return train_loader, validation_loader

def train(model, num_epochs, train_loader, validation_loader):
    best_val_loss = float('inf')
    epochs_with_no_improvement = 0
    patience = 5

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for data, targets in train_loader:
            # putting imgs and labels to device
            data = data.to(device)
            targets = targets.to(device)

            # clear previous gradient
            optimizer.zero_grad()

            scores = model(data)
            loss = criterion(scores, targets)

            # summing loss for epoch
            train_loss += loss.item()

            # compute gradient and update weights
            loss.backward()
            optimizer.step()

        # computing avg training loss for current epoch
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, targets in validation_loader:
                data = data.to(device)
                targets = targets.to(device)
                scores = model(data)
                val_loss += criterion(scores, targets).item()

        # computing avg loss for current epoch
        val_loss /= len(validation_loader)

        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_with_no_improvement = 0
            torch.save(model.state_dict(), 'model.pth')
        else:
            epochs_with_no_improvement += 1

        # early stop
        if epochs_with_no_improvement == patience:
            print(f'Early Stopping at epoch {epoch + 1}')
            break

        # printing training and validation loss
        print(f'Epoch {epoch + 1} - Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}')

def save_model_to_coreml_format():
    model = CNN()
    state_dict = torch.load('model.pth', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    trace_input = torch.rand(1, 3, 128, 128)
    traced_model = torch.jit.trace(model, trace_input)

    coreml_model = ct.convert(
        traced_model,
        convert_to='ml_program',
        inputs=[ct.TensorType(shape=trace_input.shape)]
    )

    coreml_model.save('occlusion_model.mlpackage')

if __name__ == '__main__':
    device = 'mps' if torch.mps.is_available() else 'cpu'
    model = CNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, validation_loader = load_data(filepath='./data', batchsize=BATCH_SIZE)
    train(model=model, num_epochs=NUM_EPOCHS, train_loader=train_loader, validation_loader=validation_loader)