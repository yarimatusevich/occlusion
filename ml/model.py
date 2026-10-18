import torch.nn as nn

class OcclusionModel(nn.Module):
    def __init__(self):
        super(OcclusionModel, self).__init__()

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