import torch
import torch.nn as nn

from model.unet_layers import DoubleConv


class UNetModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetModel, self).__init__()

        # Encoder part
        self.dc1 = DoubleConv(in_channels, 64)
        self.dc2 = DoubleConv(64, 128)
        self.dc3 = DoubleConv(128, 256)
        self.dc4 = DoubleConv(256, 512)
        self.dc5 = DoubleConv(512, 1024)
        self.max_pool = nn.MaxPool2d(2)

        # Decoder part
        self.tc4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.tc3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.tc2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.tc1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.dc6 = DoubleConv(1024, 512)
        self.dc7 = DoubleConv(512, 256)
        self.dc8 = DoubleConv(256, 128)
        self.dc9 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)


    def forward(self, x):
        skip_connections = []

        # Encoder passthrough / forward-pass
        x = self.dc1(x)
        skip_connections.append(x)
        x = self.max_pool(x)

        x = self.dc2(x)
        skip_connections.append(x)
        x = self.max_pool(x)

        x = self.dc3(x)
        skip_connections.append(x)
        x = self.max_pool(x)

        x = self.dc4(x)
        skip_connections.append(x)
        x = self.max_pool(x)

        x = self.dc5(x)

        # Decoder
        x = self.tc4(x)
        x = torch.cat((x, skip_connections[3]), dim=1)
        x = self.dc6(x)

        x = self.tc3(x)
        x = torch.cat((x, skip_connections[2]), dim=1)
        x = self.dc7(x)

        x = self.tc2(x)
        x = torch.cat((x, skip_connections[1]), dim=1)
        x = self.dc8(x)

        x = self.tc1(x)
        x = torch.cat((x, skip_connections[0]), dim=1)
        x = self.dc9(x)

        return self.final_conv(x)



if __name__ == '__main__':
    model = UNetModel(3, 1)

    a = torch.randn((4, 3, 512, 512))
    b = model(a)

    print(b.shape)
