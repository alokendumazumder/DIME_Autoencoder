import torch
import torch.nn as nn
import torch.nn.functional as F

class L2Norm(nn.Module):
    def forward(self, x):
        return x / x.norm(p=2, dim=1, keepdim=True)
    
class Encoder(nn.Module):
    def __init__(self, input_shape, num_filter=128, bottleneck_size=64, include_batch_norm=True):
        super(Encoder, self).__init__()
        
        self.conv1 = nn.Conv2d(input_shape[2], num_filter, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_filter, num_filter*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(num_filter*2, num_filter*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(num_filter*4, num_filter*8, kernel_size=4, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(num_filter*8 * 2 * 2, bottleneck_size)
        self.include_batch_norm = include_batch_norm
        self.bn1 = nn.BatchNorm2d(num_filter) if include_batch_norm else None
        self.bn2 = nn.BatchNorm2d(num_filter*2) if include_batch_norm else None
        self.bn3 = nn.BatchNorm2d(num_filter*4) if include_batch_norm else None
        self.l2 = L2Norm()
        
    def forward(self, x):
        x = 2.0 * x - 1.0
        x = (self.conv1(x))
        if self.include_batch_norm:
            x = self.bn1(x)
        x = self.relu(x)
        x = (self.conv2(x))
        if self.include_batch_norm:
            x = self.bn2(x)
        x = self.relu(x)
        x = (self.conv3(x))
        if self.include_batch_norm:
            x = self.bn3(x)
        x = self.relu(x)
        x = (self.conv4(x))
        # print(x.shape)
        x = self.flatten(x)
        z = self.fc(x)
        z = self.l2(z)
        return z

class Decoder(nn.Module):
    def __init__(self, input_shape, bottleneck_size=64, num_filter=128,include_batch_norm=True):
        super(Decoder, self).__init__()
        
        self.fc = nn.Linear(bottleneck_size, 8 * 8 * 1024)
        self.conv1 = nn.ConvTranspose2d(1024, num_filter*4, kernel_size=4, stride=2, padding=1,output_padding=1)
        self.conv2 = nn.ConvTranspose2d(num_filter*4, num_filter*2, kernel_size=4, stride=2, padding=1,output_padding=1)
        self.conv3 = nn.ConvTranspose2d(num_filter*2, input_shape[2], kernel_size=4, padding=1)
        self.relu = nn.ReLU()
        self.include_batch_norm = include_batch_norm
        self.bn1 = nn.BatchNorm2d(num_filter*4) if include_batch_norm else None
        self.bn2 = nn.BatchNorm2d(num_filter*2) if include_batch_norm else None
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 1024, 8, 8)
        x = (self.conv1(x))
        # print(x.shape)
        if self.include_batch_norm:
            x = self.bn1(x)
        x = self.relu(x)
        x = (self.conv2(x))
        # print(x.shape)
        if self.include_batch_norm:
            x = self.bn2(x)
        x = self.relu(x)
        x = torch.sigmoid(self.conv3(x))
        # print(x.shape)
        return x

class AE(nn.Module):
    def __init__(self,args):
        super(AE, self).__init__()
        
    
    def encode(self, x):
        z = self.encoder(x)
        # z =  F.normalize(z,p=2,dim=1)
        # print(z.norm(p=2,dim=1))
        return z

    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        # print(x.shape)
        # x_org = x
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        
        return loss,z

