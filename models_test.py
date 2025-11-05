import torch.nn as nn
import torch.nn.functional as F
import torch


class L2Norm(nn.Module):
    def forward(self, x):
        return x / x.norm(p=2, dim=1, keepdim=True)
    
# for downstream classification
class Head(nn.Module):
    def __init__(self, code_dim, out_channels):
        super(Head, self).__init__()
        self.code_dim = code_dim
        self.out_channels = out_channels
        self.hidden = nn.ModuleList()
        for k in range(2):
            self.hidden.append(nn.Linear(code_dim, code_dim, bias=True))
            self.hidden.append(nn.ReLU(True))
            self.hidden.append(nn.Dropout(0.5))
        self.hidden.append(nn.Linear(code_dim, out_channels, bias=False))

    def forward(self, z):
        for l in self.hidden:
            z = l(z)
        return z

    
class MLP(nn.Module):
    def __init__(self, code_dim, layers):
        super(MLP, self).__init__()
        self.code_dim = code_dim
        self.layers = layers
        self.hidden = nn.ModuleList()
        for k in range(layers):
            linear_layer = nn.Linear(code_dim, code_dim, bias=False)
            self.hidden.append(linear_layer)

    def forward(self, z):
        for l in self.hidden:
            z = l(z)
        return z


class Shape_Decoder(nn.Module):
    def __init__(self, code_dim, vae=False):
        super(Shape_Decoder, self).__init__()
        self.code_dim = code_dim
        self.dcnn = nn.Sequential(
            nn.ConvTranspose2d(code_dim, 256, 2, 1, 0),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid() if vae else nn.Tanh()
        )

    def forward(self, z):
        out = self.dcnn(z.view(z.size(0), self.code_dim, 1, 1))
        return out


class Shape_Encoder(nn.Module):
    def __init__(self, code_dim):
        super(Shape_Encoder, self).__init__()
        self.code_dim = code_dim
        self.dcnn = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),    # 32x32
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),      # 16x16
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),      # 8x8
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),      # 4x4
            nn.ReLU(True),
            nn.Conv2d(256, code_dim, 2, 1, 0),    # 1x1
        )

    def forward(self, z):
        return self.dcnn(z).view(z.size(0), self.code_dim)
# class MNIST_Encoder(nn.Module):
#     def __init__(self, code_dim):
#         super(MNIST_Encoder, self).__init__()
#         self.code_dim = code_dim
#         self.dcnn = nn.Sequential(
#             nn.Conv2d(1, 128, 4, 2,padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),
#             nn.Conv2d(128, 128*2, 4, 2,padding=1),
#             nn.BatchNorm2d(128*2),
#             nn.ReLU(True),
#             nn.Conv2d(128*2, 128*4, 4, 2,padding=1),
#             nn.BatchNorm2d(128*4),
#             nn.ReLU(True),
#             nn.Conv2d(128*4, 128*8, 4, 2,padding=1),
#             nn.BatchNorm2d(128*8),
#             nn.ReLU(True),
#         )
#         self.fc = nn.Linear(1024, self.code_dim)
#         self.l2 = L2Norm()

#     def forward(self, z):
#         # print(z.shape)
#         z = self.dcnn(z)
#         # print(z.shape)
#         z = z.view(z.size(0), 1024)
#         z = self.fc(z)
#         # z = self.l2(z)
#         return z


# class MNIST_Decoder(nn.Module):
#     def __init__(self, code_dim):
#         super(MNIST_Decoder, self).__init__()
#         self.code_dim = code_dim
#         self.fc = nn.Linear(self.code_dim, 16384)
#         self.dcnn = nn.Sequential(
#             nn.ReLU(True),
#             nn.ConvTranspose2d(1024, 512, 3, 2,padding=1),
#             nn.BatchNorm2d(128*4),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(512, 256, 3, 2,padding=1,output_padding=1),
#             nn.BatchNorm2d(128*2),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(256, 1, 3, 2,padding=1,output_padding=1),
#             nn.Sigmoid()
#         )

#     def forward(self, z):
#         z = self.fc(z)
#         z = z.view(z.size(0), 1024, 4, 4)
#         # print(z.shape)
#         z = self.dcnn(z)
#         return z
class MNIST_Encoder(nn.Module):
    def __init__(self, code_dim):
        super(MNIST_Encoder, self).__init__()
        self.code_dim = code_dim
        self.dcnn = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2,1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2,1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2,1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2,1),
            nn.ReLU(True),
        )
        self.fc = nn.Linear(256*2*2, self.code_dim)
        self.l2 = L2Norm()

    def forward(self, z):
        z = self.dcnn(z)
        z = z.view(z.size(0), 256*2*2)
        z = self.fc(z)
        # z = self.l2(z)
        return z


class MNIST_Decoder(nn.Module):
    def __init__(self, code_dim):
        super(MNIST_Decoder, self).__init__()
        self.code_dim = code_dim
        self.fc = nn.Linear(self.code_dim, 8*8*128)
        self.dcnn = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 1, 1, 0),
            nn.Tanh()
        )

    def forward(self, z):
        z = self.fc(z)
        z = z.view(z.size(0), 128, 8, 8)
        z = self.dcnn(z)
        return z
# class CelebA_Encoder(nn.Module):
#     def __init__(self, input_shape, num_filter=128, bottleneck_size=64, include_batch_norm=True):
#         super(CelebA_Encoder, self).__init__()

#         # Layers
#         self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=num_filter, kernel_size=4, stride=2, padding=1)
#         self.conv2 = nn.Conv2d(num_filter, num_filter*2, 4, 2, 1)
#         self.conv3 = nn.Conv2d(num_filter*2, num_filter*4, 4, 2, 1)
#         self.conv4 = nn.Conv2d(num_filter*4, num_filter*8, 4, 2, 1)
#         self.batch_norm1 = nn.BatchNorm2d(num_filter)
#         self.batch_norm2 = nn.BatchNorm2d(num_filter*2)
#         self.batch_norm3 = nn.BatchNorm2d(num_filter*4)
#         self.batch_norm4 = nn.BatchNorm2d(num_filter*8)
#         self.flatten = nn.Flatten()
#         self.fc_z = nn.Linear(1024, bottleneck_size)
#         # self.activation = nn.Sigmoid()
#         self.l2 = L2Norm()

#     def forward(self, x):
#         x = F.relu(self.batch_norm1(self.conv1(x)))
#         # print("Encoder C1",x.shape)
#         x = F.relu(self.batch_norm2(self.conv2(x)))
#         # print("Encoder C2",x.shape)
#         x = F.relu(self.batch_norm3(self.conv3(x)))
#         # print("Encoder C3",x.shape)
#         x = F.relu(self.batch_norm4(self.conv4(x)))
#         # print("Encoder C4",x.shape)
#         x = self.flatten(x)
#         z = self.fc_z(x)
#         # z = self.activation(z)
#         # z = self.l2(z)
#         return z


# class CelebA_Decoder(nn.Module):
#     def __init__(self, output_shape, num_filter=128, bottleneck_size=64):
#         super(CelebA_Decoder, self).__init__()
#         self.fc_d = nn.Linear(bottleneck_size, 8*8*1024)
#         self.reshape = lambda x: x.view(-1, 1024, 8, 8)
        
#         self.convT2 = nn.ConvTranspose2d(num_filter*8, num_filter*4, 5,2, padding=2)
#         self.convT3 = nn.ConvTranspose2d(num_filter*4, num_filter*2, 5,2, padding=1)
#         self.convT4 = nn.ConvTranspose2d(num_filter*2, num_filter, 5,2, padding=(2,2),output_padding=1)
#         self.convT5 = nn.ConvTranspose2d(num_filter, output_shape[0], 5, padding=1)
#         self.batch_norm2 = nn.BatchNorm2d(num_filter*4)
#         self.batch_norm3 = nn.BatchNorm2d(num_filter*2)
#         self.batch_norm4 = nn.BatchNorm2d(num_filter)

#     def forward(self, x):
#         x = self.reshape(F.relu(self.fc_d(x)))
#         # print("Decoder C1",x.shape)
#         x = F.relu(self.batch_norm2(self.convT2(x)))
#         # print("Decoder C2",x.shape)
#         x = F.relu(self.batch_norm3(self.convT3(x)))
#         # print("Decoder C3",x.shape)
#         x = F.relu(self.batch_norm4(self.convT4(x)))
#         # print("Decoder C4",x.shape)
#         x = torch.sigmoid(self.convT5(x))
#         # print("Decoder C5",x.shape)
#         return x

class CelebA_Encoder(nn.Module):
    def __init__(self, input_shape, num_filter=128, bottleneck_size=64, include_batch_norm=True):
        super(CelebA_Encoder, self).__init__()

        # Layers
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=num_filter, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(num_filter, num_filter*2, 5, 2, 2)
        self.conv3 = nn.Conv2d(num_filter*2, num_filter*4, 5, 2, 2)
        self.conv4 = nn.Conv2d(num_filter*4, num_filter*8, 5, 2, 2)
        self.batch_norm1 = nn.BatchNorm2d(num_filter)
        self.batch_norm2 = nn.BatchNorm2d(num_filter*2)
        self.batch_norm3 = nn.BatchNorm2d(num_filter*4)
        self.batch_norm4 = nn.BatchNorm2d(num_filter*8)
        self.flatten = nn.Flatten()
        self.fc_z = nn.Linear(num_filter* 8 * (input_shape[1] // 16) * (input_shape[2] // 16), bottleneck_size)
        self.l2 = L2Norm()

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.conv1(x)))
        # print("Encoder C1",x.shape)
        x = F.relu(self.batch_norm2(self.conv2(x)))
        # print("Encoder C2",x.shape)
        x = F.relu(self.batch_norm3(self.conv3(x)))
        # print("Encoder C3",x.shape)
        x = F.relu(self.batch_norm4(self.conv4(x)))
        # print("Encoder C4",x.shape)
        x = self.flatten(x)
        z = self.fc_z(x)
        # z = self.l2(z)
        return z


class CelebA_Decoder(nn.Module):
    def __init__(self, output_shape, num_filter=128, bottleneck_size=64):
        super(CelebA_Decoder, self).__init__()
        self.fc_d = nn.Linear(bottleneck_size, 8*8*1024)
        self.reshape = lambda x: x.view(-1, 1024, 8, 8)
        
        self.convT2 = nn.ConvTranspose2d(num_filter*8, num_filter*4, 5,2, padding=(2,2),output_padding=1)
        self.convT3 = nn.ConvTranspose2d(num_filter*4, num_filter*2, 5,2, padding=(2,2),output_padding=1)
        self.convT4 = nn.ConvTranspose2d(num_filter*2, num_filter, 5,2, padding=(2,2),output_padding=1)
        self.convT5 = nn.ConvTranspose2d(num_filter, output_shape[0], 5, padding=(2,2))
        self.batch_norm2 = nn.BatchNorm2d(num_filter*4)
        self.batch_norm3 = nn.BatchNorm2d(num_filter*2)
        self.batch_norm4 = nn.BatchNorm2d(num_filter)

    def forward(self, x):
        x = self.reshape(F.relu(self.fc_d(x)))
        # print("Decoder C1",x.shape)
        x = F.relu(self.batch_norm2(self.convT2(x)))
        # print("Decoder C2",x.shape)
        x = F.relu(self.batch_norm3(self.convT3(x)))
        # print("Decoder C3",x.shape)
        x = F.relu(self.batch_norm4(self.convT4(x)))
        # print("Decoder C4",x.shape)
        x = torch.sigmoid(self.convT5(x))
        # print("Decoder C5",x.shape)
        return x
# class Encoder(nn.Module):
#     def __init__(self, input_shape, num_filter=128, bottleneck_size=64, include_batch_norm=True):
#         super(Encoder, self).__init__()
        
#         self.conv1 = nn.Conv2d(input_shape[2], num_filter, kernel_size=4, stride=2, padding=1)
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv2d(num_filter, num_filter*2, kernel_size=4, stride=2, padding=1)
#         self.conv3 = nn.Conv2d(num_filter*2, num_filter*4, kernel_size=4, stride=2, padding=1)
#         self.conv4 = nn.Conv2d(num_filter*4, num_filter*8, kernel_size=4, stride=2, padding=1)
#         self.flatten = nn.Flatten()
#         self.fc = nn.Linear(num_filter*8 * 2 * 2, bottleneck_size)
#         self.include_batch_norm = include_batch_norm
#         self.bn1 = nn.BatchNorm2d(num_filter) if include_batch_norm else None
#         self.bn2 = nn.BatchNorm2d(num_filter*2) if include_batch_norm else None
#         self.bn3 = nn.BatchNorm2d(num_filter*4) if include_batch_norm else None
#         self.l2 = L2Norm()
        
#     def forward(self, x):
#         x = (self.conv1(x))
#         if self.include_batch_norm:
#             x = self.bn1(x)
#         x = self.relu(x)
#         x = (self.conv2(x))
#         if self.include_batch_norm:
#             x = self.bn2(x)
#         x = self.relu(x)
#         x = (self.conv3(x))
#         if self.include_batch_norm:
#             x = self.bn3(x)
#         x = self.relu(x)
#         x = (self.conv4(x))
#         # print(x.shape)
#         x = self.flatten(x)
#         z = self.fc(x)
#         # z = self.l2(z)
#         return z

# class Decoder(nn.Module):
#     def __init__(self, input_shape, bottleneck_size=64, num_filter=128,include_batch_norm=True):
#         super(Decoder, self).__init__()
        
#         self.fc = nn.Linear(bottleneck_size, 65536)
#         self.conv1 = nn.ConvTranspose2d(1024, num_filter*4, kernel_size=4, stride=2, padding=1)
#         self.conv2 = nn.ConvTranspose2d(num_filter*4, num_filter*2, kernel_size=4, stride=2, padding=1,output_padding=1)
#         self.conv3 = nn.ConvTranspose2d(num_filter*2, input_shape[2], kernel_size=4, padding=2)
#         self.relu = nn.ReLU()
#         self.include_batch_norm = include_batch_norm
#         self.bn1 = nn.BatchNorm2d(num_filter*4) if include_batch_norm else None
#         self.bn2 = nn.BatchNorm2d(num_filter*2) if include_batch_norm else None
        
#     def forward(self, x):
#         x = self.fc(x)
#         x = x.view(-1, 1024, 8, 8)
#         x = (self.conv1(x))
#         # print(x.shape)
#         if self.include_batch_norm:
#             x = self.bn1(x)
#         x = self.relu(x)
#         x = (self.conv2(x))
#         # print(x.shape)
#         if self.include_batch_norm:
#             x = self.bn2(x)
#         x = self.relu(x)
#         x = torch.sigmoid(self.conv3(x))
#         # print(x.shape)
#         return x
    
#Original used to train
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
        # z = self.l2(z)
        return z

class Decoder(nn.Module):
    def __init__(self, input_shape, bottleneck_size=64, num_filter=128,include_batch_norm=True):
        super(Decoder, self).__init__()
        
        self.fc = nn.Linear(bottleneck_size, 8 * 8 * 1024)
        self.conv1 = nn.ConvTranspose2d(1024, num_filter*4, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(num_filter*4, num_filter*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(num_filter*2, input_shape[2], kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.include_batch_norm = include_batch_norm
        self.bn1 = nn.BatchNorm2d(num_filter*4) if include_batch_norm else None
        self.bn2 = nn.BatchNorm2d(num_filter*2) if include_batch_norm else None
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 1024, 8, 8)
        x = (self.conv1(x))
        print(x.shape)
        if self.include_batch_norm:
            x = self.bn1(x)
        x = self.relu(x)
        x = (self.conv2(x))
        print(x.shape)
        if self.include_batch_norm:
            x = self.bn2(x)
        x = self.relu(x)
        x = torch.sigmoid(self.conv3(x))
        # print(x.shape)
        return x

class AE(nn.Module):
    def __init__(self, args):
        super(AE, self).__init__()
        self.args = args
        self.n = args.n

        if args.dataset == "mnist" or args.dataset == "fmnist":
            self.enc = MNIST_Encoder(args.n)
            self.dec = MNIST_Decoder(args.n)
        
        elif args.dataset == "intel" or args.dataset=='svhn' or args.dataset=='celeba':
            self.enc = CelebA_Encoder(input_shape=(3,64,64),bottleneck_size=args.n)
            self.dec = CelebA_Decoder(output_shape=(3,64,64),bottleneck_size=args.n)
        
        elif args.dataset == "shape":
            self.enc = Shape_Encoder(args.n)
            self.dec = Shape_Decoder(args.n)
        elif args.dataset =="cifar10":
            input_shape = (32,32,3)
            self.encoder = Encoder(input_shape,bottleneck_size=args.n)
            self.decoder = Decoder(input_shape,bottleneck_size=args.n)
        
        # print(self.encoder,self.decoder)
        # if args.l>0:
        #     self.mlp = MLP(args.n, args.l)

    def encode(self, x):
        # z = self.enc(x)
        z = self.encoder(x)
        # z =  F.normalize(z,p=2,dim=1)
        # print(z.norm(p=2,dim=1))
        return z

    def decode(self, z):
        # return self.dec(z)
        return self.decoder(z)

    def forward(self, x):
        z = self.encoder(x)
        
        x_hat = self.decode(z)
        
        loss = F.mse_loss(x_hat, x)
        
        return loss,z