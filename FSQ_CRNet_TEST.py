from collections import OrderedDict
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
from numpy.linalg import norm
from scipy.io import loadmat
import pandas as pd
import math
from vector_quantize_pytorch import LFQ
from vector_quantize_pytorch import FSQ
import matplotlib.pyplot as plt



start = time.time()


Nt = 32  # base station antennas
Nc = 32  # subcarriers (after DFT)
img_channels = 2
M = 512   # compression rate: 2048/M

#dataset_type = "Indoor"
dataset_type = "Outdoor"

# =====================================================================================================================================================================================
# Data from COST2100


if dataset_type == "Indoor":
    test_data = loadmat('DATA_Htestin.mat')
    H_test = test_data.get('HT')  # angular-delay channel matrix (after DFT transform --> from Nc' = 1024 subcarriers, we keep only the Nc = 32 first)
    # print(H_test.shape)   # (20000, 2048) --> 20000 samples and 2048 is 2 X 32 X 32, where 2 indicates the real and imaginary part (2 channels) and Nt = 32, Nc = 32
    # first 1024 columns (32X32): real part and the rest 1024 columns: imaginary part


    H_test = H_test.astype('float32')

    H_test = np.reshape(H_test, (len(H_test), img_channels, Nt, Nc))   # from (20000, 2048) --> (20000, 2, 32, 32)

    H_test = torch.from_numpy(H_test.astype(np.float32))


    batch_size = 200  # number of samples per pass in training




 # =========================================================================================================================================================================


if dataset_type == "Outdoor":
    test_data = loadmat('DATA_Htestout.mat')
    H_test = test_data.get(
        'HT')  # angular-delay channel matrix (after DFT transform --> from Nc' = 1024 subcarriers, we keep only the Nc = 32 first)
    # print(H_test.shape)   # (20000, 2048) --> 20000 samples and 2048 is 2 X 32 X 32, where 2 indicates the real and imaginary part (2 channels) and Nt = 32, Nc = 32
    # first 1024 columns (32X32): real part and the rest 1024 columns: imaginary part
    
    H_test = H_test.astype('float32')

    H_test = np.reshape(H_test, (len(H_test), img_channels, Nt, Nc))  # from (20000, 2048) --> (20000, 2, 32, 32)

    H_test = torch.from_numpy(H_test.astype(np.float32))

    batch_size = 200  # number of samples per pass in training



#__all__ = ["crnet"]


class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_planes))
        ]))


class CRBlock(nn.Module):
    def __init__(self):
        super(CRBlock, self).__init__()
        self.path1 = nn.Sequential(OrderedDict([
            ('conv3x3', ConvBN(2, 7, 3)),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv1x9', ConvBN(7, 7, [1, 9])),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv9x1', ConvBN(7, 7, [9, 1])),
        ]))
        self.path2 = nn.Sequential(OrderedDict([
            ('conv1x5', ConvBN(2, 7, [1, 5])),
            ('relu', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv5x1', ConvBN(7, 7, [5, 1])),
        ]))
        self.conv1x1 = ConvBN(7 * 2, 2, 1)
        self.identity = nn.Identity()
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, x):
        identity = self.identity(x)

        out1 = self.path1(x)
        out2 = self.path2(x)
        out = torch.cat((out1, out2), dim=1)
        out = self.relu(out)
        out = self.conv1x1(out)

        out = self.relu(out + identity)
        return out


class CRNet(nn.Module):
    def __init__(self, reduction=4, latent_dim=512, embedding_dim=4):
        super(CRNet, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 32

        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim  # dimension of quantized (embedding) vectors (m)

        self.FSQ_Quantizer = FSQ([8,5,5,5])    # d = 4, L1 = 8, L2 = L3 = L4 = 5 --> 8*5*5*5 = 1000 ~= 1024=2^10 --> codebook size

        self.encoder1 = nn.Sequential(OrderedDict([
            ("conv3x3_bn", ConvBN(in_channel, 2, 3)),
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x9_bn", ConvBN(2, 2, [1, 9])),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv9x1_bn", ConvBN(2, 2, [9, 1])),
        ]))
        self.encoder2 = ConvBN(in_channel, 2, 3)
        self.encoder_conv = nn.Sequential(OrderedDict([
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x1_bn", ConvBN(4, 2, 1)),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.encoder_fc = nn.Linear(total_size, total_size // reduction)

        self.decoder_fc = nn.Linear(total_size // reduction, total_size)
        decoder = OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("CRBlock1", CRBlock()),
            ("CRBlock2", CRBlock())
        ])
        self.decoder_feature = nn.Sequential(decoder)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, test_heta=0):
        n, c, h, w = x.detach().size()  # batch_size,2,32,32

        K = int(self.latent_dim / self.embedding_dim)


        #encoder
        encode1 = self.encoder1(x)
        encode2 = self.encoder2(x)
        out = torch.cat((encode1, encode2), dim=1)
        out = self.encoder_conv(out)
        out = self.encoder_fc(out.view(n, -1))    # shape: batch_size, 512

        z = out.view(len(out), K, self.embedding_dim)


        #quantizer
        z_q, indices = self.FSQ_Quantizer(z)

        if test_heta != 0:
            z_q[:, int(test_heta*K) : K, :] = 0         # test_heta == η --> B = η * K * b = n' * b (n' : number of quantized vectors sent to BS for testing)


        z_q = z_q.view(n,-1)


        #decoder
        out = self.decoder_fc(z_q).view(n, c, h, w)
        out = self.decoder_feature(out)

        decoded = self.sigmoid(out)

        return decoded, indices


# def crnet(reduction=4):
#     r""" Create a proposed CRNet.
#
#     :param reduction: the reciprocal of compression ratio
#     :return: an instance of CRNet
#     """
#
#     model = CRNet(reduction=reduction)
#     return model

reduction = 4  # from 2*32*32 = 2048 --> 512 (latent space)
latent_dimension = 512
embedding_dimension = 4


# ====================================================================================================================================
# LOAD MODEL

model = CRNet(reduction=reduction, latent_dim=latent_dimension, embedding_dim=embedding_dimension)

model_path = "FSQ_CRNet_path_OUT_final.pth"
model.load_state_dict(torch.load(model_path))
#model.eval()  # Set the model to evaluation mode (important for testing)
print(f"Model loaded from {model_path}")

# ====================================================================================================================================


#Count model's parameters
model_total_params = sum(p.numel() for p in model.parameters())
model_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Model's total parameters = ", model_total_params)
print("Model's trainable parameters = ", model_trainable_params)



# ===============================================================================================================================================
# test (X test samples)
num_test_samples = 1000
print(f"TEST ({num_test_samples} TEST SAMPLES)")

H_test = np.reshape(H_test[0:num_test_samples], (num_test_samples, 2, 32, 32))

test_heta = 1

with torch.no_grad():
    H_hat, test_indices = model(H_test, test_heta)

H_test_real = np.reshape(H_test[:, 0, :, :], (len(H_test), -1))
H_test_imag = np.reshape(H_test[:, 1, :, :], (len(H_test), -1))
H_test_C = H_test_real - 0.5 + 1j * (H_test_imag - 0.5)

H_hat = H_hat.detach().numpy()
H_hat = torch.from_numpy(H_hat.astype(np.float32))
H_hat_real = np.reshape(H_hat[:, 0, :, :], (len(H_hat), -1))
H_hat_imag = np.reshape(H_hat[:, 1, :, :], (len(H_hat), -1))
H_hat_C = H_hat_real - 0.5 + 1j * (H_hat_imag - 0.5)

H_test_C = H_test_C.numpy()
H_hat_C = H_hat_C.numpy()


MSE = np.linalg.norm(H_test_C - H_hat_C) ** 2

power = np.linalg.norm(H_test_C) ** 2

NMSE = 10 * math.log10(np.mean(MSE / power))
print("NMSE = ", NMSE, "dB")




