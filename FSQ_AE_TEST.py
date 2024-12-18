import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
from numpy.linalg import norm
from scipy.io import loadmat
import pandas as pd
import math
from vector_quantize_pytorch import LFQ
from vector_quantize_pytorch import FSQ


start = time.time()


Nt = 32  # base station antennas
Nc = 32  # subcarriers (after DFT)
img_channels = 2

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

latent_dim = 512
b = 10
C = 2**b
m = 4
beta = 0.25




class VQ_CsiNet(nn.Module):
    def __init__(self, latent_dim, embedding_dim, num_embeddings, commitment_cost):
        super().__init__()

        self.embedding_dim = embedding_dim  # dimension of quantized (embedding) vectors (m)
        self.num_embeddings = num_embeddings  # number of total quantized vectors (C)
        self.commitment_cost = commitment_cost  # beta
        self.latent_dim = latent_dim

        self.FSQ_Quantizer = FSQ([8,5,5,5])    # d = 4, L1 = 8, L2 = L3 = L4 = 5 --> 8*5*5*5 = 1000 ~= 1024=2^10 --> codebook size


        #size: batch_size, 2, 32, 32 -->  (batch_size, output channels, image size)
        self.encoder1 = nn.Sequential(

            nn.Conv2d(2, 2, 3, stride=1, padding=1),   #size: batch_size, 2, 32, 32
            nn.BatchNorm2d(2),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),

        )

        self.lin = nn.Linear(2048, latent_dim)  # compression ratio = 4

        self.lin2 = nn.Linear(latent_dim, 2048)

        # size: batch_size, 2, 32, 32
        self.refineNet = nn.Sequential(
            nn.Conv2d(2, 8, 3, stride=1, padding=1),  # size: batch_size, 8, 32, 32
            nn.BatchNorm2d(8),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),

            nn.Conv2d(8, 16, 3, stride=1, padding=1),  # size: batch_size, 16, 32, 32
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),

            nn.Conv2d(16, 2, 3, stride=1, padding=1),  # size: batch_size, 2, 32, 32
            nn.BatchNorm2d(2)

        )

        self.finalDecoder = nn.Sequential(
            nn.Conv2d(2, 2, 3, stride=1, padding=1),  # size: batch_size, 2, 32, 32
            nn.BatchNorm2d(2),
            nn.Sigmoid()
        )

    def forward(self, x, test_heta = 0):
        K = int(self.latent_dim / self.embedding_dim)


        # encoder
        y = self.encoder1(x)
        y = y.view(len(y), -1)    # reshape: (100000, 2, 32, 32) --> (100000, 2048)
        z = self.lin(y)

        z = z.view(len(z), K, self.embedding_dim)


        z_q, indices = self.FSQ_Quantizer(z)
        
        if test_heta != 0:
            z_q[:, int(test_heta*K) : K, :] = 0         # test_heta == η --> B = η * K * b = n' * b (n' : number of quantized vectors sent to BS for testing)


        z_q = z_q.view(len(z_q), -1)

        # decoder
        y = self.lin2(z_q)
        y = y.view(len(y), 2, 32, 32)

        for i in range(2):
            # y = self.refineNet(y)
            # y = y+x
            z = self.refineNet(y)
            y = y+z
            leakyrelu = nn.LeakyReLU(negative_slope=0.3, inplace=True)
            y = leakyrelu(y)

        decoded = self.finalDecoder(y)

        return decoded, indices


# ====================================================================================================================================
# LOAD MODEL

model = VQ_CsiNet(latent_dim, m, C, beta)

model_path = "FSQ_AE_path_OUT.pth"
model.load_state_dict(torch.load(model_path))
#model.eval()  # Set the model to evaluation mode (ignores batch normalizations etc)
print(f"Model loaded from {model_path}")

# ====================================================================================================================================


model_total_params = sum(p.numel() for p in model.parameters())
model_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Model's total parameters = ", model_total_params)
print("Model's trainable parameters = ", model_trainable_params)




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

#print("H TEST: ", H_test_C.shape)

H_hat = H_hat.detach().numpy()
H_hat = torch.from_numpy(H_hat.astype(np.float32))
H_hat_real = np.reshape(H_hat[:, 0, :, :], (len(H_hat), -1))
H_hat_imag = np.reshape(H_hat[:, 1, :, :], (len(H_hat), -1))
H_hat_C = H_hat_real - 0.5 + 1j * (H_hat_imag - 0.5)

#print("H HAT: ", H_hat_C.shape)

H_test_C = H_test_C.numpy()
H_hat_C = H_hat_C.numpy()

# power = np.mean(abs(H_test_C)**2, axis=1)
power = np.linalg.norm(H_test_C) ** 2
print("power = ", power)

# MSE = np.sum(abs(H_test_C-H_hat_C)**2, axis=1)
MSE = np.linalg.norm(H_test_C - H_hat_C) ** 2

print("MSE = ", MSE)
#print(MSE.shape)

NMSE = 10 * math.log10(np.mean(MSE / power))
print("NMSE = ", NMSE, "dB")






