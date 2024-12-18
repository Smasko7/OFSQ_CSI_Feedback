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

dataset_type = "Indoor"
#dataset_type = "Outdoor"

# =====================================================================================================================================================================================
# Data from COST2100


if dataset_type == "Indoor":
    test_data = loadmat('DATA_Htestin.mat')
    H_test = test_data.get('HT')  # angular-delay channel matrix (after DFT transform --> from Nc' = 1024 subcarriers, we keep only the Nc = 32 first)
    # print(H_test.shape)   # (20000, 2048) --> 20000 samples and 2048 is 2 X 32 X 32, where 2 indicates the real and imaginary part (2 channels) and Nt = 32, Nc = 32
    # first 1024 columns (32X32): real part and the rest 1024 columns: imaginary part


    train_data = loadmat('DATA_Htrainin.mat')
    H_train = train_data.get('HT')            # (100000, 2048) --> 100000 samples and 2048 is 2 X 32 X 32, where 2 indicates the real and imaginary part (2 channels) and Nt = 32, Nc = 32

    H_train = H_train.astype('float32')
    H_test = H_test.astype('float32')

    H_train = np.reshape(H_train, (len(H_train), img_channels, Nt, Nc))   # from (100000, 2048) --> (100000, 2, 32, 32)
    H_test = np.reshape(H_test, (len(H_test), img_channels, Nt, Nc))   # from (20000, 2048) --> (20000, 2, 32, 32)


    H_train = torch.from_numpy(H_train.astype(np.float32))
    H_test = torch.from_numpy(H_test.astype(np.float32))


    batch_size = 200  # number of samples per pass in training

    data_loader = torch.utils.data.DataLoader(dataset= H_train, batch_size=batch_size, shuffle=True)



 # =========================================================================================================================================================================


if dataset_type == "Outdoor":
    test_data = loadmat('DATA_Htestout.mat')
    H_test = test_data.get(
        'HT')  # angular-delay channel matrix (after DFT transform --> from Nc' = 1024 subcarriers, we keep only the Nc = 32 first)
    # print(H_test.shape)   # (20000, 2048) --> 20000 samples and 2048 is 2 X 32 X 32, where 2 indicates the real and imaginary part (2 channels) and Nt = 32, Nc = 32
    # first 1024 columns (32X32): real part and the rest 1024 columns: imaginary part
 
    train_data = loadmat('DATA_Htrainout.mat')
    H_train = train_data.get(
        'HT')  # (100000, 2048) --> 100000 samples and 2048 is 2 X 32 X 32, where 2 indicates the real and imaginary part (2 channels) and Nt = 32, Nc = 32

    H_train = H_train.astype('float32')
    H_test = H_test.astype('float32')

    H_train = np.reshape(H_train, (len(H_train), img_channels, Nt, Nc))  # from (100000, 2048) --> (100000, 2, 32, 32)
    H_test = np.reshape(H_test, (len(H_test), img_channels, Nt, Nc))  # from (20000, 2048) --> (20000, 2, 32, 32)

    H_train = torch.from_numpy(H_train.astype(np.float32))
    H_test = torch.from_numpy(H_test.astype(np.float32))

    batch_size = 200  # number of samples per pass in training

    data_loader = torch.utils.data.DataLoader(dataset=H_train, batch_size=batch_size, shuffle=True)


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

    def forward(self, x, fine_tuning):
        N, c, h, w = x.detach().size()  # batch_size,2,32,32

        K = int(self.latent_dim / self.embedding_dim)


        #encoder
        encode1 = self.encoder1(x)
        encode2 = self.encoder2(x)
        out = torch.cat((encode1, encode2), dim=1)
        out = self.encoder_conv(out)
        out = self.encoder_fc(out.view(N, -1))    # shape: batch_size, 512

        z = out.view(len(out), K, self.embedding_dim)


        #quantizer
        z_q, indices = self.FSQ_Quantizer(z)
        n = np.random.randint(1, K + 1, z.shape[0])  # n ~ U(1,K) | z.shape[0] == batch_size

        if fine_tuning == True:
            for i in range(len(n)):
                z_q[i, n[i]:K, :] = 0  # mask with zeros the last (K-n) vectors (n is the same for the specific sample)

        # is STE necessary here (there is STE in FSQ library)?
        z_q = z_q.view(N,-1)


        #decoder
        out = self.decoder_fc(z_q).view(N, c, h, w)
        out = self.decoder_feature(out)

        decoded = self.sigmoid(out)

        return decoded, indices



reduction = 4  # from 2*32*32 = 2048 --> 512 (latent space)
latent_dimension = 512
embedding_dimension = 4


model = CRNet(reduction=reduction, latent_dim=latent_dimension, embedding_dim=embedding_dimension)

criterion = nn.MSELoss()

#optimizer = torch.optim.Adam(model.parameters(), lr=5*1e-3, weight_decay=1e-4)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)   # lr=1e-2

#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 100, gamma=0.9)   # every 100 epochs decrease the lr by multplying it with 0.9

losses_pre_train = []

#training
print("TRAIN")
epochs_pre_train = 2000
epochs_fine_tune = 1000
outputs = []

print("==================================================================================================================================")
print("PRE-TRAINING")

fine_tuning = False

for epoch in range(epochs_pre_train):
    for h_batch in data_loader:
        reconstructed_h, indices = model(h_batch, fine_tuning)

        rec_loss = criterion(reconstructed_h, h_batch)   # Is this the loss of first term of formula (3)?

        print("rec loss = ", rec_loss.item())

        loss = rec_loss

        print("total loss = ", loss.item())

        optimizer.zero_grad()
        #scheduler.optimizer.zero_grad()
        loss.backward()
        #nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  # Gradient clipping
        optimizer.step()
        #scheduler.step(loss)

    print(f'Epoch: {epoch+1}, Loss: {loss.item():.4f}')
    #print(f'Epoch: {epoch + 1}, NMSE: {10*np.log10(loss.item()/norm(h_batch)**2):.4f} dB')
    outputs.append((epoch, h_batch, reconstructed_h))
    losses_pre_train.append(loss.item())
    print(outputs[-1])



print("==================================================================================================================================")
print("FINE-TUNING")
print("==================================================================================================================================")

fine_tuning = True

losses_fine_tune = []

for epoch in range(epochs_fine_tune):
    for h_batch in data_loader:

        reconstructed_h, indices = model(h_batch, fine_tuning)

        rec_loss = criterion(reconstructed_h, h_batch)   # Is this the loss of first term of formula (3)?

        print("rec loss = ", rec_loss.item())

        loss = rec_loss

        print("total loss = ", loss.item())

        optimizer.zero_grad()
        #scheduler.optimizer.zero_grad()
        loss.backward()
        #nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  # Gradient clipping
        optimizer.step()
        #scheduler.step(loss)

    print(f'Epoch: {epoch+1}, Loss: {loss.item():.4f}')
    #print(f'Epoch: {epoch + 1}, NMSE: {10*np.log10(loss.item()/norm(h_batch)**2):.4f} dB')
    outputs.append((epoch, h_batch, reconstructed_h))
    losses_fine_tune.append(loss.item())
    print(outputs[-1])

fine_tuning = False



# ====================================================================================================================================
# PLOT TRAINING CONVERGENCE

# PRE_TRAIN
iterations_pre_train = range(1, len(losses_pre_train) + 1)
plt.plot(iterations_pre_train, losses_pre_train)
plt.title('Pre-Train Reconstruction Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# SAVE THE PLOT
plot_path = "OFSQ_CRNet_pre_train_rec_loss.png"
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")

# Save the losses to a text file
losses_file_path = "OFSQ_CRNet_pre_train_rec_loss.txt"
with open(losses_file_path, 'w') as f:
    for loss in losses_pre_train:
        f.write(f"{loss}\n")
print(f"Pre train losses saved to {losses_file_path}")


# FINE-TUNING
iterations_fine_tune = range(1, len(losses_fine_tune) + 1)
plt.clf()
plt.plot(iterations_fine_tune, losses_fine_tune)
plt.title('Fine_Tuning Reconstruction Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# SAVE THE PLOT
plot_path = "OFSQ_CRNet_fine_tune_rec_loss.png"
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")

# Save the losses to a text file
losses_file_path = "OFSQ_CRNet_fine_tune_rec_loss.txt"
with open(losses_file_path, 'w') as f:
    for loss in losses_fine_tune:
        f.write(f"{loss}\n")
print(f"Pre train losses saved to {losses_file_path}")


# ====================================================================================================================================



end = time.time()

print("\nTraining time elapsed = ", end-start, " sec")


# ====================================================================================================================================
#SAVE MODEL

model_path = "OFSQ_CRNet_path.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# ====================================================================================================================================


#Count model's parameters
model_total_params = sum(p.numel() for p in model.parameters())
model_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Model's total parameters = ", model_total_params)
print("Model's trainable parameters = ", model_trainable_params)



# ===============================================================================================================================================
# test (X test samples)
num_test_samples = 10000
print(f"TEST ({num_test_samples} TEST SAMPLES)")

H_test = np.reshape(H_test[0:num_test_samples], (num_test_samples, 2, 32, 32))

with torch.no_grad():
    H_hat, test_indices = model(H_test, False)

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




