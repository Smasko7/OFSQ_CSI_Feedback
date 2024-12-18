from collections import OrderedDict
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



class Vector_Quantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        #print("NUM EMBEDDINGS = ", self._num_embeddings)

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)    # look up table (Codebook)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)   # initialize
        #self._embedding.weight.data.uniform_(0,1)
        self._commitment_cost = commitment_cost


    def forward(self, inputs):

        input_shape = inputs.shape
        #print("INPUT SHAPE = ", input_shape)

        flat_input = inputs.view(-1, self._embedding_dim)


        # print("FLAT INPUT : ", flat_input.shape)
        # print("EMBED WEIGHT : ", self._embedding.weight.shape)

        # Calculate distances
        term1 = torch.sum(flat_input ** 2, dim=1, keepdim=True)
        term2 = torch.sum(self._embedding.weight ** 2, dim=1)
        term3 = torch.matmul(flat_input, self._embedding.weight.t())

        # print("TERM1 : ", term1.shape)
        # print("TERM2 : ", term2.shape)
        # print("TERM3 : ", term3.shape)

        distances = (term1
                     + term2
                     - 2 * term3)     # formula (2)

        # print("DIST : ", distances.shape)
        # print(distances)


        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        # print("ENCODING INDICES : ", encoding_indices.shape)
        # print(encoding_indices)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        # print("ENCODINGS : ", encodings.shape)
        # print(encodings)
        encodings.scatter_(1, encoding_indices, 1)   # one hot encoding
        # print("ENCODINGS : ", encodings.shape)
        # print(encodings)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()   # formula (4)
        # avg_probs = torch.mean(encodings, dim=0)
        # perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        quantized = quantized.view(len(quantized), -1)


        return loss, quantized, encodings






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
    def __init__(self, num_embeddings, commitment_cost, reduction=4, latent_dim=512, embedding_dim=4):
        super(CRNet, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 32

        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim  # dimension of quantized (embedding) vectors (m)
        self.num_embeddings = num_embeddings  # number of total quantized vectors (C)
        self.commitment_cost = commitment_cost  # beta

        self._vq = Vector_Quantizer(num_embeddings, embedding_dim, commitment_cost)


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

    def forward(self, x):
        n, c, h, w = x.detach().size()  # batch_size,2,32,32

        K = int(self.latent_dim / self.embedding_dim)


        #encoder
        encode1 = self.encoder1(x)
        encode2 = self.encoder2(x)
        out = torch.cat((encode1, encode2), dim=1)
        out = self.encoder_conv(out)
        out = self.encoder_fc(out.view(n, -1))    # shape: batch_size, 512

        z = out.view(len(out), self.embedding_dim, K)


        #quantizer
        vq_loss, z_q, _ = self._vq(z)
        #z_q = z_q.view(n,-1)


        #decoder
        out = self.decoder_fc(z_q).view(n, c, h, w)
        out = self.decoder_feature(out)

        decoded = self.sigmoid(out)

        return vq_loss, decoded


reduction = 4  # from 2*32*32 = 2048 --> 512 (latent space)
latent_dimension = 512
embedding_dimension = 4
beta = 0.25
b = 10
C = 2**b



model = CRNet(C, beta, reduction=reduction, latent_dim=latent_dimension, embedding_dim=embedding_dimension)

criterion = nn.MSELoss()

#optimizer = torch.optim.Adam(model.parameters(), lr=5*1e-3, weight_decay=1e-4)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)   # lr=1e-2

#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 100, gamma=0.9)   # every 100 epochs decrease the lr by multplying it with 0.9

losses = []

#training
print("TRAIN")
epochs = 200
outputs = []

for epoch in range(epochs):
    for h_batch in data_loader:
        vq_loss, reconstructed_h = model(h_batch)

        rec_loss = criterion(reconstructed_h, h_batch)

        print("rec loss = ", rec_loss.item())

        loss = rec_loss + vq_loss

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
    losses.append(rec_loss.item())
    print(outputs[-1])


# ====================================================================================================================================
# PLOT TRAINING CONVERGENCE
iterations = range(1, len(losses) + 1)
plt.plot(iterations, losses)
plt.title('Reconstruction Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# SAVE THE PLOT
plot_path = "VQ_CRNet_rec_loss.png"
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")

# Save the losses to a text file
losses_file_path = "VQ_CRNet_rec_losses.txt"
with open(losses_file_path, 'w') as f:
    for loss in losses:
        f.write(f"{loss}\n")
print(f"Losses saved to {losses_file_path}")

# ====================================================================================================================================


end = time.time()

print("\nTraining time elapsed = ", end-start, " sec")


# ====================================================================================================================================
#SAVE MODEL

model_path = "VQ_CRNet_path.pth"
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
    vq_loss_test, H_hat = model(H_test)

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




