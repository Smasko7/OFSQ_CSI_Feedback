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



latent_dim = 512
b = 10
C = 2**b
m = 4
beta = 0.25


class Vector_Quantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        #print("NUM EMBEDDINGS = ", self._num_embeddings)

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)    # look up table (Codebook)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)   # initialize
        self._commitment_cost = commitment_cost


    def forward(self, inputs, fine_tuning, n):

        input_shape = inputs.shape
        #print("INPUT SHAPE = ", input_shape)

        flat_input = inputs.view(-1, self._embedding_dim)


        #print("FLAT INPUT : ", flat_input.shape)
        #print("EMBED WEIGHT : ", self._embedding.weight.shape)

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
        #quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        quantized = torch.matmul(encodings, self._embedding.weight)

        quantized = quantized.view(input_shape)

        # print("QUANTIZED SHAPE", quantized.shape)
        # print("QUANTIZED: ", quantized)

        if fine_tuning == True:
            K = input_shape[1]
            for i in range(len(n)):
                quantized[i, n[i]:K, :] = 0  # mask with zeros the last (K-n) embeding vectors (wi) (n is the same for the specific sample)

        # print("n = ", n)
        # print("QUANTIZED SHAPE2", quantized.shape)
        # print("QUANTIZED2: ", quantized)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()   # formula (4)
        # avg_probs = torch.mean(encodings, dim=0)
        # perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        #quantized = quantized.view(len(quantized), -1)


        return loss, quantized, encodings




class VQ_CsiNet(nn.Module):
    def __init__(self, latent_dim, embedding_dim, num_embeddings, commitment_cost):
        super().__init__()

        self.embedding_dim = embedding_dim  # dimension of quantized (embedding) vectors (m)
        self.num_embeddings = num_embeddings  # number of total quantized vectors (C)
        self.commitment_cost = commitment_cost  # beta
        self.latent_dim = latent_dim

        self._vq_csinet = Vector_Quantizer(num_embeddings, embedding_dim, commitment_cost)


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

    def forward(self, x, fine_tuning, test_heta=0):
        K = int(self.latent_dim / self.embedding_dim)


        # encoder
        y = self.encoder1(x)
        y = y.view(len(y), -1)    # reshape: (batch size, 2, 32, 32) --> (batch size, 2048)
        z = self.lin(y)

        #z = z.view(len(z), self.embedding_dim, K)
        z = z.view(len(z), K, self.embedding_dim)

        n = np.random.randint(1, K + 1, z.shape[0])  # n ~ U(1,K) | z.shape[0] == batch_size
 
        if fine_tuning == True:
            for i in range(len(n)):
                z[i, n[i]:K, :] = 0    # mask with zeros the last (K-n) vectors (n is the same for the specific sample)


        vq_loss, z_q, _ = self._vq_csinet(z, fine_tuning, n)

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

        return vq_loss, decoded




#model = VQ_AE(latent_dim, m, C, beta)
model = VQ_CsiNet(latent_dim, m, C, beta)

criterion = nn.MSELoss()

#optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)  # for COST2100
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=False)

#training
print("TRAIN")
epochs_pre_train = 200
epochs_fine_tune = 100
outputs = []
losses_pre_train = []

print("==================================================================================================================================")
print("PRE-TRAINING")

fine_tuning = False

for epoch in range(epochs_pre_train):
    for h_batch in data_loader:
        vq_loss, reconstructed_h = model(h_batch, fine_tuning)

        rec_loss = criterion(reconstructed_h, h_batch)   # Is this the loss of first term of formula (3)?

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
    losses_pre_train.append(rec_loss.item())
    print(outputs[-1])



print("==================================================================================================================================")
print("FINE-TUNING")
print("==================================================================================================================================")

fine_tuning = True

losses_fine_tune = []

for epoch in range(epochs_fine_tune):
    for h_batch in data_loader:

        vq_loss, reconstructed_h = model(h_batch, fine_tuning)

        rec_loss = criterion(reconstructed_h, h_batch)   # Is this the loss of first term of formula (3)?

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
    losses_fine_tune.append(rec_loss.item())
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
plot_path = "OVQ_Scheme_New_pre_train_rec_loss.png"
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")

# Save the losses to a text file
losses_file_path = "OVQ_Scheme_New_pre_train_rec_loss.txt"
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
plot_path = "OVQ_Scheme_New_fine_tune_rec_loss.png"
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")

# Save the losses to a text file
losses_file_path = "OVQ_Scheme_New_fine_tune_rec_loss.txt"
with open(losses_file_path, 'w') as f:
    for loss in losses_fine_tune:
        f.write(f"{loss}\n")
print(f"Pre train losses saved to {losses_file_path}")

# ====================================================================================================================================



end = time.time()

print("\nTraining time elapsed = ", end-start, " sec")


# ====================================================================================================================================
#SAVE MODEL

model_path = "OVQ_Scheme_New_path.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# ====================================================================================================================================



model_total_params = sum(p.numel() for p in model.parameters())
model_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Model's total parameters = ", model_total_params)
print("Model's trainable parameters = ", model_trainable_params)



# test (X test samples)
print("TEST (X TEST SAMPLES)")

num_test_samples = 5000

H_test = np.reshape(H_test[0:num_test_samples], (num_test_samples, 2, 32, 32))

test_heta = 1

vq_loss_test, H_hat = model(H_test, False, test_heta)

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


power = np.linalg.norm(H_test_C) ** 2
print("power = ", power)

# MSE = np.sum(abs(H_test_C-H_hat_C)**2, axis=1)
MSE = np.linalg.norm(H_test_C - H_hat_C) ** 2

print("MSE = ", MSE)
#print(MSE.shape)

NMSE = 10 * math.log10(np.mean(MSE / power))
print("OVQ_Scheme_New")
print("NMSE = ", NMSE, "dB")



