import numpy as np
from scipy.io import loadmat
import torch
import math
import sys
import io
import time
import matplotlib.pyplot as plt



start = time.time()

Nt = 32  # base station antennas
Nc = 32  # subcarriers (after DFT)
Nc_all = 125
img_channels = 2

#dataset_type = "Indoor"
dataset_type = "Outdoor"


# =====================================================================================================================================================================================
# Data from COST2100

if dataset_type == "Indoor":
    test_data = loadmat('DATA_Htestin.mat')
    H_test = test_data.get('HT')  # angular-delay channel matrix (after DFT transform --> from Nc' = 1024 subcarriers, we keep only the Nc = 32 first)
    # print(H_test)
    # print(H_test.shape)   # (20000, 2048) --> 20000 samples and 2048 is 2 X 32 X 32, where 2 indicates the real and imaginary part (2 channels) and Nt = 32, Nc = 32
    # first 1024 columns (32X32): real part and the rest 1024 columns: imaginary part
    
    
    H_test = H_test.astype('float32')
    
    H_test = np.reshape(H_test, (len(H_test), img_channels, Nt, Nc))   # from (20000, 2048) --> (20000, 2, 32, 32)
    
    H_test = torch.from_numpy(H_test.astype(np.float32))
    
    num_test_samples = 1000
    
    H_test = np.reshape(H_test[0:num_test_samples], (num_test_samples, 2, 32, 32))
    
    
    test_data_F = loadmat('DATA_HtestFin_all.mat')
    H_test_F = test_data_F.get('HF_all')    #  spatial-frequency channel matrix --> Nc' = 125 subcarriers
    H_test_F = np.reshape(H_test_F, (len(H_test_F), Nt, Nc_all))  # from (20000, 4000) --> (20000, 32, 125)
    H_test_F = H_test_F[0:num_test_samples]
    
    
    
    
if dataset_type == "Outdoor":
    test_data = loadmat('DATA_Htestout.mat')
    H_test = test_data.get('HT')  # angular-delay channel matrix (after DFT transform --> from Nc' = 1024 subcarriers, we keep only the Nc = 32 first)
    # print(H_test)
    # print(H_test.shape)   # (20000, 2048) --> 20000 samples and 2048 is 2 X 32 X 32, where 2 indicates the real and imaginary part (2 channels) and Nt = 32, Nc = 32
    # first 1024 columns (32X32): real part and the rest 1024 columns: imaginary part
    
    
    H_test = H_test.astype('float32')
    
    H_test = np.reshape(H_test, (len(H_test), img_channels, Nt, Nc))   # from (20000, 2048) --> (20000, 2, 32, 32)
    
    H_test = torch.from_numpy(H_test.astype(np.float32))
    
    num_test_samples = 1000
    
    H_test = np.reshape(H_test[0:num_test_samples], (num_test_samples, 2, 32, 32))
    
    
    test_data_F = loadmat('DATA_HtestFout_all.mat')
    H_test_F = test_data_F.get('HF_all')    #  spatial-frequency channel matrix --> Nc' = 125 subcarriers
    H_test_F = np.reshape(H_test_F, (len(H_test_F), Nt, Nc_all))  # from (20000, 4000) --> (20000, 32, 125)
    H_test_F = H_test_F[0:num_test_samples]

# =====================================================================================================================================================================================


def NMSE(test_heta, TEST_MODEL, FSQ, Ordering):

    if Ordering==True:
        if FSQ==True:
            with torch.no_grad():
                H_hat, test_indices = TEST_MODEL.model(H_test, False, test_heta)

        else:
            with torch.no_grad():
                vq_loss_test, H_hat = TEST_MODEL.model(H_test, False, test_heta)

    if Ordering == False:
        if FSQ == True:
            with torch.no_grad():
                H_hat, test_indices = TEST_MODEL.model(H_test, test_heta)

        else:
            with torch.no_grad():
                vq_loss_test, H_hat = TEST_MODEL.model(H_test, test_heta)


    H_test_real = np.reshape(H_test[:, 0, :, :], (len(H_test), -1))
    H_test_imag = np.reshape(H_test[:, 1, :, :], (len(H_test), -1))
    H_test_C = H_test_real - 0.5 + 1j * (H_test_imag - 0.5)

    # print("H TEST: ", H_test_C.shape)

    H_hat = H_hat.detach().numpy()
    H_hat = torch.from_numpy(H_hat.astype(np.float32))
    H_hat_real = np.reshape(H_hat[:, 0, :, :], (len(H_hat), -1))
    H_hat_imag = np.reshape(H_hat[:, 1, :, :], (len(H_hat), -1))
    H_hat_C = H_hat_real - 0.5 + 1j * (H_hat_imag - 0.5)

    # print("H HAT: ", H_hat_C.shape)

    H_test_C = H_test_C.numpy()
    H_hat_C = H_hat_C.numpy()

    # power = np.mean(abs(H_test_C)**2, axis=1)
    power = np.linalg.norm(H_test_C) ** 2
    # print("power = ", power)

    # MSE = np.sum(abs(H_test_C-H_hat_C)**2, axis=1)
    MSE = np.linalg.norm(H_test_C - H_hat_C) ** 2

    NMSE = 10 * math.log10(np.mean(MSE / power))
    #print("NMSE = ", NMSE, "dB")

    return NMSE



def Cosine_similarity(test_heta, TEST_MODEL, FSQ, Ordering):

    if Ordering==True:
        if FSQ==True:
            with torch.no_grad():
                H_hat, test_indices = TEST_MODEL.model(H_test, False, test_heta)

        else:
            with torch.no_grad():
                vq_loss_test, H_hat = TEST_MODEL.model(H_test, False, test_heta)

    if Ordering == False:
        if FSQ == True:
            with torch.no_grad():
                H_hat, test_indices = TEST_MODEL.model(H_test, test_heta)

        else:
            with torch.no_grad():
                vq_loss_test, H_hat = TEST_MODEL.model(H_test, test_heta)



    H_hat = H_hat.detach().numpy()
    H_hat = torch.from_numpy(H_hat.astype(np.float32))
    H_hat_real = np.reshape(H_hat[:, 0, :, :], (len(H_hat), -1))
    H_hat_imag = np.reshape(H_hat[:, 1, :, :], (len(H_hat), -1))
    H_hat_C = H_hat_real - 0.5 + 1j * (H_hat_imag - 0.5)
    H_hat_C = H_hat_C.numpy()


    H_hat_F = np.reshape(H_hat_C, (len(H_hat_C), Nt, Nc))
    H_hat_F = np.fft.fft(np.concatenate((H_hat_F, np.zeros((len(H_hat_C), Nt, 257 - Nc))), axis=2), axis=2)
    # print(H_hat_F.shape)  # (test samples, 32, 257)
    H_hat_F = H_hat_F[:, :, 0:Nc_all]
    # print(H_hat_F.shape)  # (test samples, 32, 125)

    n1 = abs(np.sqrt(np.sum(np.conj(H_test_F) * H_test_F, axis=1)))
    n1 = n1.astype('float64')
    n2 = abs(np.sqrt(np.sum(np.conj(H_hat_F) * H_hat_F, axis=1)))
    n2 = n2.astype('float64')
    aa = abs(np.sum(np.conj(H_hat_F) * H_test_F, axis=1))
    rho = np.mean(aa / (n1 * n2), axis=1)
    rho = np.mean(rho)

    #print("Correlation is ", rho)

    return rho



# Create a string buffer to temporarily redirect stdout
buffer = io.StringIO()

# Redirect stdout to the buffer to suppress prints
sys.stdout = buffer

import FSQ_AE_TEST
import Vector_Quantization_AE_TEST
import OVQ_FSQ_TEST
import OVQ_Scheme_New_TEST
#import FSQ_CRNet_TEST

# Reset stdout to its original state
sys.stdout = sys.__stdout__


fsq_csinet = []
vq_csinet = []
ovq_fsq_csinet = []
ovq_csinet = []

fsq_csinet_rho = []
vq_csinet_rho = []
ofsq_csinet_rho = []
ovq_csinet_rho = []



test_hetas = np.array([1/8, 1/4, 1/2, 3/4, 1])

for test_heta in test_hetas:
    fsq_csinet.append(NMSE(test_heta, FSQ_AE_TEST, True, False))
    vq_csinet.append(NMSE(test_heta, Vector_Quantization_AE_TEST, False, False))
    ovq_fsq_csinet.append(NMSE(test_heta, OVQ_FSQ_TEST, True, True))
    ovq_csinet.append(NMSE(test_heta, OVQ_Scheme_New_TEST, False, True))

    fsq_csinet_rho.append(Cosine_similarity(test_heta, FSQ_AE_TEST, True, False))
    vq_csinet_rho.append(Cosine_similarity(test_heta, Vector_Quantization_AE_TEST, False, False))
    ofsq_csinet_rho.append(Cosine_similarity(test_heta, OVQ_FSQ_TEST, True, True))
    ovq_csinet_rho.append(Cosine_similarity(test_heta, OVQ_Scheme_New_TEST, False, True))


end = time.time()

print("\nTime elapsed = ", end-start, " sec")


K = 128
b = 10
B = K*b*test_hetas   # B = [160, 320, 640, 1280]


# Plotting NMSE
plt.figure(1)
plt.plot(B, fsq_csinet, marker='o', linestyle='-', color='b', label='FSQ-CsiNet')
plt.plot(B, vq_csinet, marker='s', linestyle='--', color='r', label='VQ-CsiNet')
plt.plot(B, ovq_fsq_csinet, marker='^', linestyle='-.', color='g', label='OFSQ-CsiNet')
plt.plot(B, ovq_csinet, marker='d', linestyle=':', color='m', label='OVQ-CsiNet')

# Add a legend
plt.legend(loc='upper right', fontsize=10, frameon=True, shadow=True)

# Add titles and labels
plt_title = "NMSE " + dataset_type + " with CsiNet"
plt.title(plt_title, fontsize=16, fontweight='bold')
plt.xlabel("B [bit]", fontsize=12)
plt.ylabel("NMSE [dB]", fontsize=12)

# Add grid for better readability
plt.grid(True, which='both', linestyle='--', linewidth=0.6)

# Improve layout
plt.tight_layout()

# Save plot
plot_path = "CsiNet_PLOTS_NMSE_" + dataset_type + ".png"
plt.savefig(plot_path)



#Plotting rho
plt.figure(2)
plt.plot(B, fsq_csinet_rho, marker='o', linestyle='-', color='b', label='FSQ-CsiNet')
plt.plot(B, vq_csinet_rho, marker='s', linestyle='--', color='r', label='VQ-CsiNet')
plt.plot(B, ofsq_csinet_rho, marker='^', linestyle='-.', color='g', label='OFSQ-CsiNet')
plt.plot(B, ovq_csinet_rho, marker='d', linestyle=':', color='m', label='OVQ-CsiNet')
plt.plot(B, np.ones(len(B), dtype=int), linestyle='--', color='y', label='Perfect CSI')

# Add a legend
plt.legend(loc='lower right', fontsize=10, frameon=True, shadow=True)

# Add titles and labels
plt.title("Cosine Similarity " + dataset_type + " with CsiNet", fontsize=16, fontweight='bold')
plt.xlabel("B [bit]", fontsize=12)
plt.ylabel("ρ", fontsize=12)

# Add grid for better readability
plt.grid(True, which='both', linestyle='--', linewidth=0.6)

# Improve layout
plt.tight_layout()

# Save plot
plot_path = "CsiNet_PLOTS_rho_" + dataset_type + ".png"
plt.savefig(plot_path)


# Show the plot
plt.show()



fsq_csinet_plot = "fsq_csinet_plot_" + dataset_type + ".txt"
with open(fsq_csinet_plot, 'w') as f:
    f.write("NMSE\tCosine_Similarity\n")
    for nmse, r in zip(fsq_csinet, fsq_csinet_rho):
        f.write(f"{nmse}\t{r}\n")

vq_csinet_plot = "vq_csinet_plot_" + dataset_type + ".txt"
with open(vq_csinet_plot, 'w') as f:
    f.write("NMSE\tCosine_Similarity\n")
    for nmse, r in zip(vq_csinet, vq_csinet_rho):
        f.write(f"{nmse}\t{r}\n")

ovq_fsq_csinet_plot = "ovq_fsq_csinet_plot_" + dataset_type + ".txt"
with open(ovq_fsq_csinet_plot, 'w') as f:
    f.write("NMSE\tCosine_Similarity\n")
    for nmse, r in zip(ovq_fsq_csinet, ofsq_csinet_rho):
        f.write(f"{nmse}\t{r}\n")

ovq_csinet_plot = "ovq_csinet_plot_" + dataset_type + ".txt"
with open(ovq_csinet_plot, 'w') as f:
    f.write("NMSE\tCosine_Similarity\n")
    for nmse, r in zip(ovq_csinet, ovq_csinet_rho):
        f.write(f"{nmse}\t{r}\n")



