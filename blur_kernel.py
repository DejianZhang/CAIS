import numpy as np


def core_G(x0,sigma,core_c):

    x = np.linspace(-10, 10, 250)

    gaussian_wave_packet =np.exp(-(x - x0)**2 / (2 * sigma**2))
    core_value = gaussian_wave_packet*core_c
    
    return core_value

def kernel(tof_data,gaussian_wave_packet):
    M, N = tof_data.shape[0], tof_data.shape[2]
    tof_G = np.zeros((M,N,N))
    sigma_index = 0 
    for i in range(N):
        for j in range(N):
            a = tof_data[:,i, j]
            
            b = np.convolve(a, gaussian_wave_packet, mode='same')/sum(gaussian_wave_packet)
            
            tof_G[:,i, j] =b 
            sigma_index += 1 
    return tof_G

