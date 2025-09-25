import h5py
import numpy as np
from bp import *
from blur_kernel import *
import matplotlib.pyplot as plt


z_trim = 0
z_offset = 0
width = 1
bin_resolution = 32e-12
isbackprop = 1
isdiffuse = 0
snr = 1e-1

tof_data = np.array(h5py.File('statue.mat')['meas'])
tof_data = np.transpose(tof_data, (0, 2, 1)) 
result_csv_path = 'result.csv' 
thresholds_data = np.genfromtxt(result_csv_path, delimiter=',')
thresholds = np.ravel(thresholds_data)
x0=thresholds[0]
sigma=thresholds[1]
core_c=thresholds[2:252]
core_value = core_G(x0,sigma,core_c)
tof_k = kernel(tof_data,core_value)
tof_data_k = tof_data - tof_k

vol, tic_x, tic_y, tic_z = backProjection(tof_data_k, width, bin_resolution, z_offset, z_trim, isdiffuse, isbackprop,
                                          snr)
plt.figure()
plt.imshow(np.max(vol, axis=0))
plt.title(f'statue Front view ')
plt.set_cmap('hot')
plt.show()  
plt.close()  
