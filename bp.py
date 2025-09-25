import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy import fftpack
from scipy.io import loadmat
from scipy.signal import savgol_filter
from skimage.metrics import structural_similarity
from scipy.sparse import lil_matrix, spdiags
# import torch
from IM3 import *

def backProjection(tof_data, width, bin_resolution, z_offset, z_trim, isdiffuse, isbackprop, snr):
    c = 3e8
    N = tof_data.shape[1]
    M = tof_data.shape[0]
    range = M * c * bin_resolution
    tof_data[0:z_trim, :, :,] = 0
    psf = definePsf(N, M, width / range)
    fpsf = fftpack.fftn(psf)
    invpsf = (
        np.conj(fpsf) if isbackprop else np.conj(fpsf) / (np.abs(fpsf) ** 2 + 1 / snr)
    )
    mtx, mtxi = resamplingOperator(M)
    data = tof_data
    grid_z = np.tile(np.linspace(0, 1, M)[:, np.newaxis, np.newaxis], (1, N, N))
    data = data * (grid_z**4) if isdiffuse else data * (grid_z**2)
    tdata = np.zeros((2 * M, 2 * N, 2 * N))
    tdata[:M, :N, :N] = np.reshape(mtx @ data.reshape(M, -1), (M, N, N))
    tvol = fftpack.ifftn(fftpack.fftn(tdata) * invpsf)
    tvol = tvol[:M, :N, :N]
    vol = np.reshape(mtxi @ tvol.reshape(M, -1), (M, N, N))
    vol = np.maximum(np.real(vol), 0)
    tic_z = np.linspace(0, range / 2, vol.shape[0])
    tic_y = np.linspace(-width, width, vol.shape[1])
    tic_x = np.linspace(-width, width, vol.shape[2])
    ind = round(M * 2 * width / (range / 2))
    #vol = vol[:, :, ::-1]
    vol = vol[z_offset : ind + z_offset, :, :]
    return vol, tic_x, tic_y, tic_z


def definePsf(U, V, slope):
    x = np.linspace(-1, 1, 2 * U)
    y = np.linspace(-1, 1, 2 * U)
    z = np.linspace(0, 2, 2 * V)
    grid_z, grid_y, grid_x = np.meshgrid(z, y, x, indexing='ij')
    psf = np.abs((4*slope)**2 * (grid_x**2 + grid_y**2) - grid_z)
    psf = np.array(psf == np.tile(np.min(psf, axis=0), (2*V, 1, 1)))
    psf = psf / np.sum(psf[:, U, U])
    psf = psf / np.linalg.norm(psf)
    psf = np.roll(psf, U, axis=(1,2))
    return psf


def resamplingOperator(M):
    mtx = lil_matrix((M**2, M))
    x = np.arange(1, M**2 + 1)
    mtx[x - 1, np.ceil(np.sqrt(x)).astype(int) - 1] = 1
    mtx = spdiags(1 / np.sqrt(x),0,M**2,M**2) @ mtx
    mtxi = mtx.T
    K = np.log2(M)
    for k in range(int(K)):
        mtx = 0.5 * (mtx[::2, :] + mtx[1::2, :])
        mtxi = 0.5 * (mtxi[:, ::2] + mtxi[:, 1::2])
    return mtx, mtxi

# data = h5py.File('bunny.mat')
# tof_data = np.array(data['tof_data'])
# Z, M, N = tof_data.shape

# vol, tic_x,tic_y,tic_z = GuassBP_LCT(tof_data, width, bin_resolution, z_offset, z_trim, isdiffuse)

# fig = plt.figure()
# ax1 = fig.add_subplot(1, 3, 1)
# ax1.imshow(np.max(vol, axis=0), extent=[min(tic_x), max(tic_x), min(tic_y), max(tic_y)], origin='lower')
# ax1.set_title('Front view')
# ax1.set_xticks(np.linspace(min(tic_x), max(tic_x), 3))
# ax1.set_yticks(np.linspace(min(tic_y), max(tic_y), 3))
# ax1.set_xlabel('x (m)')
# ax1.set_ylabel('y (m)')
# ax1.set_aspect('equal', adjustable='box')
# plt.set_cmap('hot')
# ax2 = fig.add_subplot(1, 3, 2)
# ax2.imshow(np.max(vol, axis=1), extent=[min(tic_x), max(tic_x), min(tic_z), max(tic_z)], origin='lower')
# ax2.set_title('Top view')
# ax2.set_xticks(np.linspace(min(tic_x), max(tic_x), 3))
# ax2.set_yticks(np.linspace(min(tic_z), max(tic_z), 3))
# ax2.set_xlabel('x (m)')
# ax2.set_ylabel('z (m)')
# ax2.set_aspect('equal', adjustable='box')
# plt.set_cmap('hot')
# ax3 = fig.add_subplot(1, 3, 3)
# ax3.imshow(np.max(vol, axis=2).T, extent=[min(tic_z), max(tic_z), min(tic_y), max(tic_y)], origin='lower')
# ax3.set_title('Side view')
# ax3.set_xticks(np.linspace(min(tic_z), max(tic_z), 3))
# ax3.set_yticks(np.linspace(min(tic_y), max(tic_y), 3))
# ax3.set_xlabel('z (m)')
# ax3.set_ylabel('y (m)')
# ax3.set_aspect('equal', adjustable='box')
# plt.set_cmap('hot')
# plt.show()



# max_values = np.max(vol, axis=0)

# plt.figure()
# plt.imshow(np.max(vol, axis=0))
# plt.title('Front view')
# plt.set_cmap('hot')
# plt.show()