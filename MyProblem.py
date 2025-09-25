import numpy as np
import geatpy as ea
import h5py
import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool as ProcessPool
from bp import *
from blur_kernel import *
from Identification import *

z_trim = 0
z_offset = 0
width = 1
bin_resolution = 32e-12
isbackprop = 1
isdiffuse = 0
snr = 1e-1

class MyProblem(ea.Problem):

    def __init__(self, PoolType):  
        name = 'MyProblem' 
        M = 2 
        Dim = 252
        maxormins = [-1, 1]
        varTypes = [0] * Dim
        lb_first_three = [-3, 0.01] 
        ub_first_three = [3, 3]
        lb_rest = [0.001] * 250
        ub_rest = [10] * 250
        lb = lb_first_three + lb_rest
        ub = ub_first_three + ub_rest
        lbin = [1] * Dim
        ubin = [1] * Dim
        super().__init__(name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

       
        self.tof_data = np.array(h5py.File('statue.mat')['meas'])
        
        
        self.PoolType = PoolType
        if self.PoolType == 'Thread':
            self.pool = ThreadPool(2)
        elif self.PoolType == 'Process':
            num_cores = int(mp.cpu_count()) 
            self.pool = ProcessPool(num_cores)  

    def evalVars(self, Vars):
        num_individuals = Vars.shape[0]  
        args = [(Vars, self.tof_data, i) for i in range(num_individuals)]
        if self.PoolType == 'Thread':
            result_list = list(self.pool.map(subAimFunc, args))
        elif self.PoolType == 'Process':
            result = self.pool.map_async(subAimFunc, args)
            result.wait()
            result_list = result.get()
        ObjV1 = np.array([result[0] for result in result_list]).reshape(-1, 1)
        ObjV2 = np.array([result[1] for result in result_list]).reshape(-1, 1)
        print(ObjV1, ObjV2)
        return np.hstack([ObjV1, ObjV2])  
        

    def __del__(self):
       
        self.pool.close()
        self.pool.join()


def subAimFunc(args):
    
    Vars, tof_data, i = args
    x0 = Vars[i,0]  
    sigma = Vars[i,1]   
    core_c = Vars[i,2:252]
    core_value = core_G(x0,sigma, core_c)
    tof_data = np.transpose(tof_data, (0, 2, 1)) 
    tof_k = kernel(tof_data, core_value)
    tof_data_k = tof_data - tof_k
    vol, tic_x, tic_y, tic_z = backProjection(tof_data_k, width, bin_resolution, z_offset, z_trim, isdiffuse,
                                              isbackprop, snr)
    generated_image = np.max(vol, axis=0)
    Object, Artifact = Identification(generated_image)
    tenengrad_value = tenengrad(Object)
    count1 = count_zero_pixels(Object)
    sum_value = calculate_pixel_sum(Artifact)
    ratio = 3000*sum_value / count1 if count1 != 0 else 0
    return tenengrad_value,ratio