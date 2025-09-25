# Non-line-of-sight imaging with adaptive artifact cancellation Code & Datasets

This repository contains code for the paper _Canny operator based artifact identification and suppression for Non-line-of-sight imaging_ by Zhongyun Chen,Ziyang Chen,Zhiwei Li,Wenjie Tang,Dejian Zhang,Tongbiao Wang,Qiegen Liu,Tianbao Yu

## Data

### "statue" 

- Description: TOF data of retroreflective letters placed at a distance of approximately 0.8m from the wall.
- Resolution: 64 x 64
- Scanned Area: 0.8 m x 0.8 m planar wall
- Acquisition Method: Experiment 

## Code

### MyProblem
These codes implement an adaptive artifact identification and suppression pipeline driven by a genetic algorithm(GA). Each GA individual encodes the blur-kernel parameters — including an amplitude weight, a center offset, the standard deviation, and the kernel coefficients — which are used to construct an adaptive blur kernel. For each candidate, the kernel is convolved with the measured ToF histogram and the convolution result is subtracted from the original histogram; the residual histogram is then reconstructed via back-projection and the resulting image is analyzed to separate object and artifact regions. The optimization is cast as a two-objective problem: maximize image sharpness as measured by the Tenengrad, while minimizing residual artifact strength as measured by the MAE.

### main
This program implements a multi-objective optimization solution based on the NSGA-II algorithm using the GeatPy library, featuring a user-defined optimization problem through the MyProblem class. The optimization process employs parallel computing with PoolType='Process' to accelerate execution, utilizing a real-valued encoded population (Encoding='RI') of 20 individuals (NIND=20). The algorithm runs for a maximum of 150 generations (MAXGEN=150) with an early stopping mechanism that halts optimization when fitness improvement remains below 1e-6 for 100 consecutive generations. Genetic operator parameters include a mutation probability (Pm) of 0.4 and a crossover rate (XOVR) of 0.6.

### algorithm
This code processes TOF data by applying an artifact suppression module optimized through a GA, followed by back-projection reconstruction to generate 3D volumetric results.

### blur_kernel
This code provides functions for generating and applying a high-degree-of-freedom blur kernel, implementing convolution operations to process TOF data.

### Identification
This code implements the Tenengrad gradient method to quantitatively measure image sharpness and focus quality, while also supporting calculations for zero-value pixel counts and total pixel value sums, primarily used for MAE computation in image quality assessment. 
The Identification module separates input images into distinct object regions and artifact regions through edge detection and morphological operations.
