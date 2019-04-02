"""
# 2D kitti dataset test
# In this demo, 1) how to run kitti in cpu/gpu with pytorch, 2) how to partition the map and run parallelly
# TODO: setup for cuda
"""
import sys
import os
import time
import numpy as np
import pandas as pd
import torch as pt
import matplotlib.pyplot as pl
from bhmtorch_cuda import BHM2D_PYTORCH_CUDA

def getPartitions(cell_max_min, nPartx1, nPartx2):
    """
    :param cell_max_min: The size of the entire area
    :param nPartx1: How many partitions along the longitude
    :param nPartx2: How many partitions along the latitude
    :return: a list of all partitions
    """
    width = cell_max_min[1] - cell_max_min[0]
    height = cell_max_min[3] - cell_max_min[2]
    cell_max_min_segs = []
    for x in range(nPartx1):
        for y in range(nPartx2):
            seg_i = (cell_max_min[0] + width / nPartx1 * x, cell_max_min[0] + width / nPartx1 * (x + 1), \
                     cell_max_min[2] + height / nPartx2 * y, cell_max_min[2] + height / nPartx2 * (y + 1))
            cell_max_min_segs.append(seg_i)

    return cell_max_min_segs

def load_parameters(case):
    parameters = \
        {'kitti1': \
             ( os.path.abspath('../../Datasets/kitti/kitti2011_09_26_drive0001_frame'),
              (2, 2), #hinge point resolution
              (-80, 80, -80, 80), #area [min1, max1, min2, max2]
              None,
              None,
              0.5, #gamma
              ),

         }

    return parameters[case]

# Settings
dtype = pt.float32
device = pt.device("cpu")
#device = pt.device("cuda:0") # Uncomment this to run on GPU

# Read the file
fn_train, cell_resolution, cell_max_min, _, _, gamma = load_parameters('kitti1')

# Partition the environment into to 4 areas
# TODO: We can parallelize this
cell_max_min_segments = getPartitions(cell_max_min, 2, 2)

# Query data

# Read data
for framei in range(108):
    print('\nReading '+fn_train+'{}.csv...'.format(framei))
    g = pd.read_csv(fn_train+'{}.csv'.format(framei), delimiter=',').values[:, :]

    # Filter data
    layer = np.logical_and(g[:,2] >= 0.02, g[:,2] <= 0.125)
    #layer = np.logical_and(g[:, 2] >= -0.6, g[:, 2] <= -0.5)
    g = pt.tensor(g[layer, :], dtype=pt.float32)
    X = g[:, :2]
    y = g[:, 3].reshape(-1, 1)
    if pt.cuda.is_available():
        X = X.cuda()
        y = y.cuda()

    toPlot = []
    totalTime = 0
    for segi in range(len(cell_max_min_segments)):
        print(' Mapping segment {} of {}...'.format(segi+1,len(cell_max_min_segments)))
        cell_max_min = cell_max_min_segments[segi]

        bhm_mdl = BHM2D_PYTORCH_CUDA(gamma=gamma, grid=None, cell_resolution=cell_resolution, cell_max_min=cell_max_min, X=X, nIter=1)

        t1 = time.time()
        bhm_mdl.fit(X, y)
        t2 = time.time()
        totalTime += (t2-t1)

        # query the model
        q_resolution = 0.5
        #TODO: move this meshrid creation outside the loop
        xx, yy = np.meshgrid(np.arange(cell_max_min[0], cell_max_min[1] - 1, q_resolution),
                             np.arange(cell_max_min[2], cell_max_min[3] - 1, q_resolution))
        grid = np.hstack((xx.ravel()[:, np.newaxis], yy.ravel()[:, np.newaxis]))
        Xq = pt.tensor(grid, dtype=pt.float32).cuda()
        yq = bhm_mdl.predict(Xq)
        toPlot.append((Xq,yq))
    print(' Total training time={} s'.format(np.round(totalTime, 2)))

    # Plot frame i
    pl.close('all')
    for segi in range(len(cell_max_min_segments)):
        ploti = toPlot[segi]
        Xq, yq = ploti[0].cpu(), ploti[1].cpu()
        pl.scatter(Xq[:, 0], Xq[:, 1], c=yq, cmap='jet', s=5, vmin=0, vmax=1, edgecolors='')
    pl.colorbar()
    pl.xlim([-80,80]); pl.ylim([-80,80])
    pl.title('kitti2011_09_26_drive0001_frame{}'.format(framei))
    #pl.savefig(os.path.abspath('../../Outputs/kitti2011_09_26_drive0001_frame{}.png'.format(framei)))
    pl.show()