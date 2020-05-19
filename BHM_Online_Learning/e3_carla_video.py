"""
# stacking data and kernels
# memory - about 3 frames: current and two past ones
# Lydia Chan
"""
# import matplotlib.patches as pc
# import random
import pandas as pd
import math
import os
import time
import numpy as np
import torch as pt
import matplotlib.pyplot as pl
from bhmtorch_cpu import BHM2D_PYTORCH

def _initial_value(weight):
    return (weight[0] == 0) or (weight[1] == 10000)

def _save_new_weights(weights, opt = 'pick_highest_confidence'):
    weights_dict = {}

    for weight in weights:
        t, x1, x2, mu, sig = weight
        kernel = (x1, x2)

        if kernel in weights_dict:
            # replace
            if np.size(weights_dict[kernel]) == 2 and _initial_value(weights_dict[kernel]):
                weights_dict[kernel] = [mu, sig]
            elif not _initial_value([mu, sig]):
                weights_dict[kernel] = np.vstack((weights_dict[kernel], [mu, sig]))
                # take only last two, by now guaranteed at least two elements
                weights_dict[kernel] = weights_dict[kernel][-2:]
        else:
            weights_dict[kernel] = [mu, sig]

    kernel_weight = np.empty([0,4])
    for kernel in weights_dict:
        x1, x2 = kernel
        params = weights_dict[kernel]

        # params should not be empty
        if np.size(params) > 2:
            num_param = (np.size(params))/2
            
            if opt == 'weight_equally':
                mu = sum(params[:,0]) / (num_param)
                sig = math.sqrt(sum([s**2 for s in params[:,1]]) / (num_param**2))
            elif opt == 'remove_low_confidence':
                confident_paras_indx = np.log(params[:,1]) <= 3 #3-4 in log-scale seems to be a good value. visualize sig and see.
                mu = sum(params[confident_paras_indx,0]) / (num_param)
                sig = math.sqrt(sum([s**2 for s in params[confident_paras_indx,1]]) / (num_param**2))
            elif opt == 'pick_highest_confidence':
                hightest_conf_indx = np.argmin(params[:,1])
                mu = params[hightest_conf_indx, 0]
                sig = params[hightest_conf_indx, 1]
            elif opt == 'pick_first_weight':
                mu = params[0, 0]
                sig = params[0, 1]
            kernel_weight = np.vstack((kernel_weight, [x1, x2, mu, sig]))
        elif np.size(params) > 0:
            mu, sig = params
            kernel_weight = np.vstack((kernel_weight, [x1, x2, mu, sig]))

    return kernel_weight
    

def _hinge_round(x, base):
    return base * pt.round(x/base)

def _frame_grid(Xt, res):
    min1, max1 = Xt[:,0].min(), Xt[:,0].max()
    min2, max2 = Xt[:,1].min(), Xt[:,1].max()
    grid = [min1, max1, min2, max2]

    grid = [_hinge_round(x, res) for x in grid]
    grid[0] -= 2*res
    grid[2] -= 2*res
    grid[1] += 2*res
    grid[3] += 2*res

    print('GRID min/max x1, x2 ', grid)

    return grid

def load_parameters(case):
    parameters = \
        {
        'e1_carla': \
        # ( os.path.abspath('carla_town1_50m_1channel.npz'),
        (os.path.abspath('../Datasets/carla_town1/carla_town1_lydia_50m_1channel.npz'),
           (2,2),  #x1 and x2 resolutions for positioning hinge points
         (-450, 50, -50, 380),  # area to be mapped [x1_min, x1_max, x2_min, x2_max]
         100,
         1,  #N/A
         0.3,  #threshold for filtering data
         0.7 #gamma: kernel parameter
          ),
         }

    return parameters[case]


def train_data(filename, frames, out_path, condition):
    # Settings
    dtype = pt.float32
    device = pt.device("cpu")

    # Global variables and hyperparameters
    QUERY_RESOLUTION = 0.5
    QUERY_RESOLUTION_FULLMAP = 2*QUERY_RESOLUTION
    all_parameters = np.asarray([])

    # Read the train and test files
    fn_train, cell_resolution, cell_max_min, max_distance, _, _, gamma = load_parameters(filename)
    print('\nReading '+fn_train)
    dt = np.load(fn_train)
    Xt = dt['X_train']
    yt = dt['Y_train'].reshape(-1, 1)

    # Combine x and y data in one big matrix
    g = np.hstack((Xt, yt))

    # Learn parameters for each frame
    for framei in frames:
        print('\nReading frame {}'.format(framei))

        # Filter by frame
        layer = (g[:,0] == framei)
        buffer_datapoints = g[layer, :]

        # Extract X and y data from memory buffer for plotting and training
        f = pt.tensor(buffer_datapoints, dtype=pt.float32)
        X = f[:, 1:3]
        y = f[:, 3].reshape(-1, 1)

        # Create kernels for learning based on datapoints max and min
        frame_limit = _frame_grid(X, cell_resolution[0])
        xx, yy = np.meshgrid(np.arange(frame_limit[0], frame_limit[1], cell_resolution[0]), \
                             np.arange(frame_limit[2], frame_limit[3], cell_resolution[1]))
        model_grid = np.hstack((xx.ravel()[:, np.newaxis], yy.ravel()[:, np.newaxis]))

        # Graph LiDAR points
        ones_ = np.where(buffer_datapoints[:,3]==1)
        hit_pts = buffer_datapoints[ones_]
        zeros_ = np.where(buffer_datapoints[:,3]==0)
        free_pts = buffer_datapoints[zeros_]

        # IMAGE 1: LiDAR Hit Points and Free Points
        fig, axs = pl.subplots(1, 4, figsize=(18, 3.5))
        axs[0].plot(free_pts[:,1], free_pts[:,2],'b.',ms=1.0)
        axs[0].plot(hit_pts[:,1], hit_pts[:,2], 'r.',ms=1.0)
        axs[0].set_title('LiDAR Points')
        fig.suptitle('Frame {}'.format(framei))

        # Learn parameters for the current frame
        totalTime = 0

        bhm_mdl = BHM2D_PYTORCH(gamma=gamma, grid=model_grid, cell_resolution=cell_resolution, cell_max_min=cell_max_min, X=X, nIter=1)

        t1 = time.time()
        mu, sig = bhm_mdl.fit(X, y)
        t2 = time.time()
        totalTime += (t2-t1)
        
        # Record mu, sig parameters for kernels in format: time, x1, x2, mu, sig
        mu = mu.numpy().reshape(-1,1)
        sig = sig.numpy().reshape(-1,1)
        current_parameters = np.hstack((model_grid, np.hstack((mu,sig))))
        frame_header = np.full((current_parameters.shape[0], 1), framei)
        current_parameters = np.hstack((frame_header, current_parameters))
        all_parameters = np.append(all_parameters.reshape(-1, 5), current_parameters, axis=0)

        # Query the model
        xx, yy= np.meshgrid(np.arange(frame_limit[0]-cell_resolution[0], frame_limit[1], QUERY_RESOLUTION),
                            np.arange(frame_limit[2]-cell_resolution[1], frame_limit[3], QUERY_RESOLUTION))
        query_grid = np.hstack((xx.ravel()[:, np.newaxis], yy.ravel()[:, np.newaxis]))
        Xq = pt.tensor(query_grid, dtype=pt.float32)
        yq = bhm_mdl.predict(Xq)
        mean, zq = bhm_mdl.predictSampling(Xq)
        print(' Total training time={} s'.format(np.round(totalTime, 2)))
        
        # IMAGE 2: Instantaneous MEAN map
        axs[1].scatter(Xq[:, 0], Xq[:, 1], c=yq, cmap='jet', s=5, vmin=0, vmax=1, edgecolors='')
        pcm =axs[1].get_children()[0] #get the mappable, the 1st and the 2nd are the x and y axes
        pl.colorbar(pcm, ax=axs[1])
        axs[1].set_title('Instantaneous map (mean)')

        # IMAGE 3: Instantaneous VARIANCE map
        axs[2].scatter(Xq[:, 0], Xq[:, 1], c=zq, cmap='jet', s=5, vmin=0, vmax=0.51, edgecolors='')
        pcm =axs[2].get_children()[0] #get the mappable, the 1st and the 2nd are the x and y axes
        pl.colorbar(pcm, ax=axs[2])
        axs[2].set_title('Instantaneous map (variance)')
        
        current_data = np.hstack((Xq, yq.reshape(-1,1)))
        current_data = np.hstack((current_data, zq.reshape(-1,1)))
        np.save(out_path + 'mean_var_frame{}'.format(framei), current_data)

        # Add in entire map
        all_data = _save_new_weights(all_parameters, condition)
        kernel_grid = all_data[:, 0:2]
        kernel_weights = all_data[:, 2:4]

        # Create the model
        bhm_mdl = BHM2D_PYTORCH(gamma=gamma, grid=kernel_grid, cell_resolution=cell_resolution, cell_max_min=cell_max_min, X=None, nIter=1, mu_sig = kernel_weights)
        
        # Query the model
        xx, yy= np.meshgrid(np.arange(cell_max_min[0], cell_max_min[1], QUERY_RESOLUTION_FULLMAP),
                            np.arange(cell_max_min[2], cell_max_min[3], QUERY_RESOLUTION_FULLMAP))
        all_query_grid = np.hstack((xx.ravel()[:, np.newaxis], yy.ravel()[:, np.newaxis]))
        
        Xq_all = pt.tensor(all_query_grid, dtype=dtype)

        print("Grid for entire map initialized")

        t5 = time.time()
        all_yq = bhm_mdl.predict(Xq_all)
        t6 = time.time()
        globalPredTime = (t6-t5)
        print(' Total global predict time={} s'.format(np.round(globalPredTime, 2)))

        print("\nPlotting for frame {}".format(framei))

        # IMAGE 4 MEAN MAP OF TOWN
        axs[3].scatter(Xq_all[:, 0], Xq_all[:, 1], c=all_yq, cmap='jet', s=5, vmin=0, vmax=1, edgecolors='')
        pcm =axs[3].get_children()[0] #get the mappable, the 1st and the 2nd are the x and y axes
        pl.colorbar(pcm, ax=axs[3])
        axs[3].set_xlim(cell_max_min[0], cell_max_min[1])
        axs[3].set_ylim(cell_max_min[2], cell_max_min[3])
        axs[3].set_title('Map (mean) of the town')

        # Save the figure
        pl.savefig(os.path.abspath(out_path + 'frame{}.png'.format(framei)))
        pl.close('all')

        np.save(out_path + 'all_parameters_frame{}_to_{}'.format(FRAMES[0], FRAMES[-1]),all_parameters)

    return all_parameters
    
# SET UP
CONDITIONS = {'we':'weight_equally', 'lc':'remove_low_confidence', 'ph': 'pick_highest_confidence', \
                'pf': 'pick_first_weight'}
FRAMES = range(0,4088)
FILENAME = 'e1_carla'
OUT_PATH = 'out/'

OPTION = 'ph'

# MAIN
all_parameters = train_data(FILENAME, FRAMES, OUT_PATH, CONDITIONS[OPTION])
