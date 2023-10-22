import sys, os, argparse, yaml
from datasets.config.config import data_analysis_parameters
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cbook as cbook
def analysis_kitti(args):
    # Load the data
    flow_volume = []
    masks = []
    height, width = args.height , args.width
    for flow_path in os.listdir(args.flow_path_occ):
        flow = cv.resize(cv.imread(os.path.join(args.flow_path_occ, flow_path), cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH), (width, height))
        u = flow[:,:,2]/64.0 - 512
        v = flow[:,:,1]/64.0 - 512
        mask = flow[:,:,0]
        mag = np.sqrt(u**2 + v**2)
        flow_volume.append(mag)
        masks.append(mask + 0.00000001)
    flow_volume = np.array(flow_volume)
    mean = np.mean(flow_volume,axis=0)
    # standard_deviation = np.std(flow_volume,axis=0)
    masks = np.array(masks)
    mean = np.average(flow_volume,weights = masks,axis=0)
    # add plot and colorbar
    fig, ax = plt.subplots(1,1)
    
    mean_flow_plot = ax.imshow(mean,cmap='Blues',norm=colors.LogNorm(vmin=mean.min()+ 0.01, vmax=mean.max()+0.0000001))
    fig.colorbar(mean_flow_plot, ax=ax)
    # ax[1].imshow(standard_deviation,cmap='rainbow')
    plt.show()
    
def analysis_vkitti(args):
    flow_volume = []
    masks = []
    height, width = args.height , args.width
    for flow_path in os.listdir(args.flow_path_occ):
        flow = cv.resize(cv.imread(os.path.join(args.flow_path_occ, flow_path), cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH), (width, height))
        u = flow[:,:,2]
        v = flow[:,:,1]
        u = 2*(u/(2**16)) #- 0.5 
        v = 2*(v/(2**16)) #- 0.5
        # u = u*(width - 1)
        # v = v*(height - 1)
        mask = flow[:,:,0]
        print(min(u.flatten()),max(u.flatten()))
        mag = np.sqrt(u**2 + v**2)
        flow_volume.append(mag)
        masks.append(mask)
    flow_volume = np.array(flow_volume)
    masks = np.array(masks)
    mean = np.average(flow_volume,weights = masks,axis=0)
    # standard_deviation = np.std(flow_volume,axis=0)
    # add plot and colorbar
    fig, ax = plt.subplots(1,1)
    
    mean_flow_plot = ax.imshow(mean,cmap='Blues',norm=colors.LogNorm(vmin=mean.min() + 0.01, vmax=mean.max()))
    fig.colorbar(mean_flow_plot, ax=ax)
    # ax[1].imshow(standard_deviation,cmap='rainbow')
    plt.show()

# Load dataset and the parameters to analyse from the config file
parser = argparse.ArgumentParser()
parser.add_argument('-config', help="configuration file *.yml", type=str, required=False, default='data_analysis/config/vkitti.yml')
parser.add_argument('-dataset', help="dataset", type=str, required=False, default="vkitti")

analysis_args = parser.parse_args()

# Load the configuration file arguments
args = data_analysis_parameters(analysis_args.dataset, analysis_args.config)
analysis_vkitti(args)


    