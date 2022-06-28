import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from torch import absolute
def evaluate_flow(estimated_flow, gt_flow):
    error_u = estimated_flow[:, :, 0] - gt_flow[:, :, 0]
    error_v = estimated_flow[:, :, 1] - gt_flow[:, :, 1]
    
    # EPE
    epe_map = np.sqrt(error_u**2 + error_v**2)
    epe = np.mean(epe_map)
    
    # absolute error
    abs_map = np.abs(error_u - error_v)
    abs_error = np.mean(abs_map)
    
    error_map = gt_flow - estimated_flow
    # generate error map visualization
    hsv = np.zeros((error_map.shape[0], error_map.shape[1], 3), dtype=np.uint8)
    error_map_magnitude, error_map_angle = cv.cartToPolar(error_map[..., 0].astype(np.float32), error_map[..., 1].astype(np.float32))

    nans = np.isnan(error_map_magnitude)
    if np.any(nans):
        nans = np.where(nans)
        error_map_magnitude[nans] = 0.
        
    hsv[..., 0] = error_map_angle * 180 / np.pi / 2
    hsv[..., 1] = cv.normalize(error_map_magnitude, None, 0, 255, cv.NORM_MINMAX)
    hsv[..., 2] = 255
    error_map = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)
    return error_map, epe, abs_error, error_u , error_v, epe_map


def extract_flow_map(path,w,h,id,flow_method):
    if path.endswith('.png'):
        flow = cv.imread(path, cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
        
        mask = flow[:,:,0]*(1/(2**16 - 1))
        actual_flow = flow[:,:,1:]*(1/(2**16 - 1))
        actual_flow = actual_flow*2 - 1
        actual_flow[:,:,0] = actual_flow[:,:,0]*(w-1)
        actual_flow[:,:,1] = actual_flow[:,:,1]*(h-1)
        
        return actual_flow, mask
    
    elif path.endswith('.npz'):
        # read npz file
        flow = np.load(path)
        # print(np.array(flow))
        flow = np.array(flow[id]).reshape((h,w,2))
        if flow_method == "gt":
            flow[:,:,0] = -1*flow[:,:,0]*(w/2)
            flow[:,:,1] = flow[:,:,1]*(h/2)
        mask = np.ones((w,h))
        return flow, mask
    

result_dir = 'result/MidRainyNoon/'
gt_dir = "datasets/carla/"

for seq in sorted(os.listdir(result_dir)):
    #list all npz file in the result directory
    print(seq)
    # if(seq == "flow_camera_2"):
    list_seq = sorted(os.listdir(result_dir+"/" +seq))
    paths = []
    epes = []
    absolute_errors = []
    for path in list_seq:
        if path.endswith('npz'):
            paths.append(path)
    for i in range(len(paths) - 1):
        pred_flow, pred_mask = extract_flow_map(result_dir+"/" +seq+"/" + paths[i] ,800,600,"flow","pred")
        gt_flow, gt_mask = extract_flow_map(gt_dir+"MidRainyNoon/" + seq +"/flow_npz/" + paths[i] ,800,600,"flow","gt")
        # print(pred_flow.shape, gt_flow.shape)
        error_map, epe, abs_error, error_u, error_v , epe_map = evaluate_flow(pred_flow, gt_flow)
        # print(result_dir+"/" +seq+"/vis/" + paths[i].split(".")[0] + "." + paths[i].split(".")[1] + ".png")
        cv.imwrite(result_dir+"/" +seq+"/vis/" + paths[i].split(".")[0] + "." + paths[i].split(".")[1] + ".png", error_map)
        epes.append(epe)
        absolute_errors.append(abs_error) 
        # # find minimum of minima & maximum of maxima
        minmin = np.min([np.min(pred_flow), np.min(gt_flow)])
        maxmax = np.max([np.max(pred_flow), np.max(gt_flow)])

        # fig, ax = plt.subplots(2,4)
        # ax[0][0].imshow(pred_flow[:,:,0], vmin=minmin, vmax=maxmax)
        # ax[0][0].set_title("pred_flow_u")
        # ax[0][1].imshow(pred_flow[:,:,1], vmin=minmin, vmax=maxmax)
        # ax[0][1].set_title("pred_flow_v")
        # ax[0][2].imshow(gt_flow[:,:,0], vmin=minmin, vmax=maxmax)
        # ax[0][2].set_title("gt_flow_u")
        # ax[0][3].imshow(gt_flow[:,:,1], vmin=minmin, vmax=maxmax)
        # ax[0][3].set_title("gt_flow_v")
        # ax[1][0].imshow(error_u, vmin=minmin, vmax=maxmax)
        # ax[1][0].set_title("error_u")
        # ax[1][1].imshow(error_v, vmin=minmin, vmax=maxmax)
        # ax[1][1].set_title("error_v")
        # ax[1][2].imshow(epe_map)
        # ax[1][2].set_title("epe_map")
        # ax[1][3].imshow(error_map)
        # ax[1][3].set_title("error_map_vis, epe = {}".format(epe))
        # figManager = plt.get_current_fig_manager()
        # figManager.window.showMaximized()
        # plt.show()
    data_error = {
    "EPE" : epes,
    "absolute_error": absolute_errors
    }
    df = pd.DataFrame.from_dict(data_error)
    # df.to_csv(path + "/errors_flow_noc" + "/error_data2.csv")
    print(df.describe())
        
            
            # print(epe, abs_error)
# img1 = cv.imread('/home/sushlok/optical_flow/GMFlowNet/result/vkitti_flow2/00033.png',cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
# /home/sushlok/new_approach/datasets/carla/ClearNoon/flow_camera_-1/flow_npz/23500_2792.8022905162143.npz
# flow_est = img1[:,:,1:]*(1/(2**16 - 1))
# flow_est = flow_est*2 - 1
# flow_est[:,:,0] = flow_est[:,:,0]*(flow_est.shape[1]-1)
# flow_est[:,:,1] = flow_est[:,:,1]*(flow_est.shape[0]-1)
# # print(flow_est.min(),flow_est.max())


# # read npz file
# flow = np.load('/home/sushlok/carla-python/flow_npz/1148_299.25492858454527.npz')
# flow = np.array(flow["arr_0"]).reshape((flow_est.shape[0],flow_est.shape[1],2))

# # print(flow)
# img2 = cv.imread('/home/sushlok/carla-python/flow/0001/flow2/1148_299.25492858454527.png',cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
# img2 = cv.cvtColor(img2, cv.COLOR_BGR2HSV)
# ang = img2[:,:,0]/28.64788975654116
# # print(ang)
# mag = img2[:,:,2]/2550.0
# # print(mag)
# u = mag*np.sin(ang)*(flow_est.shape[0]-1)
# v = mag*np.cos(ang)*(flow_est.shape[1]-1)
# flow_gt = np.stack((u,v),axis=2)
# # print(v.max(),v.min())

# # est_flow = img1[:,:,1:]*(1/(2**16 - 1))
# # print(est_flow.max(),est_flow.min())
# # est_flow = est_flow*2 - 1
# # print(u.max(),u.min())
# fig, ax = plt.subplots(1,3)
# ax[0].imshow(flow_est[:,:,0])
# ax[1].imshow(u)
# ax[2].imshow(abs(flow_est[:,:,0] - u))
# print(np.average(abs(flow_est[:,:,0] - u)))
# # ax[0].imshow(flow_est[:,:,1])
# # ax[1].imshow(v)
# # ax[2].imshow((-1*flow[:,:,0]))
# plt.show()
# print("finally")