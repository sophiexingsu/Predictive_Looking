import pandas as pd
import numpy as np
import pickle
import csv
from joblib import Parallel, delayed
from pandas import DataFrame
import itertools
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt


def GaussianMask(sizex,sizey, sigma, center=None, fix=1):
    """
    sizex  : mask width
    sizey  : mask height
    sigma  : gaussian Sd
    center : gaussian mean
    fix    : gaussian max
    return gaussian mask
    """
    x = np.arange(0, sizex, 1, float)
    y = np.arange(0, sizey, 1, float)
    x, y = np.meshgrid(x, y)
    if center is None:
        x0 = sizex // 2
        y0 = sizey // 2
    else:
        if np.isnan(center[0]) == False and np.isnan(center[1]) == False:
            x0 = center[0]
            y0 = center[1]
        else:
            return np.zeros((sizey, sizex))
    return fix*np.exp(-((x-x0)**2/sigma**2+(y-y0)**2/sigma**2))


def Fixpos2Densemap(fix_arr, width, height):
    """
    fix_arr   : fixation array number of subjects x 3(x,y,fixation)
    width     : output image width
    height    : output image height
    imgfile   : image file (optional)
    alpha     : marge rate imgfile and heatmap (optional)
   threshold : heatmap threshold(0~255)
    return heatmap
    """
    heatmap = np.zeros((height, width), np.float32)
    for n_subject in range(fix_arr.shape[0]):
        center=(fix_arr[n_subject, 4], fix_arr[n_subject, 5]) # this need to be changed based on the x and y axis of the things
        place=GaussianMask(width,height,5,center=center)
        heatmap=heatmap+place
    heatmap=heatmap/fix_arr.shape[0]
    return heatmap 

def get_mean(array,xindex, yindex):
  temp=np.hsplit(np.vsplit(array, 6)[yindex], 8)[xindex]
  np.mean(temp)
  return(np.mean(temp))


def get_all_the_frame(i,j,f): 
    """  
    f: the frame
    file_read:
    sigma: the temporal kernel 
    return: the smoothed out pixel across the temporal frame
    """
    pixel_list=[]
    frame=file_read[file_read["frame"]==f]
    values=frame.to_numpy()
    all_temp=Fixpos2Densemap(values, 128, 72)
    pixel=all_temp[i,j]
    #pixel_list.append(pixel)
    #frame_list.append(f)
    #smoothed_pixel_list = gaussian_filter(pixel_list, sigma=sigma) 
    return(pixel)
def create_the_smoothed_dataset(i,j,sigma=10):
    #all_smoothed_pixel_list = {"x":[],"y":[],"smoothed":[],"frame":[]}
    pixel_by_frame=Parallel(n_jobs=24)(delayed(get_all_the_frame)(i,j,f) for f in unique_list)
    smoothed_pixel_by_frame=gaussian_filter(pixel_by_frame, sigma=10, mode='nearest')
    #gaussian_kernel = gaussian_filter1d(pixel_by_frame, sigma=60, mode='nearest')
    f = interp1d(unique_list, smoothed_pixel_by_frame, kind='linear', fill_value='extrapolate')
    sampled_pixel_by_frame = f(sampled_list)
    #all_smoothed_pixel_list['x'].append(j)
    #all_smoothed_pixel_list['y'].append(i)
    #all_smoothed_pixel_list['smoothed'].append(smolsothed_pixel_by_frame)
    return(sampled_pixel_by_frame)
def generate_list(n):
    # Start with 30 and add 60 to each subsequent number until we reach n.
    num = 30
    lst = [num]
    while num < n:
        num += 60
        lst.append(num)

    # Determine which of the last two numbers in the list is closer to n.
    last_num = lst[-1]
    prev_num = lst[-2]
    if abs(last_num - n) < abs(prev_num - n):
        # If the last number is closer to n, return the full list.
        return lst
    else:
        # If the second-to-last number is closer to n, remove the last number and add the second-to-last.
        return lst[:-1] + [prev_num]

# Test the function with n = 325.
#n = 325
#result = generate_list(n)

file_read = pd.read_csv('movie_639_hand_openpose.csv')
file_read['x_rescaled']=file_read['x']/10
file_read['y_rescaled']=file_read['y']/10

frame_list = file_read['frame'].tolist()
unique_list=np.unique(frame_list).tolist()  # all the frame list 
sampled_list=generate_list(max(unique_list)) # do the down sampling in the stadnarzied way. 
#print(sampled_list)



all_smootehd_pixel=Parallel(n_jobs=24)(delayed(create_the_smoothed_dataset)(i,j) for i,j in itertools.product(range(72), range(128)))

all_smoothed_pixel_data = {"x":[],"y":[]}

for i,j in itertools.product(range(72), range(128)): 
    all_smoothed_pixel_data['y'].append(i)
    all_smoothed_pixel_data['x'].append(j)

all_smoothed_pixel_data['smoothed']=all_smootehd_pixel
#len(index_data['smoothed'][1])
with open('all_smootehd_pixel_639_smoothed60_sampled.pickle', 'wb') as f:
    pickle.dump(all_smoothed_pixel_data, f)
