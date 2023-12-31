import pandas as pd
import numpy as np
import pickle
import csv
from joblib import Parallel, delayed
from pandas import DataFrame
import itertools
from scipy.ndimage import gaussian_filter
from utils import GaussianMask, Fixpos2Densemap, get_mean, get_all_the_frame, create_the_smoothed_dataset, generate_list

#set the parameters 
x_dimension=128
x_ratio=8
y_dimension=72
y_ratio=6
njobs=12 # number of cores you want to use

file_read = pd.read_csv('your_hand_location_data')
#To optimize the efficiency, we rescale the x and y to 1/10  of the original size
file_read['x_rescaled']=file_read['x']/10
file_read['y_rescaled']=file_read['y']/10

#list of frames
unique_list=np.unique(file_read['frame'].tolist()).tolist()

all_smootehd_pixel=Parallel(n_jobs=njobs)(delayed(create_the_smoothed_dataset)(i,j) for i,j in itertools.product(range(y_dimension), range(x_dimension)))

all_smoothed_pixel_data = {"x":[],"y":[]}

for i,j in itertools.product(range(y_dimension), range(x_dimension)): 
    all_smoothed_pixel_data['y'].append(i)
    all_smoothed_pixel_data['x'].append(j)

all_smoothed_pixel_data['smoothed']=all_smootehd_pixel
 

all_smoothed_pixel_data = all_smoothed_pixel_data.explode('smoothed').reset_index(drop=True)
all_smoothed_pixel_data['frame']=unique_list * x_dimension*y_dimension

all_smoothed_pixel_data ['x'] = (all_smoothed_pixel_data ['x'] ) // 2*x_ratio+1
all_smoothed_pixel_data ['y'] = (all_smoothed_pixel_data ['y']) // 2*y_ratio+1
all_smoothed_pixel_data  = all_smoothed_pixel_data .groupby(['frame',"x","y"], as_index=False).mean() 
all_smoothed_pixel_data  = all_smoothed_pixel_data .rename(columns={'x': 'gridx', 'y': 'gridy', 'smoothed': 'hand_value'})
all_smoothed_pixel_data.to_csv("hand_location_output",index=False)

