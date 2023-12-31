import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import itertools
from pandas import DataFrame
from utils import Fixpos2Densemap, get_mean 

#set the parameters 
y_dimension=720 
x_dimension=1280
y_grid_number=6 
x_grid_number=8 
number_of_jobs=100

#read in the gaze location file
file_read = pd.read_csv("your_gaze_location_file.csv")
#read in the number of frames in the movie
frame_read=pd.read_csv("list_frames_in_the_movie.csv")
unique_list=frame_read["frame"].unique()
whole = pd.DataFrame()
def save_saliency(f):
    all_value = []
    frame=file_read[file_read["frame"]==f]
    new_frame = frame.groupby(['sub']).mean()
    values = new_frame.to_numpy()
    all_temp=Fixpos2Densemap(values, x_dimension, y_dimension)
    for i, j in itertools.product(range(x_grid_number), range(y_grid_number)):
         temp = get_mean(all_temp, i, j)
         temp_data = [i + 1,j + 1, temp]
         all_value.append(temp_data) 
    return all_value
res=Parallel(n_jobs=number_of_job)(delayed(save_saliency)(f) for f in unique_list)
res=np.array(res)
df = pd.DataFrame(res.reshape(-1, 3), columns=["gridx","gridy","gaze_value"])
df["frame"]= [element for element in unique_list for i in range(y_grid_number*x_grid_number)]
df.to_csv("your_gaze_grid.csv",index=False)