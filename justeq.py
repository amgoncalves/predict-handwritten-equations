import numpy as np
import sys
import os
from skimage import io

test_path = './annotated/'
res_path = './annotated2/'
#save all file names in an array. Make sure the extension is .png
file_list = [name for name in os.listdir(test_path) if len(name.split("_"))==3] 
for image_filename in file_list: 
  img = io.imread(test_path + "/" + image_filename) 
  io.imsave(res_path + "/" + image_filename,img)

