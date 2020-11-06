#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sys import argv
from time import time
from noiseprint.noiseprint import genNoiseprint
from noiseprint.utility.utilityRead import imread2f
from noiseprint.utility.utilityRead import jpeg_qtableinv
import numpy as np
import cv2


# In[2]:




imgfilename = 'C:\\Users\\rotar\\OneDrive\\Desktop\\Master\\UNITN\\Semester 1\\Multimedia Data Security\\2nd Competition\\dev-dataset\\dev-dataset-forged\\dev_0414.tif'
outfilename = 'out'

timestamp = time()
img, mode = imread2f(imgfilename, channel=1)
try:
    QF = jpeg_qtableinv(strimgfilenameeam)
except:
    QF = 200
    
res = genNoiseprint(img,QF)
timeApproach = time() - timestamp

out_dict = dict()
out_dict['noiseprint'] = res
out_dict['QF'] = QF
out_dict['time'] = timeApproach

if outfilename[-4:] == '.mat':
    import scipy.io as sio
    sio.savemat(outfilename, out_dict)
else:
    import numpy as np
    np.savez(outfilename, **out_dict)


# In[3]:


import pickle
import numpy as numpy
file = open('store.pckl', 'rb')
prev_prnus = pickle.load(file)


# In[4]:


import math
def mean2(x):
    y = np.sum(x) / np.size(x);
    return y

def corr2(a,b):
    a = a - mean2(a)
    b = b - mean2(b)

    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum());
    return r


# In[5]:


def get_camera_prnu(camera_nr: int) -> np.ndarray:
    path = 'C:\\Users\\rotar\\OneDrive\\Desktop\\Master\\UNITN\\Semester 1\\Multimedia Data Security\\2nd Competition\\dev-dataset\\'
    index = '000'
    noises = []
    
    for i in range(100):
        index ="%03d" % (i+1)
        img, mode = imread2f(path + 'flat-camera-'+ str(camera_nr) + '\\flat_c'+ str(camera_nr) + '_' + index + '.tif', channel=1)
        try:
            QF = jpeg_qtableinv(strimgfilenameeam)
        except:
            QF = 200

        res = genNoiseprint(img,QF)
        noises.append(res)
        print(i)
      
    final = numpy.mean(noises,0)
    return final


# In[5]:


prnu1 = get_camera_prnu(1)
f = open('prnu1_cnn.pckl', 'wb')
pickle.dump(prnu1, f)
f.close()


# In[75]:


prnu2 = get_camera_prnu(2)
f = open('prnu2_cnn.pckl', 'wb')
pickle.dump(prnu2, f)
f.close()


# In[76]:


prnu3 = get_camera_prnu(3)
f = open('prnu3_cnn.pckl', 'wb')
pickle.dump(prnu3, f)
f.close()


# In[77]:


prnu4 = get_camera_prnu(4)
f = open('prnu4_cnn.pckl', 'wb')
pickle.dump(prnu4, f)
f.close()


# In[8]:


import pickle
file1 = open('prnu1_cnn.pckl', 'rb')
file2 = open('prnu2_cnn.pckl', 'rb')
file3 = open('prnu3_cnn.pckl', 'rb')
file4 = open('prnu4_cnn.pckl', 'rb')

prnu1 = pickle.load(file1)
prnu2 = pickle.load(file2)
prnu3 = pickle.load(file3)
prnu4 = pickle.load(file4)

file1.close()
file2.close()
file3.close()
file4.close()


# In[ ]:


from functions import *
from PIL import Image
from multiprocessing import Pool, cpu_count

import numpy as np
from PIL import Image
import pywt
from numpy.fft import fft2, ifft2
from scipy.ndimage import filters
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

im = Image.open('C:\\Users\\rotar\\OneDrive\\Desktop\\Master\\UNITN\\Semester 1\\Multimedia Data Security\\2nd Competition\\dev-dataset\\dev-dataset-forged\\dev_0001.tif')


imarray = np.array(im)

print(type(imarray))

noise = noise_extract(imarray)


# In[1]:


from PIL import Image
filename = '../dev-dataset/dev-dataset-forged\dev_0002.jpg'

try:
    test_prnus_file = open('test_prnus.pckl', 'rb')
    test_prnus = pickle.load(test_prnus_file)
except:
    test_prnus = {}

if(filename in test_prnus):
    noise = test_prnus[filename]
else:
    img, _ = imread2f(filename)
    noise = genNoiseprint(img,200)
    test_prnus[filename] = noise
    file_write = open('test_prnus.pckl', 'wb')
    pickle.dump(test_prnus, file_write)
    file_write.close()
        
#img, mode = imread2f('C:\\Users\\rotar\\OneDrive\\Desktop\\Master\\UNITN\\Semester 1\\Multimedia Data Security\\2nd Competition\\dev-dataset\\demo_images\\demo_1.jpg')


# In[57]:


import numpy as np
print(corr2(noise, prev_prnus[0][:,:,1]))
print(corr2(noise, prev_prnus[1][:,:,1]))
print(corr2(noise, prev_prnus[2][:,:,1]))
print(corr2(noise, prev_prnus[3][:,:,1]))


# In[11]:


def get_map(prnu : np.ndarray or list, noise: np.ndarray or list) -> np.ndarray:
    
    diff_map = numpy.zeros_like(prnu)
    for i in range(len(prnu)):
        for j in range(len(prnu[0])):
            diff_map[i][j] = abs(prnu[i][j] - noise[i][j]) * 75
            
    return diff_map


# In[12]:


def is_inside(i: int, j: int, rows: int, cols: int):
    if (i>=0 and j >=0 and i<rows and j<cols): return True
    return False


# In[33]:


def get_map_segment(prnu : np.ndarray or list, noise: np.ndarray or list, rows: int, cols: int, thr: float, clear: bool) -> np.ndarray:
    row_inc = prnu.shape[0]/rows
    col_inc = prnu.shape[1]/cols
    
    avg = np.average(np.absolute(prnu-noise))
    
    diff_map = diff_map = np.zeros_like(prnu, np.uint8)
    for i in range(rows):
        for j in range(cols):
            i1 = int(i*row_inc)
            i2 = int((i+1)*row_inc)
            j1 = int(j*col_inc)
            j2 = int((j+1)*col_inc)
            
            m = np.mean(np.absolute(prnu[i1:i2, j1:j2] - noise[i1:i2, j1:j2]))
            if(m > avg*1.1):
                diff_map[i1:i2, j1:j2].fill(255)
            else:
                diff_map[i1:i2, j1:j2].fill(0)
                
    if(clear):
        for i in range(rows):
            for j in range(cols):
                i1 = int(i*row_inc)
                i2 = int((i+1)*row_inc)
                j1 = int(j*col_inc)
                j2 = int((j+1)*col_inc)

                if(is_inside(i1-1,j1, prnu.shape[0], prnu.shape[1]) and diff_map[i1-1][j1] != diff_map[i1][j1] and 
                   is_inside(i2+1,j1, prnu.shape[0], prnu.shape[1]) and diff_map[i2+1][j1] != diff_map[i1][j1] and 
                   is_inside(i1,j2+1, prnu.shape[0], prnu.shape[1]) and diff_map[i1][j2+1] != diff_map[i1][j1] and 
                   is_inside(i1,j1-1, prnu.shape[0], prnu.shape[1]) and diff_map[i1][j1-1] != diff_map[i1][j1]):
                    diff_map[i1:i2, j1:j2].fill(abs(diff_map[i1][j1] - 255))
            
    return diff_map


# In[67]:


diff = get_map_segment(noise, prnu3, 150, 200, 0.5, True )
#diff = get_map(noise, prnu4)
diff = cv2.medianBlur(diff, 17)
img = Image.fromarray(diff)
img.show()


# In[82]:



kernel = np.ones((29,29),np.uint8)
kernel1 = np.ones((13,13),np.uint8)
#erosion = cv2.erode(diff,kernel,iterations = 2)
erosion = cv2.erode(diff,kernel1,iterations = 1)
erosion = cv2.dilate(erosion,kernel1,iterations = 1)
erosion = cv2.dilate(erosion,kernel,iterations = 2)
erosion = cv2.erode(erosion,kernel,iterations = 2)
img = Image.fromarray(erosion)
img.show()
#cv2.imshow("res", diff)


# In[22]:


def inverse_bits(temper_map) -> np.ndarray:
    return np.absolute(temper_map - 255)*255


# In[19]:


import imageio 
import numpy as np



def f_measure(map_gt,map_est):
    if not map_gt.shape == map_est.shape:
        print('The compared maps must have the same size')

    # Vectorize maps
    map_gt = np.ndarray.flatten(map_gt)
    map_est = np.ndarray.flatten(map_est)

    # Number of pixels
    N = map_gt.size
    # Indices of forged pixels in the ground truth
    i_pos = np.where(map_gt == 1)
    # Number of forged pixels in the ground truth
    n_pos = map_gt[i_pos].shape[0]

    # True Positive Rate: fraction of forged pixels correctly identified as forged
    tp = np.where(map_gt[i_pos]==map_est[i_pos])[0].shape[0]
    tpr = tp / n_pos
    # False Negative Rate: fraction of forged pixels wrongly identified as non-forged
    fn = n_pos-tp
    fnr = fn / n_pos
    # False Positive Rate: fraction of non-forged pixels wrongly identified as forged
    # Indices of non-forged pixels in the ground truth
    i_neg = np.where(map_gt == 0)
    fp = np.where(map_est[i_neg]==1)[0].shape[0]
    fpr = fp / (N-n_pos)

    F = 2*tp / (2*tp+fn+fp)

    return F, tp, fn, fp


# In[2]:


map_gt = imageio.imread('C:\\Users\\rotar\\OneDrive\\Desktop\\Master\\UNITN\\Semester 1\\Multimedia Data Security\\2nd Competition\\dev-dataset\\dev-dataset-maps\\dev_0004.bmp')
map_gt = np.array(map_gt/255,dtype=np.uint8)

#map_est = np.array(erosion/255,dtype=np.uint8)
map_est = np.array(inverse_bits(erosion)/255,dtype=np.uint8)

F,_,_,_ = f_measure(map_gt,map_est)
print(F)


# In[7]:


import imageio
map_gt = imageio.imread('C:\\Users\\rotar\\OneDrive\\Desktop\\Master\\UNITN\\Semester 1\\Multimedia Data Security\\2nd Competition\\dev-dataset\\dev-dataset-maps\\dev_0004.bmp')

map_gt.shape


# In[ ]:




