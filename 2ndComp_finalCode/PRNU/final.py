#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
import math
import os


# In[29]:


def get_map(filename):
    from PIL import Image
    from noiseprint.noiseprint import genNoiseprint
    from noiseprint.utility.utilityRead import imread2f
    import pickle
    import io
    import sys
    import cv2

    text_trap = io.StringIO()
    sys.stdout = text_trap

    file1 = open('prnu1_cnn.pckl', 'rb')
    file2 = open('prnu2_cnn.pckl', 'rb')
    file3 = open('prnu3_cnn.pckl', 'rb')
    file4 = open('prnu4_cnn.pckl', 'rb')

    prnu1 = pickle.load(file1)
    prnu2 = pickle.load(file2)
    prnu3 = pickle.load(file3)
    prnu4 = pickle.load(file4)

    prnu = [prnu1, prnu2, prnu3, prnu4]

    file1.close()
    file2.close()
    file3.close()
    file4.close()

    img, _ = imread2f(filename)
    noise = genNoiseprint(img, 200)

    corrs = []
    corrs.append(corr2(noise, prnu1))
    corrs.append(corr2(noise, prnu2))
    corrs.append(corr2(noise, prnu3))
    corrs.append(corr2(noise, prnu4))

    max_val = max(corrs)
    index = corrs.index(max_val)

    diff = get_map_segment(noise, prnu[index], 150, 200, 0.5, True)

    kernel = np.ones((29, 29), np.uint8)
    erosion = cv2.dilate(diff, kernel, iterations=3)
    erosion = cv2.erode(erosion, kernel, iterations=3)
    img = Image.fromarray(erosion)
    base_filename = os.path.basename(filename)
    base_filename_without_ext = os.path.splitext(base_filename)
    img.save('../../DEMO-RESULTS/' + base_filename_without_ext[0] + '.bmp')

    sys.stdout = sys.__stdout__


# In[23]:


def mean2(x):
    y = np.sum(x) / np.size(x)
    return y


def corr2(a, b):
    a = a - mean2(a)
    b = b - mean2(b)

    r = (a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum())
    return r


def is_inside(i: int, j: int, rows: int, cols: int):
    if (i >= 0 and j >= 0 and i < rows and j < cols): return True
    return False


# In[26]:


def get_map_segment(prnu: np.ndarray or list, noise: np.ndarray or list, rows: int, cols: int, thr: float,
                    clear: bool) -> np.ndarray:
    row_inc = prnu.shape[0] / rows
    col_inc = prnu.shape[1] / cols

    avg = np.average(np.absolute(prnu - noise))

    diff_map = np.zeros_like(prnu, np.uint8)
    for i in range(rows):
        for j in range(cols):
            i1 = int(i * row_inc)
            i2 = int((i + 1) * row_inc)
            j1 = int(j * col_inc)
            j2 = int((j + 1) * col_inc)

            m = np.mean(np.absolute(prnu[i1:i2, j1:j2] - noise[i1:i2, j1:j2]))
            if (m > avg * 1.1):
                diff_map[i1:i2, j1:j2].fill(255)
            else:
                diff_map[i1:i2, j1:j2].fill(0)

    if (clear):
        for i in range(rows):
            for j in range(cols):
                i1 = int(i * row_inc)
                i2 = int((i + 1) * row_inc)
                j1 = int(j * col_inc)
                j2 = int((j + 1) * col_inc)

                if (is_inside(i1 - 1, j1, prnu.shape[0], prnu.shape[1]) and diff_map[i1 - 1][j1] != diff_map[i1][j1] and
                        is_inside(i2 + 1, j1, prnu.shape[0], prnu.shape[1]) and diff_map[i2 + 1][j1] != diff_map[i1][
                            j1] and
                        is_inside(i1, j2 + 1, prnu.shape[0], prnu.shape[1]) and diff_map[i1][j2 + 1] != diff_map[i1][
                            j1] and
                        is_inside(i1, j1 - 1, prnu.shape[0], prnu.shape[1]) and diff_map[i1][j1 - 1] != diff_map[i1][
                            j1]):
                    diff_map[i1:i2, j1:j2].fill(abs(diff_map[i1][j1] - 255))

    return diff_map


# In[30]:


if __name__ == '__main__':
    import sys
    path = sys.argv[1]
    get_map(path)

# In[ ]:


# In[ ]:
