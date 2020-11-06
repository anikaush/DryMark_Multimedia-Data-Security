#!/usr/bin/env python
# coding: utf-8

# In[1]:


from multiprocessing import Pool, cpu_count

import numpy as np
from PIL import Image
import pywt
from numpy.fft import fft2, ifft2
from scipy.ndimage import filters
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm


# In[2]:



class ArgumentError(Exception):
    pass


"""
Extraction functions
"""


def extract_single(im: np.ndarray,
                   levels: int = 4,
                   sigma: float = 5,
                   wdft_sigma: float = 0) -> np.ndarray:
    """
    Extract noise residual from a single image
    :param im: grayscale or color image, np.uint8
    :param levels: number of wavelet decomposition levels
    :param sigma: estimated noise power
    :param wdft_sigma: estimated DFT noise power
    :return: noise residual
    """

    W = noise_extract(im, levels, sigma)
    W = rgb2gray(W)
    W = zero_mean_total(W)
    W_std = W.std(ddof=1) if wdft_sigma == 0 else wdft_sigma
    W = wiener_dft(W, W_std).astype(np.float32)

    return W


def noise_extract(im: np.ndarray, levels: int = 4, sigma: float = 5) -> np.ndarray:
    """
    NoiseExtract as from Binghamton toolbox.
    :param im: grayscale or color image, np.uint8
    :param levels: number of wavelet decomposition levels
    :param sigma: estimated noise power
    :return: noise residual
    """

    assert (im.dtype == np.uint8)
    assert (im.ndim in [2, 3])

    im = im.astype(np.float32)

    noise_var = sigma ** 2

    if im.ndim == 2:
        im.shape += (1,)

    W = np.zeros(im.shape, np.float32)

    for ch in range(im.shape[2]):

        wlet = None
        while wlet is None and levels > 0:
            try:
                wlet = pywt.wavedec2(im[:, :, ch], 'db4', level=levels)
            except ValueError:
                levels -= 1
                wlet = None
        if wlet is None:
            raise ValueError('Impossible to compute Wavelet filtering for input size: {}'.format(im.shape))

        wlet_details = wlet[1:]

        wlet_details_filter = [None] * len(wlet_details)
        # Cycle over Wavelet levels 1:levels-1
        for wlet_level_idx, wlet_level in enumerate(wlet_details):
            # Cycle over H,V,D components
            level_coeff_filt = [None] * 3
            for wlet_coeff_idx, wlet_coeff in enumerate(wlet_level):
                level_coeff_filt[wlet_coeff_idx] = wiener_adaptive(wlet_coeff, noise_var)
            wlet_details_filter[wlet_level_idx] = tuple(level_coeff_filt)

        # Set filtered detail coefficients for Levels > 0 ---
        wlet[1:] = wlet_details_filter

        # Set to 0 all Level 0 approximation coefficients ---
        wlet[0][...] = 0

        # Invert wavelet transform ---
        wrec = pywt.waverec2(wlet, 'db4')
        try:
            W[:, :, ch] = wrec
        except ValueError:
            W = np.zeros(wrec.shape[:2] + (im.shape[2],), np.float32)
            W[:, :, ch] = wrec

    if W.shape[2] == 1:
        W.shape = W.shape[:2]

    W = W[:im.shape[0], :im.shape[1]]

    return W


def noise_extract_compact(args):
    """
    Extract residual, multiplied by the image. Useful to save memory in multiprocessing operations
    :param args: (im, levels, sigma), see noise_extract for usage
    :return: residual, multiplied by the image
    """
    w = noise_extract(*args)
    im = args[0]
    return (w * im / 255.).astype(np.float32)


def extract_multiple_aligned(imgs: list, levels: int = 4, sigma: float = 5, processes: int = None,
                             batch_size=cpu_count(), tqdm_str: str = '') -> np.ndarray:
    """
    Extract PRNU from a list of images. Images are supposed to be the same size and properly oriented
    :param tqdm_str: tqdm description (see tqdm documentation)
    :param batch_size: number of parallel processed images
    :param processes: number of parallel processes
    :param imgs: list of images of size (H,W,Ch) and type np.uint8
    :param levels: number of wavelet decomposition levels
    :param sigma: estimated noise power
    :return: PRNU
    """
    assert (isinstance(imgs[0], np.ndarray))
    assert (imgs[0].ndim == 3)
    assert (imgs[0].dtype == np.uint8)

    h, w, ch = imgs[0].shape

    RPsum = np.zeros((h, w, ch), np.float32)
    NN = np.zeros((h, w, ch), np.float32)

    if processes is None or processes > 1:
        args_list = []
        for im in imgs:
            args_list += [(im, levels, sigma)]
        pool = Pool(processes=processes)

        for batch_idx0 in tqdm(np.arange(start=0, step=batch_size, stop=len(imgs)), disable=tqdm_str == '',
                               desc=(tqdm_str + ' (1/2)'), dynamic_ncols=True):
            nni = pool.map(inten_sat_compact, args_list[batch_idx0:batch_idx0 + batch_size])
            for ni in nni:
                NN += ni
            del nni

        for batch_idx0 in tqdm(np.arange(start=0, step=batch_size, stop=len(imgs)), disable=tqdm_str == '',
                               desc=(tqdm_str + ' (2/2)'), dynamic_ncols=True):
            wi_list = pool.map(noise_extract_compact, args_list[batch_idx0:batch_idx0 + batch_size])
            for wi in wi_list:
                RPsum += wi
            del wi_list

        pool.close()

    else:  # Single process
        for im in tqdm(imgs, disable=tqdm_str is None, desc=tqdm_str, dynamic_ncols=True):
            RPsum += noise_extract_compact((im, levels, sigma))
            NN += (inten_scale(im) * saturation(im)) ** 2

    K = RPsum / (NN + 1)
    K = rgb2gray(K)
    K = zero_mean_total(K)
    K = wiener_dft(K, K.std(ddof=1)).astype(np.float32)

    return K


def cut_ctr(array: np.ndarray, sizes: tuple) -> np.ndarray:
    """
    Cut a multi-dimensional array at its center, according to sizes
    :param array: multidimensional array
    :param sizes: tuple of the same length as array.ndim
    :return: multidimensional array, center cut
    """
    array = array.copy()
    if not (array.ndim == len(sizes)):
        raise ArgumentError('array.ndim must be equal to len(sizes)')
    for axis in range(array.ndim):
        axis_target_size = sizes[axis]
        axis_original_size = array.shape[axis]
        if axis_target_size > axis_original_size:
            raise ValueError(
                'Can\'t have target size {} for axis {} with original size {}'.format(axis_target_size, axis,
                                                                                      axis_original_size))
        elif axis_target_size < axis_original_size:
            axis_start_idx = (axis_original_size - axis_target_size) // 2
            axis_end_idx = axis_start_idx + axis_target_size
            array = np.take(array, np.arange(axis_start_idx, axis_end_idx), axis)
    return array


def wiener_dft(im: np.ndarray, sigma: float) -> np.ndarray:
    """
    Adaptive Wiener filter applied to the 2D FFT of the image
    :param im: multidimensional array
    :param sigma: estimated noise power
    :return: filtered version of input im
    """
    noise_var = sigma ** 2
    h, w = im.shape

    im_noise_fft = fft2(im)
    im_noise_fft_mag = np.abs(im_noise_fft / (h * w) ** .5)

    im_noise_fft_mag_noise = wiener_adaptive(im_noise_fft_mag, noise_var)

    zeros_y, zeros_x = np.nonzero(im_noise_fft_mag == 0)

    im_noise_fft_mag[zeros_y, zeros_x] = 1
    im_noise_fft_mag_noise[zeros_y, zeros_x] = 0

    im_noise_fft_filt = im_noise_fft * im_noise_fft_mag_noise / im_noise_fft_mag
    im_noise_filt = np.real(ifft2(im_noise_fft_filt))

    return im_noise_filt.astype(np.float32)


def zero_mean(im: np.ndarray) -> np.ndarray:
    """
    ZeroMean called with the 'both' argument, as from Binghamton toolbox.
    :param im: multidimensional array
    :return: zero mean version of input im
    """
    # Adapt the shape ---
    if im.ndim == 2:
        im.shape += (1,)

    h, w, ch = im.shape

    # Subtract the 2D mean from each color channel ---
    ch_mean = im.mean(axis=0).mean(axis=0)
    ch_mean.shape = (1, 1, ch)
    i_zm = im - ch_mean

    # Compute the 1D mean along each row and each column, then subtract ---
    row_mean = i_zm.mean(axis=1)
    col_mean = i_zm.mean(axis=0)

    row_mean.shape = (h, 1, ch)
    col_mean.shape = (1, w, ch)

    i_zm_r = i_zm - row_mean
    i_zm_rc = i_zm_r - col_mean

    # Restore the shape ---
    if im.shape[2] == 1:
        i_zm_rc.shape = im.shape[:2]

    return i_zm_rc


def zero_mean_total(im: np.ndarray) -> np.ndarray:
    """
    ZeroMeanTotal as from Binghamton toolbox.
    :param im: multidimensional array
    :return: zero mean version of input im
    """
    im[0::2, 0::2] = zero_mean(im[0::2, 0::2])
    im[1::2, 0::2] = zero_mean(im[1::2, 0::2])
    im[0::2, 1::2] = zero_mean(im[0::2, 1::2])
    im[1::2, 1::2] = zero_mean(im[1::2, 1::2])
    return im


def rgb2gray(im: np.ndarray) -> np.ndarray:
    """
    RGB to gray as from Binghamton toolbox.
    :param im: multidimensional array
    :return: grayscale version of input im
    """
    rgb2gray_vector = np.asarray([0.29893602, 0.58704307, 0.11402090]).astype(np.float32)
    rgb2gray_vector.shape = (3, 1)

    if im.ndim == 2:
        im_gray = np.copy(im)
    elif im.shape[2] == 1:
        im_gray = np.copy(im[:, :, 0])
    elif im.shape[2] == 3:
        w, h = im.shape[:2]
        im = np.reshape(im, (w * h, 3))
        im_gray = np.dot(im, rgb2gray_vector)
        im_gray.shape = (w, h)
    else:
        raise ValueError('Input image must have 1 or 3 channels')

    return im_gray.astype(np.float32)


def threshold(wlet_coeff_energy_avg: np.ndarray, noise_var: float) -> np.ndarray:
    """
    Noise variance theshold as from Binghamton toolbox.
    :param wlet_coeff_energy_avg:
    :param noise_var:
    :return: noise variance threshold
    """
    res = wlet_coeff_energy_avg - noise_var
    return (res + np.abs(res)) / 2


def wiener_adaptive(x: np.ndarray, noise_var: float, **kwargs) -> np.ndarray:
    """
    WaveNoise as from Binghamton toolbox.
    Wiener adaptive flter aimed at extracting the noise component
    For each input pixel the average variance over a neighborhoods of different window sizes is first computed.
    The smaller average variance is taken into account when filtering according to Wiener.
    :param x: 2D matrix
    :param noise_var: Power spectral density of the noise we wish to extract (S)
    :param window_size_list: list of window sizes
    :return: wiener filtered version of input x
    """
    window_size_list = list(kwargs.pop('window_size_list', [3, 5, 7, 9]))

    energy = x ** 2

    avg_win_energy = np.zeros(x.shape + (len(window_size_list),))
    for window_idx, window_size in enumerate(window_size_list):
        avg_win_energy[:, :, window_idx] = filters.uniform_filter(energy,
                                                                  window_size,
                                                                  mode='constant')

    coef_var = threshold(avg_win_energy, noise_var)
    coef_var_min = np.min(coef_var, axis=2)

    x = x * noise_var / (coef_var_min + noise_var)

    return x


def inten_scale(im: np.ndarray) -> np.ndarray:
    """
    IntenScale as from Binghamton toolbox
    :param im: type np.uint8
    :return: intensity scaled version of input x
    """

    assert (im.dtype == np.uint8)

    T = 252
    v = 6
    out = np.exp(-1 * (im - T) ** 2 / v)
    out[im < T] = im[im < T] / T

    return out


def saturation(im: np.ndarray) -> np.ndarray:
    """
    Saturation as from Binghamton toolbox
    :param im: type np.uint8
    :return: saturation map from input im
    """
    assert (im.dtype == np.uint8)

    if im.ndim == 2:
        im.shape += (1,)

    h, w, ch = im.shape

    if im.max() < 250:
        return np.ones((h, w, ch))

    im_h = im - np.roll(im, (0, 1), (0, 1))
    im_v = im - np.roll(im, (1, 0), (0, 1))
    satur_map =         np.bitwise_not(
            np.bitwise_and(
                np.bitwise_and(
                    np.bitwise_and(
                        im_h != 0, im_v != 0
                    ), np.roll(im_h, (0, -1), (0, 1)) != 0
                ), np.roll(im_v, (-1, 0), (0, 1)) != 0
            )
        )

    max_ch = im.max(axis=0).max(axis=0)

    for ch_idx, max_c in enumerate(max_ch):
        if max_c > 250:
            satur_map[:, :, ch_idx] =                 np.bitwise_not(
                    np.bitwise_and(
                        im[:, :, ch_idx] == max_c, satur_map[:, :, ch_idx]
                    )
                )

    return satur_map


def inten_sat_compact(args):
    """
    Memory saving version of inten_scale followed by saturation. Useful for multiprocessing
    :param args:
    :return: intensity scale and saturation of input
    """
    im = args[0]
    return ((inten_scale(im) * saturation(im)) ** 2).astype(np.float32)


"""
Cross-correlation functions
"""


def crosscorr_2d(k1: np.ndarray, k2: np.ndarray) -> np.ndarray:
    """
    PRNU 2D cross-correlation
    :param k1: 2D matrix of size (h1,w1)
    :param k2: 2D matrix of size (h2,w2)
    :return: 2D matrix of size (max(h1,h2),max(w1,w2))
    """
    assert (k1.ndim == 2)
    assert (k2.ndim == 2)

    max_height = max(k1.shape[0], k2.shape[0])
    max_width = max(k1.shape[1], k2.shape[1])

    k1 -= k1.flatten().mean()
    k2 -= k2.flatten().mean()

    k1 = np.pad(k1, [(0, max_height - k1.shape[0]), (0, max_width - k1.shape[1])], mode='constant', constant_values=0)
    k2 = np.pad(k2, [(0, max_height - k2.shape[0]), (0, max_width - k2.shape[1])], mode='constant', constant_values=0)

    k1_fft = fft2(k1, )
    k2_fft = fft2(np.rot90(k2, 2), )

    return np.real(ifft2(k1_fft * k2_fft)).astype(np.float32)


def aligned_cc(k1: np.ndarray, k2: np.ndarray) -> dict:
    """
    Aligned PRNU cross-correlation
    :param k1: (n1,nk) or (n1,nk1,nk2,...)
    :param k2: (n2,nk) or (n2,nk1,nk2,...)
    :return: {'cc':(n1,n2) cross-correlation matrix,'ncc':(n1,n2) normalized cross-correlation matrix}
    """

    # Type cast
    k1 = np.array(k1).astype(np.float32)
    k2 = np.array(k2).astype(np.float32)

    ndim1 = k1.ndim
    ndim2 = k2.ndim
    assert (ndim1 == ndim2)

    k1 = np.ascontiguousarray(k1).reshape(k1.shape[0], -1)
    k2 = np.ascontiguousarray(k2).reshape(k2.shape[0], -1)

    assert (k1.shape[1] == k2.shape[1])

    k1_norm = np.linalg.norm(k1, ord=2, axis=1, keepdims=True)
    k2_norm = np.linalg.norm(k2, ord=2, axis=1, keepdims=True)

    k2t = np.ascontiguousarray(k2.transpose())

    cc = np.matmul(k1, k2t).astype(np.float32)
    ncc = (cc / (k1_norm * k2_norm.transpose())).astype(np.float32)

    return {'cc': cc, 'ncc': ncc}


def pce(cc: np.ndarray, neigh_radius: int = 2) -> dict:
    """
    PCE position and value
    :param cc: as from crosscorr2d
    :param neigh_radius: radius around the peak to be ignored while computing floor energy
    :return: {'peak':(y,x), 'pce': peak to floor ratio, 'cc': cross-correlation value at peak position
    """
    assert (cc.ndim == 2)
    assert (isinstance(neigh_radius, int))

    out = dict()

    max_idx = np.argmax(cc.flatten())
    max_y, max_x = np.unravel_index(max_idx, cc.shape)

    peak_height = cc[max_y, max_x]

    cc_nopeaks = cc.copy()
    cc_nopeaks[max_y - neigh_radius:max_y + neigh_radius, max_x - neigh_radius:max_x + neigh_radius] = 0

    pce_energy = np.mean(cc_nopeaks.flatten() ** 2)

    out['peak'] = (max_y, max_x)
    out['pce'] = (peak_height ** 2) / pce_energy * np.sign(peak_height)
    out['cc'] = peak_height

    return out


"""
Statistical functions
"""


def stats(cc: np.ndarray, gt: np.ndarray, ) -> dict:
    """
    Compute statistics
    :param cc: cross-correlation or normalized cross-correlation matrix
    :param gt: boolean multidimensional array representing groundtruth
    :return: statistics dictionary
    """
    assert (cc.shape == gt.shape)
    assert (gt.dtype == np.bool)

    assert (cc.shape == gt.shape)
    assert (gt.dtype == np.bool)

    fpr, tpr, th = roc_curve(gt.flatten(), cc.flatten())
    auc_score = auc(fpr, tpr)

    # EER
    eer_idx = np.argmin((fpr - (1 - tpr)) ** 2, axis=0)
    eer = float(fpr[eer_idx])

    outdict = {
        'tpr': tpr,
        'fpr': fpr,
        'th': th,
        'auc': auc_score,
        'eer': eer,
    }

    return outdict


def gt(l1: list or np.ndarray, l2: list or np.ndarray) -> np.ndarray:
    """
    Determine the Ground Truth matrix given the labels
    :param l1: fingerprints labels
    :param l2: residuals labels
    :return: groundtruth matrix
    """
    l1 = np.array(l1)
    l2 = np.array(l2)

    assert (l1.ndim == 1)
    assert (l2.ndim == 1)

    gt_arr = np.zeros((len(l1), len(l2)), np.bool)

    for l1idx, l1sample in enumerate(l1):
        gt_arr[l1idx, l2 == l1sample] = True

    return gt_arr


# In[10]:


def get_camera_prnu(camera_nr: int) -> np.ndarray:
    path = 'C:\\Users\\rotar\\OneDrive\\Desktop\\Master\\UNITN\\Semester 1\\Multimedia Data Security\\2nd Competition\\dev-dataset\\'
    index = '000'
    noises = []
    
    for i in range(100):
        index ="%03d" % (i+1)
        im = Image.open(path + 'flat-camera-'+ str(camera_nr) + '\\flat_c'+ str(camera_nr) + '_' + index + '.tif')
        imarray = np.array(im)
        noise = noise_extract(imarray)
        noises.append(noise)
      
    final = np.mean(noises,0)
    return final
    


# In[11]:


prnu1 = get_camera_prnu(1)
prnu2 = get_camera_prnu(2)
prnu3 = get_camera_prnu(3)
prnu4 = get_camera_prnu(4)
#prnu1_img = Image.fromarray(prnu1, 'RGB')
#prnu1_img.show()


# In[12]:


import pickle
f = open('store.pckl', 'wb')
prnus = [prnu1, prnu2, prnu3, prnu4]
pickle.dump(prnus, f)
f.close()


# In[7]:


import pickle
file = open('store.pckl', 'rb')
prev_prnus = pickle.load(file)
prnu1 = prev_prnus[0]
prnu2 = prev_prnus[1]
prnu3 = prev_prnus[2]
prnu4 = prev_prnus[3]


# In[8]:


import math
def mean2(x):
    y = np.sum(x) / np.size(x);
    return y

def corr2(a,b):
    a = a - mean2(a)
    b = b - mean2(b)

    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum());
    return r


# In[9]:


def get_map(prnu : np.ndarray or list, noise: np.ndarray or list) -> np.ndarray:
    
    diff_map = numpy.zeros_like(prnu)
    for i in range(len(prnu)):
        for j in range(len(prnu[0])):
            diff_map[i][j] = round(abs(prnu[i][j] - noise[i][j]))
            
    return diff_map


# In[76]:



#im = Image.open('C:\\Users\\rotar\\OneDrive\\Desktop\\Master\\UNITN\\Semester 1\\Multimedia Data Security\\2nd Competition\\dev-dataset\\dev-dataset-forged\\dev_0004.tif')

im = Image.open('C:\\Users\\rotar\\OneDrive\\Desktop\\Master\\UNITN\\Semester 1\\Multimedia Data Security\\2nd Competition\\dev-dataset\\dev-dataset-forged\\dev_0029.tif')


imarray = np.array(im)

noise = noise_extract(imarray)


# In[77]:


print(corr2(noise, prnu1))
print(corr2(noise, prnu2))
print(corr2(noise, prnu3))
print(corr2(noise, prnu4))


# In[267]:


#diffs1 = get_map(noise[:,:,1], prnu1[:,:,1])
#diffs2 = get_map(noise[:,:,1], prnu2[:,:,1])
#diffs3 = get_map(noise[:,:,1], prnu3[:,:,1])
diffs41 = get_map(noise[:,:,0], prnu4[:,:,0])
diffs42 = get_map(noise[:,:,1], prnu4[:,:,1])
diffs43 = get_map(noise[:,:,2], prnu4[:,:,2])


# In[116]:


np.amax(diffs)


# In[227]:


diffs4 = numpy.zeros_like(diffs41)
for i in range(len(diffs41)):
        for j in range(len(diffs41[0])):
            if (diffs41[i][j] + diffs42[i][j] + diffs43[i][j]) >7:
                diffs4[i][j] = 255
            else:
                diffs4[i][j] = 0


# In[48]:


def is_inside(i: int, j: int, rows: int, cols: int):
    if (i>=0 and j >=0 and i<rows and j<cols): return True
    return False


# In[46]:


def get_map_segment(prnu : np.ndarray or list, noise: np.ndarray or list, rows: int, cols: int, thr: float, clear: bool) -> np.ndarray:
    row_inc = prnu.shape[0]/rows
    col_inc = prnu.shape[1]/cols
    
    avg = numpy.average(numpy.absolute(prnu-noise))
    
    diff_map = diff_map = numpy.zeros_like(prnu, np.uint8)
    for i in range(rows):
        for j in range(cols):
            i1 = int(i*row_inc)
            i2 = int((i+1)*row_inc)
            j1 = int(j*col_inc)
            j2 = int((j+1)*col_inc)
            
            m = numpy.mean(numpy.absolute(prnu[i1:i2, j1:j2] - noise[i1:i2, j1:j2]))
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


# In[78]:


import cv2
diff = get_map_segment(noise[:,:,1], prnu4[:,:,1], 200, 300, 1, True )
#diff = cv2.medianBlur(diff, 29)
img = Image.fromarray(diff)
img.show()


# In[80]:


kernel = np.ones((29,29),np.uint8)
#erosion = cv2.erode(diff,kernel,iterations = 2)
erosion = cv2.dilate(diff,kernel,iterations = 3)
erosion = cv2.erode(erosion,kernel,iterations = 3)
img = Image.fromarray(erosion)
img.show()
#cv2.imshow("res", diff)


# In[16]:


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


# In[72]:


map_gt = imageio.imread('C:\\Users\\rotar\\OneDrive\\Desktop\\Master\\UNITN\\Semester 1\\Multimedia Data Security\\2nd Competition\\dev-dataset\\dev-dataset-maps\\dev_0027.bmp')
map_gt = np.array(map_gt/255,dtype=np.uint8)

map_est = np.array(erosion/255,dtype=np.uint8)

F,_,_,_ = f_measure(map_gt,map_est)
print(F)


# In[ ]:




