"""
Eavluation functions.

This file contains evaluation functions needed to 
assess results from the models:
- Correlation of matrices
- Correlation of values
- Linear models
- ANOVA and t-tests

Probably the tests should be done on a separate, interactive script, to
be able to easily run them, but we can put repeatable functions here later
"""

from scipy.stats import pearsonr
import numpy as np
from scipy.signal import hilbert
import cmath
from scipy.signal import butter, sosfilt, sosfreqz, detrend
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift
from scipy.signal import lfilter

def manual_bandpass(data, tr_ms, freq_cutoff=0.07, freq_low=0.001):
    """
    Do the bandpass using fft and building the filter manually.

    Adapted from the fmri pipeline.

    So that the BOLd data is applied the exact same filtering as the 
    normal data
    """
    tdim = data.shape[1]

    #build filter
    time_all = np.arange(0,(tdim*(tr_ms/1000))-freq_low,freq_low)
    time_subTR = time_all[0:-1:int(tr_ms)]
    length = len(time_subTR)
    ccc = 1.0/(tr_ms/1000)/length
    cccc = freq_cutoff/ccc
    len1 = round(length/2.0-(cccc-2))
    len2 = round(length/2.0+(cccc+1))

    tmp = np.zeros([tdim,1])
    tmp[int(len1):int(len2)]=1
    tmpMA = len1-4
    tmpMA2 = round(tmpMA/2)
    tmpAB = np.divide(np.add(1,np.cos(np.arange(np.pi, 2*np.pi+((np.pi/tmpMA)/2), np.pi/tmpMA))),2)
    tmpAB = tmpAB.reshape(tmpAB.shape[0],1)
    tmpBA = np.divide(np.add(1,np.cos(np.arange(2*np.pi,np.pi-((np.pi/tmpMA)/2), -np.pi/tmpMA))),2)
    tmpBA = tmpBA.reshape(tmpBA.shape[0],1)

    tmp[int(len1-tmpMA+tmpMA2-1):int(len1+tmpMA2)]=tmpAB
    tmp[int(len2-tmpMA2-1):int(len2+tmpMA-tmpMA2)]=tmpBA

    tmp_mean = np.mean(data, axis=1) # fa el mean across time

    arr_f = detrend(data)
    arr_f = fftshift(arr_f, axes=1)
    arr_f = fft(arr_f, axis=1, workers=1)
    arr_f = fftshift(arr_f, axes=1)
    # arr_f = fftshift(fft(fftshift(detrend(data), axes=1), axis=1, workers=1), axes=1)
    arr_fc = np.multiply(arr_f, tmp.T)
    data_lowpass = np.real(fftshift(ifft(fftshift(arr_fc, axes=1), axis=1, workers=1), axes=1))

    data_lowpass += tmp_mean.reshape(tmp_mean.shape + (1,))

    # aixo ultim que onda
    data_lowpass -=  np.min(data_lowpass)
    data_lowpass *= 30000.0 / np.max(data_lowpass)

    return data_lowpass

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    lowcut = lowcut / nyq
    highcut = highcut / nyq
    sos = butter(order, [lowcut, highcut], analog=False, btype='band', output='sos')
    return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y

def butter_bandpass_i(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    lowcut = lowcut / nyq
    highcut = highcut / nyq
    b, a  = butter(order, [lowcut, highcut], analog=False, btype='band')
    return b, a

def butter_bandpass_filter_i(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass_i(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def fmri_uncentered_corr(FC_subj, FC_pred):
    "data1 & data2 should be numpy arrays."
    #mean1 = FC_subj.mean() 
    #mean2 = FC_pred.mean()

    triu_subj = FC_subj[np.triu_indices(FC_subj.shape[0], k=1)]
    triu_pred = FC_pred[np.triu_indices(FC_pred.shape[0], k=1)]

    denom = np.sqrt(np.sum(triu_pred**2)) * np.sqrt(np.sum(triu_subj**2))

    #corr = ((data1-mean1)*(data2-mean2)).mean()/(std1*std2)
    corr = np.dot(triu_subj,triu_pred)/denom
    return corr

def fmri_corr(FC_subj, FC_pred, type='corr'):
    """
    Function that computes the pearson correlation between the real fmri connectivity
    matrix and the simulated fmri matrix. Different modes:
    FC_subj: original functional connectivity matrix. 
    FC_pred: predicted/generated functional connectivity matrix.

    If needed, can extend to different models of correlation
    """

    # GET LOWER TRIANGLES
    # the 1 is to ignore the diagonal
    # triu_subj = np.triu(FC_subj, 1)
    # triu_pred = np.triu(FC_pred, 1)

    triu_subj = FC_subj[np.triu_indices(FC_subj.shape[0], k=1)]
    triu_pred = FC_pred[np.triu_indices(FC_pred.shape[0], k=1)]

    # get only the values of the upper triangle, and ravel to compute the correlation
    # triu_subj = triu_subj[np.nonzero(triu_subj)].ravel()
    # triu_pred = triu_pred[np.nonzero(triu_pred)].ravel()
    
    # COMPUTE PEARSON CORRELATION
    if type=='corr':
        corr, pval = pearsonr(triu_subj, triu_pred)
    elif type=='sq':
        corr = np.square(np.subtract(triu_subj, triu_pred)).mean()
    else:
        corr = 0
        print('nyi')
    # RETURN
    return corr

def remove_mean(x, axis):
    """
    Remove mean from numpy array along axis
    """
    # Example for demean(x, 2) with x.shape == 2,3,4,5
    # m = x.mean(axis=2) collapses the 2'nd dimension making m and x incompatible
    # so we add it back m[:,:, np.newaxis, :]
    # Since the shape and axis are known only at runtime
    # Calculate the slicing dynamically
    idx = [slice(None)] * x.ndim
    idx[axis] = np.newaxis

    return ( x - x.mean(axis=axis)[tuple(idx)] ) # / x.std(axis=axis)[idx]
    # return ( x - x.mean(axis=axis)[idx] ) # / x.std(axis=axis)[idx]


def kuromoto_metastability(time_series):
    """
    Compute metastability of the kuramoto index.
    Time series is a BOLD signal of shape nregions x timepoints.

    Papers of interest:
    Hellyer PJ, Shanahan M, Scott G, Wise RJ, Sharp DJ, Leech R. 
    The control of global brain dynamics: opposing actions of frontoparietal control and default mode networks on attention. 
    J Neurosci. 2014;34(2):451-461. doi:10.1523/JNEUROSCI.1853-13.2014

    Deco, G., Kringelbach, M.L., Jirsa, V.K. et al. The dynamics of resting fluctuations in the brain: 
    metastability and its dynamical cortical core. Sci Rep 7, 3095 (2017). https://doi.org/10.1038/s41598-017-03073-5
    https://www.nature.com/articles/s41598-017-03073-5#Sec6
    """


    # input: demeaned BOLD signal of shape Nregions x timepoints
    hb = hilbert(time_series, axis=1)

    # polar torna una tuple que es ( modulus (abs), phase )
    # theta_sum = np.sum(np.exp(1j * (np.vectorize(cmath.polar)(hb)[1])), axis=0)
    theta = np.exp(1j * np.unwrap(np.angle(hb)))
    theta_sum = np.sum(theta, axis=0)
    
    # kuramoto = np.vectorize(cmath.polar)(theta_sum / time_series.shape[0])[0]
    kuramoto = np.abs(theta_sum) / time_series.shape[0]

    # ara que tinc el kuramoto, calcular metastability across time
    metastability = kuramoto.std()
    """
    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(15, 5),
                            subplot_kw={
                                "ylim": (-1.1, 1.1),
                                "xlim": (-1.1, 1.1),
                                "xlabel": r'$\cos(\theta)$',
                                "ylabel": r'$\sin(\theta)$',
                            })

    times = [0, 100, 194]
    for ax, time in zip(axes, times):
        ax.plot(np.cos(np.angle(hb)[:, time]),
                np.sin(np.angle(hb)[:, time]),
                'o',
                markersize=10)
        ax.set_title(f'Time = {time}')

    plt.savefig('test_kuramoto.png')
    print(metastability)
    """
    return metastability