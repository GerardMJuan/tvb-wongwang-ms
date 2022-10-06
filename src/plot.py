"""
Plot functions.

Functions to plot:
-matrices
-graphs
-etc

and everything that is needed to visualize results
"""

import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import glob
from src.eval import fmri_corr, fmri_uncentered_corr, manual_bandpass

def plot_full_FC(data_path, exp_dir, opt_path, fMRI, hagmann=False):
    """
    Function that plots, for a specific run of a specific subject, as many figures
    as FC have been computed- Each figure has the FC, SymFC, SC, histograms, and correlations between
    them.
    """
    # plot the histogram
    i = 0

    if not os.path.exists(opt_path):
        os.mkdir(opt_path)
    fmri_list = []
    corr_list = []
    subj = os.path.basename(data_path.rstrip('/'))

    SC = np.loadtxt(f"{data_path}/results/{os.path.basename(subj)}_SC_weights.txt", delimiter=' ')
    
    # hagmann
    if hagmann:
        SC = SC[:-2, :-2]

    """
    if not hagmann:
        z_fmri = np.arctanh(fMRI)
        infs = np.isinf(z_fmri).nonzero()
        #replace the infs with 0            
        for idx in range(len(infs[0])):
            z_fmri[infs[0][idx]][infs[1][idx]] = 0
        np.fill_diagonal(z_fmri, 0)
    else:
    """
    z_fmri = fMRI
    corr_sc = fmri_corr(z_fmri, SC)

    for fmri_dir in sorted(glob.glob(f'{exp_dir}/FCsyn_*.txt')):
        plt.figure(figsize=(20, 15))
        fMRI_syn = np.genfromtxt(f"{fmri_dir}", delimiter=' ')
        print(fmri_dir)
        #compute the zfisher correlation
        z_fmri_syn = np.arctanh(fMRI_syn)
        infs = np.isinf(z_fmri_syn).nonzero()
        #replace the infs with 0            
        for idx in range(len(infs[0])):
            z_fmri_syn[infs[0][idx]][infs[1][idx]] = 0
        np.fill_diagonal(z_fmri_syn, 0)

        # compute correlation of both matrices with SC
        corr_fc = fmri_uncentered_corr(z_fmri, z_fmri_syn)
        fmri_list.append(z_fmri_syn)
        print(corr_fc)
        corr_list.append(corr_fc)
        fig_dir2 = f'{opt_path}/{os.path.basename(subj)}_{i}.png'
        plt.subplot(231), plt.imshow(z_fmri_syn, interpolation='none', cmap="jet"), plt.colorbar(), plt.title(f"Simulated FC\n {corr_fc = :.3f}")
        plt.subplot(232), plt.imshow(z_fmri, interpolation='none', cmap="jet"), plt.colorbar(), plt.title("Empirical FC")
        plt.subplot(233), plt.imshow(SC, interpolation='none', cmap="jet"), plt.colorbar(), plt.title(f"Empirical SC\n {corr_sc = :.3f}")
        plt.subplot(234), plt.hist(z_fmri_syn), plt.colorbar(), plt.title(f"Histogram sim. FC")
        plt.subplot(235), plt.hist(z_fmri), plt.colorbar(), plt.title(f"Histogram emp. FC")
        plt.subplot(236), plt.hist(SC), plt.colorbar(), plt.title(f"Histogram emp. SC")
        plt.savefig(fig_dir2)
        plt.close()
        i+=1

    # if more than one, compute avg
    if len(fmri_list)>1:
        fmri_avg = np.mean(fmri_list, axis=0)
        corr_fc = np.mean(corr_list, axis=0)

        plt.figure(figsize=(20, 15))
        fig_dir2 = f'{opt_path}/{os.path.basename(subj)}_avg.png'
        plt.subplot(231), plt.imshow(fmri_avg, interpolation='none', cmap="jet"), plt.colorbar(), plt.title(f"Simulated FC\n {corr_fc = :.3f}")
        plt.subplot(232), plt.imshow(z_fmri, interpolation='none', cmap="jet"), plt.colorbar(), plt.title("Empirical FC")
        plt.subplot(233), plt.imshow(SC, interpolation='none', cmap="jet"), plt.colorbar(), plt.title(f"Empirical SC\n {corr_sc = :.3f}")
        plt.subplot(234), plt.hist(fmri_avg), plt.colorbar(), plt.title(f"Histogram sim. FC")
        plt.subplot(235), plt.hist(z_fmri), plt.colorbar(), plt.title(f"Histogram emp. FC")
        plt.subplot(236), plt.hist(SC), plt.colorbar(), plt.title(f"Histogram emp. SC")
        plt.savefig(fig_dir2)
        plt.close()


def plot_connectivity():
    """
    Plot a connectivity matrix
    """
    print("nyi")

def plot_count_weights(SC):
    """
    Count plot, in bar form,
    of the values in the SC.
    """
    sns.histplot(SC.ravel())
    plt.show()


def plot_isocline_pse(exp_id, params, value="GlobalVariance"):
    """
    Function that plots a 2D map of the global variance of an optimization
    over two different parameters in the TVB, similarly at how is it done in the 
    TVB webapp.
    exp_id is the experiment from MLflow
    params is a tuple with (param1, param2), which are the naem of the params searched
    value can be either "GlobalVariance" or "Corr"
    """

    # Get data from the experiment

    # Get those two parameters
    # assume they are the only ones, but eventually we have
    # to manually fix the others?
    # or maybe plot several maps, for all the other combinations of params?
    # this could be a param of the function, too

    # plot the map
    print('nyi')