"""
Script that, given a complete run of the simulation, creates
a new optimization with two different objectives, those being
correlation and maximum meta, and generates new best FC and 
parameter maps.
"""

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import statsmodels.api as sm
import statsmodels.formula.api as smf
import joblib
import scipy.stats as stat
from scipy.stats import pearsonr
import optuna
from eval import fmri_corr, fmri_uncentered_corr, manual_bandpass, kuromoto_metastability, remove_mean, butter_bandpass_filter, butter_bandpass_filter_i
import math
# parameter 1: input directory
exp_path = sys.argv[1]
data_path = sys.argv[2]

#hardcoded parameters
g_low = 0.05
g_high = 3.5
cs_low = 0.1
cs_high = 50.0

#output will be saved in two new folders: Corr and Meta, in the root
# of the input directory
opt_names = ["Corr"]#, "Meta"]
for opt in opt_names:        
    #for each subject done, load the study     
    for subj in glob.glob(f'{exp_path}/*'):
        print(subj)
        opt_path = os.path.join(subj, opt)
        if not os.path.exists(opt_path):
            os.makedirs(opt_path)

        if not os.path.isfile(f"{subj}/study_{opt}.pkl"): continue
        study = joblib.load(f"{subj}/study_{opt}.pkl")

        df_study = study.trials_dataframe()
        print(len(df_study))

        ts_path = f"{data_path}/{os.path.basename(subj)}/results/corrlabel_ts.txt"
        corr_ts = np.loadtxt(ts_path)
        corr_ts = corr_ts.T

        # DECOMMENT IF NEEDED
        # corr_ts_man = manual_bandpass(corr_ts, 2500, freq_cutoff=0.08)
        N_filt = 1
        print(N_filt)
        corr_ts_but = butter_bandpass_filter_i(remove_mean(corr_ts, axis=1), 0.04, 0.07, 1.0/(2500/1000), N_filt)
        
        metast_emp = kuromoto_metastability(remove_mean(corr_ts, axis=1))
        print("metastability: base")
        print(metast_emp)

        metast_emp = kuromoto_metastability(remove_mean(corr_ts_but, axis=1))
        print("metastability: butter")
        print(metast_emp)

        #Create fMRI
        fMRI = np.corrcoef(corr_ts_but)
        fMRI = np.nan_to_num(fMRI)

        fMRI_r = np.loadtxt(f"{data_path}/{os.path.basename(subj)}/results/r_matrix.csv", delimiter=',')
        SC = np.loadtxt(f"{data_path}/{os.path.basename(subj)}/results/{os.path.basename(subj)}_SC_weights.txt", delimiter=' ')
        
        z_fmri = np.arctanh(fMRI)
        infs = np.isinf(z_fmri).nonzero()
        #replace the infs with 0            
        for idx in range(len(infs[0])):
                    z_fmri[infs[0][idx]][infs[1][idx]] = 0
        np.fill_diagonal(z_fmri, 0)
        # z_fmri = fMRI_zr
        print(z_fmri.shape)
        print(SC.shape)
        corr_sc = fmri_corr(z_fmri, SC)

        #now, we would need to recreate the study, right?
        study_new = optuna.create_study(direction="maximize")
        for i, dict_res in enumerate(df_study.itertuples(index=False)):
            
            value_opt = dict_res[df_study.columns.get_loc(f"user_attrs_{opt}")]
            if opt == "Corr":
                # load the fmri for that iteration
                G = dict_res.params_g
                cs = dict_res.params_cs

                corr_array = []

                for fmri_dir in sorted(glob.glob(f'{subj}/runs/{cs}*_{G}*/BOLDsyn_*.txt')):
                    corr_ts = np.genfromtxt(f"{fmri_dir}", delimiter=' ')
                    # corr_ts = butter_bandpass_filter_i(remove_mean(corr_ts,axis=1), 0.04, 0.07, 1.0/2.5, 1)
                    # corr_ts = manual_bandpass(corr_ts, 2500, freq_cutoff=0.08)

                    fMRI_syn = np.corrcoef(corr_ts)
                    fMRI_syn = np.nan_to_num(fMRI_syn)

                    #compute the zfisher correlation
                    z_fmri_syn = np.arctanh(fMRI_syn)
                    infs = np.isinf(z_fmri_syn).nonzero()
                    #replace the infs with 0            
                    for idx in range(len(infs[0])):
                        z_fmri_syn[infs[0][idx]][infs[1][idx]] = 0
                    np.fill_diagonal(z_fmri_syn, 0)
                    # compute correlation of both matrices with SC
                    corr_fc1 = fmri_uncentered_corr(z_fmri, z_fmri_syn)
                    corr_array.append(corr_fc1)

                fmri_avg = np.mean(corr_array)

                value_opt = fmri_avg

            trial = optuna.trial.create_trial(
                        params={
                                "g": dict_res.params_g,
                                "cs": dict_res.params_cs
                                },
                        distributions={
                                    "g": optuna.distributions.UniformDistribution(g_low, g_high),
                                    "cs": optuna.distributions.UniformDistribution(cs_low, cs_high)
                                    },
                        value= value_opt, #change to whatever we need
                    
                        user_attrs={
                            "Meta": dict_res.user_attrs_Meta,
                            "Meta_std": dict_res.user_attrs_Meta_std,
                            "Metaemp": dict_res.user_attrs_Metaemp,
                            "DeltaMeta": dict_res.user_attrs_DeltaMeta,
                            "DeltaMeta_std": dict_res.user_attrs_DeltaMeta_std,
                            "Corr": dict_res.user_attrs_Corr,
                            "Corr_std": dict_res.user_attrs_Corr_std,
                            "FCsyn": dict_res.user_attrs_FCsyn,
                        }
                    )
            study_new.add_trial(trial)
        
        # get best results
        best_trial = study_new.best_trial
        print(best_trial)

        # plot the map
        fig_meta = f'{opt_path}/{os.path.basename(subj)}_figmeta_{opt}.png'
        optuna.visualization.matplotlib.plot_contour(study_new, params=["g", "cs"])
        plt.savefig(fig_meta)
        plt.close()

        # plot the histogram
        i = 0

        fmri_list = []

        ts_path = f"{data_path}/{os.path.basename(subj)}/results/corrlabel_ts.txt"
        corr_ts = np.loadtxt(ts_path)
        corr_ts = corr_ts.T

        # DECOMMENT IF NEEDED
        # corr_ts_man = manual_bandpass(corr_ts, 2500, freq_cutoff=0.08)
        N_filt = 1
        print(N_filt)
        corr_ts_but = butter_bandpass_filter_i(remove_mean(corr_ts, axis=1), 0.04, 0.07, 1.0/(2500/1000), N_filt)
        
        metast_emp = kuromoto_metastability(remove_mean(corr_ts, axis=1))
        print("metastability: base")
        print(metast_emp)

        metast_emp = kuromoto_metastability(remove_mean(corr_ts_man, axis=1))
        print("metastability: manual")
        print(metast_emp)

        metast_emp = kuromoto_metastability(remove_mean(corr_ts_but, axis=1))
        print("metastability: butter")
        print(metast_emp)

        #Create fMRI
        fMRI = np.corrcoef(corr_ts_but)
        fMRI = np.nan_to_num(fMRI)

        fMRI_r = np.loadtxt(f"{data_path}/{os.path.basename(subj)}/results/r_matrix.csv", delimiter=',')

        for fmri_dir in sorted(glob.glob(f'{subj}/runs/{best_trial.params["cs"]}*_{best_trial.params["g"]}*/BOLDsyn_*.txt')):
            plt.figure(figsize=(20, 15))

            corr_ts = np.genfromtxt(f"{fmri_dir}", delimiter=' ')
            # corr_ts = manual_bandpass(corr_ts, 2500, freq_cutoff=0.08)

            fMRI_syn = np.corrcoef(corr_ts)
            fMRI_syn = np.nan_to_num(fMRI_syn)

            print(fmri_dir)
            #compute the zfisher correlation
            z_fmri_syn = np.arctanh(fMRI_syn)
            infs = np.isinf(z_fmri_syn).nonzero()
            #replace the infs with 0            
            for idx in range(len(infs[0])):
                z_fmri_syn[infs[0][idx]][infs[1][idx]] = 0
            np.fill_diagonal(z_fmri_syn, 0)
            print("The correlations")
            # compute correlation of both matrices with SC
            corr = best_trial.value
            print(corr)
            corr_fc1 = fmri_uncentered_corr(z_fmri, z_fmri_syn)
            print(corr_fc1)

            corr_fc = fmri_uncentered_corr(fMRI, fMRI_syn)
            print(corr_fc)

            corr_fc = fmri_corr(z_fmri, z_fmri_syn)
            print(corr_fc)

            corr_fc = fmri_corr(fMRI, fMRI_syn)
            print(corr_fc)

            fmri_list.append(fMRI_syn)

            fig_dir2 = f'{opt_path}/{os.path.basename(subj)}_{i}.png'
            """
            plt.subplot(241), plt.imshow(fMRI_syn, interpolation='none'), plt.colorbar(), plt.title(f"FC Syn\n {corr_fc1 = :.3f}")
            plt.subplot(242), plt.imshow(fMRI, interpolation='none'), plt.colorbar(), plt.title("FC from corrlabel")
            plt.subplot(243), plt.imshow(z_fmri, interpolation='none'), plt.colorbar(), plt.title(f"r matrix\n {corr_sc = :.3f}")
            plt.subplot(244), plt.imshow(z_fmri_syn, interpolation='none'), plt.colorbar(), plt.title(f"zr matrix\n {corr_sc = :.3f}")
            plt.subplot(245), plt.hist(fMRI_syn), plt.colorbar(), plt.title(f"Histogram sim. FC")
            plt.subplot(246), plt.hist(fMRI), plt.colorbar(), plt.title(f"Histogram emp. FC")
            plt.subplot(247), plt.hist(z_fmri), plt.colorbar(), plt.title(f"Histogram emp. SC")
            plt.subplot(248), plt.hist(z_fmri_syn), plt.colorbar(), plt.title(f"Histogram emp. SC")
            """
            plt.subplot(231), plt.imshow(fMRI_syn, interpolation='none', cmap="jet"), plt.colorbar(), plt.title(f"Simulated FC\n {corr_fc1 = :.3f}")
            plt.subplot(232), plt.imshow(fMRI, interpolation='none', cmap="jet"), plt.colorbar(), plt.title("Empirical FC")
            plt.subplot(233), plt.imshow(SC, interpolation='none', cmap="jet"), plt.colorbar(), plt.title(f"Empirical SC\n {corr_sc = :.3f}")
            plt.subplot(234), plt.hist(fMRI_syn), plt.colorbar(), plt.title(f"Histogram sim. FC")
            plt.subplot(235), plt.hist(fMRI), plt.colorbar(), plt.title(f"Histogram emp. FC")
            plt.subplot(236), plt.hist(SC), plt.colorbar(), plt.title(f"Histogram emp. SC")
            plt.savefig(fig_dir2)
            plt.close()
            
            i+=1
        """

        # if more than one, compute avg
        if len(fmri_list)>1:
            fmri_avg = np.mean(fmri_list, axis=0)
            z_fmri_avg = np.arctanh(fmri_avg)
            infs = (z_fmri_avg == np.inf).nonzero()
            #replace the infs with 0            
            for idx in range(len(infs[0])):
                z_fmri_avg[infs[0][idx]][infs[1][idx]] = 0
            np.fill_diagonal(z_fmri_avg, 0)

            corr_fc = fmri_uncentered_corr(z_fmri, z_fmri_avg)
            
            plt.figure(figsize=(20, 15))
            fig_dir2 = f'{opt_path}/{os.path.basename(subj)}_avg.png'
            plt.subplot(231), plt.imshow(fmri_avg, interpolation='none', cmap="jet"), plt.colorbar(), plt.title(f"Simulated FC\n {corr_fc = :.3f}")
            plt.subplot(232), plt.imshow(fMRI_r, interpolation='none', cmap="jet"), plt.colorbar(), plt.title("Empirical FC")
            plt.subplot(233), plt.imshow(SC, interpolation='none', cmap="jet"), plt.colorbar(), plt.title(f"Empirical SC\n {corr_sc = :.3f}")
            plt.subplot(234), plt.hist(fmri_avg), plt.colorbar(), plt.title(f"Histogram sim. FC")
            plt.subplot(235), plt.hist(fMRI_r), plt.colorbar(), plt.title(f"Histogram emp. FC")
            plt.subplot(236), plt.hist(SC), plt.colorbar(), plt.title(f"Histogram emp. SC")
            plt.savefig(fig_dir2)
            plt.close()
            """
            
