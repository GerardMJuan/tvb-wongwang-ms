"""
Small script that creates N FC from the whole simulation
of a single subject to compare between them
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import statsmodels.api as sm
import statsmodels.formula.api as smf
import joblib
import sys
sys.path.insert(0, "..")
from src.eval import fmri_corr, kuromoto_metastability, remove_mean, manual_bandpass

subj = ""

out_dir = f"{subj}"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

exp_path = f"/{subj}"
# information about the subject

bold_tr = 2500
simlen = 517500
t_to_discard = int(np.ceil(30000. / bold_tr))

# do N iterations
N = 10
simlen = (simlen - 30000) * N + 30000

# load the study and get best results
study = joblib.load(f"{exp_path}/study.pkl")
# get best results
best_trial = study.best_trial

g = best_trial.params["g"]
cs = best_trial.params["cs"]

print(f'{exp_path}/runs/{cs}_{g}_*/*_fMRI.txt')
# Print the FC matrix, together with the actual one
txt_path = glob.glob(f'{exp_path}/runs/{cs}_{g}_*/*_fMRI.txt')[0]

# load the result
BOLD_syn = np.genfromtxt(txt_path, delimiter=' ')

# discard first 30 sec
BOLD_syn = BOLD_syn[:,t_to_discard:]
simlen_act = BOLD_syn.shape[1] / N

for i in range(N):

    # select the part of the bold signal
    BOLD_syn_i = BOLD_syn[:,int(i*simlen_act):int(i*simlen_act+simlen_act)]
    BOLD_syn_i_lowpass = manual_bandpass(BOLD_syn_i, bold_tr, freq_cutoff=0.08)
    metast = kuromoto_metastability(remove_mean(BOLD_syn_i_lowpass, axis=1))

    fMRI_syn = np.corrcoef(BOLD_syn_i_lowpass)
    fMRI_syn = np.nan_to_num(fMRI_syn)
    fMRI_syn = (fMRI_syn + fMRI_syn.T) / 2.0
    plt.figure(figsize=(7, 7))
    plt.imshow(fMRI_syn, interpolation='none'), plt.colorbar()
    plt.savefig(f"{out_dir}/{subj}_FC_syn_iter_{i}.png")
    plt.close()
