"""
Run over a existing, run subject,
read the output experiments, and save the results in a new directory.

It needs to have the same structure as hpc_run_subj.py, as it will be run
parallel in hpc from the parent script hpc_run_all.
"""
import numpy as np
import time
from multiprocessing import Pool
import os
import psutil
import itertools
from subprocess import Popen
from src.eval import fmri_corr, kuromoto_metastability, remove_mean, manual_bandpass, fmri_uncentered_corr, butter_bandpass_filter
import glob
import optuna
import matplotlib.pyplot as plt
import joblib
import click
from src.plot import plot_full_FC


def run_iterations_separate(params, out_dir, subj_path, metast_emp, fMRI, short_run=False):
    """
    parallel function

    calls to the C program N times, instead of doing a very long simulation
    """
    bold_tr = params["bold_tr"]

    # g and cs are d from out_dir
    print(out_dir)
    g = os.path.basename(out_dir.rstrip('/')).split('_')[1]
    cs = os.path.basename(out_dir.rstrip('/')).split('_')[0]

    # compute scantime
    # it is TotalScanTime - TR*5 (5 scans discarded in preprocessing) + 80000 (80 seg discarded at first in train_fast.py)
    subjID = os.path.basename(subj_path.rstrip('/'))

    t_to_discard = int(np.ceil(30000. / bold_tr))

    N = len(glob.glob(f'{out_dir}/{subjID}_param_set_{cs}_{g}_*.txt_fMRI.txt'))
    metast_arr = []
    deltameta_arr = []
    corr_arr = []
    
    z_fmri = fMRI
    for i in range(N):

        # Read and load the result
        #txt_path = f'{out_exp}/{subjID}_param_set_{cs}_{g}_{seed}.txt_fMRI.txt'
        txt_path = glob.glob(f'{out_dir}/{subjID}_param_set_{cs}_{g}_*.txt_fMRI.txt')[i]

        # load the result
        BOLD_syn = np.genfromtxt(txt_path, delimiter=' ')
        # discard first 80 sec
        BOLD_syn_i = BOLD_syn[:,t_to_discard:]

        # select the part of the bold signal
        # BOLD_syn_i = manual_bandpass(BOLD_syn_i, bold_tr, freq_cutoff=0.08)
        BOLD_syn_i_butt = butter_bandpass_filter(remove_mean(BOLD_syn_i, axis=1), 0.04, 0.07, 1/(bold_tr/1000), order=1)

        # BOLD_syn_i_lowpass = BOLD_syn_i
        metast = kuromoto_metastability(remove_mean(BOLD_syn_i_butt, axis=1))

        diff_meta = np.abs(metast_emp - metast)
        fMRI_syn = np.corrcoef(BOLD_syn_i)
        fMRI_syn = np.nan_to_num(fMRI_syn)

        #compute the zfisher correlation
        # need to do the things!

        z_fmri_syn = np.arctanh(fMRI_syn)
        infs = np.isinf(z_fmri_syn).nonzero()
        #replace the infs with 0            
        for idx in range(len(infs[0])):
            z_fmri_syn[infs[0][idx]][infs[1][idx]] = 0
        np.fill_diagonal(z_fmri_syn, 0)
    
        corr = fmri_uncentered_corr(z_fmri, z_fmri_syn)

        metast_arr.append(metast)
        deltameta_arr.append(diff_meta)
        corr_arr.append(corr)


    # prepare return
    dict_return = {
        "run_dir": out_dir,
        "g" : float(g),
        "cs" : float(cs),
        "Meta": np.mean(metast_arr),
        "Meta_std": np.std(metast_arr),
        "Metaemp": metast_emp,
        "DeltaMeta": np.mean(deltameta_arr),
        "DeltaMeta_std": np.std(deltameta_arr),
        "Corr": np.mean(corr_arr),
        "Corr_std": np.std(corr_arr),
        "FCsyn": f'{out_dir}/FCsyn_0.txt'
    }

    return dict_return


#@click.command(help="Perform hyperparameter search with a single subject using the fast-tvb model.")
#@click.option('--short_run', '-s', is_flag=True, help="Short run.")
#@click.argument("path", type=click.STRING)
#@click.argument("out_path", type=click.STRING)
#@click.argument("tr", type=click.STRING)
#@click.argument('simlen', type=click.STRING)
#@click.argument('model', type=click.STRING)
def run_subj(path, out_path, tr, simlen, model, short_run):
    """
    Run a single subject in parallel, doing the grid search

    path: path where the subject info is
    out_path: path where we will save the results of the experiment
    tr: TR of the BOLD
    simlen: simulation length
    model: type of model to use: can be either 'wongwang' or 'wongwangosc' or  'wilsoncowan'
    short_run: if this parameter is True, we run the simulation for half the time, and then run the simulation at
    top time at the best found point. This is useful for testing the model.
    """
    #how many cores we have?
    print(psutil.cpu_count(logical=False))
    

    ## STUDY_NAME
    # THE NAME OF THE EXTRA STUDY, HARD-CODED
    # this will be the suffix of the output files
    studyname = 'fmridti_35'

    # HARCODED PARAMETERS
    cpu_count = 1

    # SET OF PARAMETERS
    params = {}
    params["model"] = model
    params["bold_tr"] = float(tr)
    params["simlen"] = float(simlen)

    g_low = 0.0
    g_high = 5.0
    cs_low = 0.0
    cs_high = 99999999.0

    #compute metastability and matrix of the original scan to not recalculate it each iteration

    ts_path = f'{path}/results/corrlabel_ts.txt'
    corr_ts = np.loadtxt(ts_path)
    corr_ts = corr_ts.T

    # DECOMMENT IF NEEDED
    # corr_ts = manual_bandpass(corr_ts, params["bold_tr"], freq_cutoff=0.08)
    corr_ts_meta = butter_bandpass_filter(remove_mean(corr_ts, axis=1), 0.04, 0.07, 1/(float(tr)/1000), order=1)
    metast_emp = kuromoto_metastability(remove_mean(corr_ts_meta, axis=1))

    #Create fMRI
    fMRI = np.corrcoef(corr_ts)
    fMRI = np.nan_to_num(fMRI)

    fMRI = np.arctanh(fMRI)
    infs = np.isinf(fMRI).nonzero()
    #replace the infs with 0            
    for idx in range(len(infs[0])):
        fMRI[infs[0][idx]][infs[1][idx]] = 0
    np.fill_diagonal(fMRI, 0)


    # find all of the subjects
    iterations_to_run = sorted(glob.glob(f'{out_path}/runs/*'))

    # create pool of workers
    # sequence_of_args = [ (params, x, path, metast_emp, fMRI, short_run) for x in iterations_to_run]

    # run
    # with Pool(cpu_count) as pool:
    #     list_of_results = pool.starmap(run_iterations_separate, sequence_of_args)

    list_of_results = [run_iterations_separate(params, x, path, metast_emp, fMRI, short_run) for x in iterations_to_run]

    study_deltameta = optuna.create_study()
    study_meta = optuna.create_study(direction="maximize")
    study_corr = optuna.create_study(direction="maximize")

    for dict_res in list_of_results:

        ### HAVE IT HERE ONLY TO LIMIT G        
        if dict_res["g"] > 3.5: continue

        params_t = {"g": dict_res["g"],
                  "cs": dict_res["cs"]}

        distributions = {"g": optuna.distributions.UniformDistribution(g_low, g_high),
                         "cs": optuna.distributions.UniformDistribution(cs_low, cs_high)}
        
        user_attrs = {"Meta": dict_res["Meta"],
                        "Meta_std": dict_res["Meta_std"],
                        "Metaemp": dict_res["Metaemp"],
                        "DeltaMeta": dict_res["DeltaMeta"],
                        "DeltaMeta_std": dict_res["DeltaMeta_std"],
                        "Corr": dict_res["Corr"],
                        "Corr_std": dict_res["Corr_std"],
                        "FCsyn": dict_res["FCsyn"],
                        "run_dir": dict_res["run_dir"]}

        trial_deltameta = optuna.trial.create_trial(
                           params=params_t,
                           distributions=distributions,
                           value= dict_res["DeltaMeta"],
                           user_attrs=user_attrs)
        study_deltameta.add_trial(trial_deltameta)

        trial_meta = optuna.trial.create_trial(
                           params=params_t,
                           distributions=distributions,
                           value= dict_res["Meta"],
                           user_attrs=user_attrs)
        study_meta.add_trial(trial_meta)

        trial_corr = optuna.trial.create_trial(
                           params=params_t,
                           distributions=distributions,
                           value= dict_res["Corr"],
                           user_attrs=user_attrs)
        study_corr.add_trial(trial_corr)

    # create figures and dataframe
    for (study, name) in zip([study_deltameta, study_meta, study_corr], ["DeltaMeta", "Meta", "Corr"]):
        # dataframe
        csv_study = f'{out_path}/study_{name}_{studyname}.csv'
        df = study.trials_dataframe()
        df.to_csv(csv_study)

        # figures
        fig_meta = f'{out_path}/contour_{name}_{studyname}.png'
        if model == "wongwang":
            optuna.visualization.matplotlib.plot_slice(study, params=["g"])
        else:
            optuna.visualization.matplotlib.plot_contour(study, params=["g", "cs"])
        plt.savefig(fig_meta)
        plt.close()
        
        joblib.dump(study, f"{out_path}/study_{name}_{studyname}.pkl")

if __name__ == "__main__":
    path = ""
    out_path = ""

    tr = 
    simlen = 
    model = "wongwang"
    short_run = False
    run_subj(path, out_path, tr, simlen, model, short_run)
    #  run_subj()