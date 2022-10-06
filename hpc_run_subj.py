"""
Run a single subject.

Minimum overhead, paralelize grid search using numba and jit.

Call C program.
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

def run_iterations_separate(g, cs, params, out_dir, subj_path, metast_emp, fMRI, short_run=False):
    """
    parallel function

    calls to the C program N times, instead of doing a very long simulation
    """
    bold_tr = params["bold_tr"]
    simlen = params["simlen"]
    j_ndma = params["j_ndma"]
    w_plus = params["w_plus"]
    ji = params["j_i"]
    noise = params["noise"]
    model = params["model"]

    # compute scantime
    # it is TotalScanTime - TR*5 (5 scans discarded in preprocessing) + 80000 (80 seg discarded at first in train_fast.py)
    # AMSTERDAM NOMES DESCARTEM 2 scans
    if "AMSTERDAM" in subj_path: n_to_discard = 2.0
    else: n_to_discard = 5.0

    if short_run: simlen = simlen*0.5
    simlen = simlen - bold_tr*n_to_discard + 30000.0

    subjID = os.path.basename(subj_path.rstrip('/'))

    t_to_discard = int(np.ceil(30000. / bold_tr))

    cs = '%.5f'%(cs)
    g = '%.5f'%(g)

    njobs = 1
    # create directory
    out_exp = f'{out_dir}/runs/{cs}_{g}'
    if not os.path.exists(out_exp):
        os.makedirs(out_exp)

    N = 10
    metast_arr = []
    deltameta_arr = []
    corr_arr = []
    
    z_fmri = fMRI
    for i in range(N):
        seed = int.from_bytes(os.urandom(2), byteorder="big")

        tic = time.time()
        # Create parameter file
        param_set_file = f'{subj_path}/results/param_set_{cs}_{g}_{seed}.txt'

        # Wilson-cowan
        if model == "wilsoncowan":
            param_set = f"76 {g} {w_plus} {noise} {int(simlen)} {int(bold_tr)} {cs} {seed} {ji}"
        else:
            param_set = f"76 {g} {j_ndma} {w_plus} {ji} {noise} {int(simlen)} {int(bold_tr)} {cs} {seed}"

        with open(param_set_file, "w") as text_file:
            print(param_set, file=text_file)

        # name of the set of params could change if tal
        #instead of os system, do popen
        _ = Popen(f"sh start_simulation.sh {subj_path}/results {out_exp} param_set_{cs}_{g}_{seed}.txt {subjID} {njobs} {out_dir} >/dev/null 2>&1", shell=True).wait() #  

        # Read and load the result
        txt_path = f'{out_exp}/{subjID}_param_set_{cs}_{g}_{seed}.txt_fMRI.txt'
        #txt_path = glob.glob(f'{out_exp}/{subjID}_param_set_{cs}_{g}_*.txt_fMRI.txt')[i]

        # load the result
        BOLD_syn = np.genfromtxt(txt_path, delimiter=' ')
        # discard first 80 sec
        BOLD_syn_i = BOLD_syn[:,t_to_discard:]

        # select the part of the bold signal
        #BOLD_syn_i = manual_bandpass(BOLD_syn_i, bold_tr, freq_cutoff=0.08)
        BOLD_syn_i_butt = butter_bandpass_filter(remove_mean(BOLD_syn_i, axis=1), 0.04, 0.07, 1/(bold_tr/1000), order=1)

        # BOLD_syn_i_lowpass = BOLD_syn_i
        metast = kuromoto_metastability(remove_mean(BOLD_syn_i_butt, axis=1))

        diff_meta = np.abs(metast_emp - metast)
        fMRI_syn = np.corrcoef(BOLD_syn_i)
        fMRI_syn = np.nan_to_num(fMRI_syn)
        # fMRI_syn = (fMRI_syn + fMRI_syn.T) / 2.0
        np.savetxt(f'{out_exp}/FCsyn_{i}.txt', fMRI_syn)
        #np.savetxt(f'{out_exp}/BOLDsyn_{i}.txt', BOLD_syn_i)

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

        os.system(f'rm -rf {param_set_file}')

        tac = time.time()

    # prepare return
    dict_return = {
        "run_dir": out_exp,
        "g" : float(g),
        "cs" : float(cs),
        "Meta": np.mean(metast_arr),
        "Meta_std": np.std(metast_arr),
        "Metaemp": metast_emp,
        "DeltaMeta": np.mean(deltameta_arr),
        "DeltaMeta_std": np.std(deltameta_arr),
        "Corr": np.mean(corr_arr),
        "Corr_std": np.std(corr_arr),
        "FCsyn": f'{out_exp}/FCsyn_0.txt'
    }

    return dict_return


@click.command(help="Perform hyperparameter search with a single subject using the fast-tvb model.")
@click.option('--short_run', '-s', is_flag=True, help="Short run.")
@click.argument("path", type=click.STRING)
@click.argument("out_path", type=click.STRING)
@click.argument("tr", type=click.STRING)
@click.argument('simlen', type=click.STRING)
@click.argument('model', type=click.STRING)
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
    
    # HARCODED PARAMETERS
    cpu_count = 10

    # SET OF PARAMETERS
    params = {}
    params["model"] = model

    if model == "wongwang":
        params["j_ndma"] = 0.15
        params["w_plus"] = 1.4
        params["j_i"] = 1.0
        params["noise"] = 0.001 # 0.00316228
        # HARDCODED
        g_low = 0
        g_high = 10.0
        cs_low = 99999999.0
        cs_high = 99999999.0

        num_g = 51
        num_cs = 1

    elif model == "wongwangosc":
        params["j_ndma"] = 1.271
        params["w_plus"] = 2.4
        params["j_i"] = 3.099
        params["noise"] = 0.001 # 0.00316228
        # HARDCODED
        g_low = 0.01
        g_high = 2.0
        cs_low = 0.1
        cs_high = 40.0

        num_g = 20
        num_cs = 20

    elif model == "wilsoncowan":
        params["j_ndma"] = -1
        params["w_plus"] = 1.5
        params["j_i"] = 1.5
        params["noise"] = 0.1 # 0.00316228 #0.01 
        # HARDCODED
        g_low = 0.05
        g_high = 1.5
        cs_low = 0.1
        cs_high = 30.0

        num_g = 20
        num_cs = 20

    params["bold_tr"] = float(tr)
    params["simlen"] = float(simlen)

    # compile tvb
    os.system(f"sh compile_and_copy.sh {out_path} {model}")

    ## Do loops in parallel
    # 484 iterations
    cs_list = np.linspace(cs_low, cs_high, num=num_cs)
    g_list = np.linspace(g_low, g_high, num=num_g)
    print(g_list)

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

    # create pool of workers
    iterations_to_run = itertools.product(g_list, cs_list)
    sequence_of_args = [ (x[0], x[1], params, out_path, path, metast_emp, fMRI, short_run) for x in iterations_to_run]

    # run
    with Pool(cpu_count) as pool:
        list_of_results = pool.starmap(run_iterations_separate, sequence_of_args)

    #create three different studies, one with each different objective
    # check first if such studies exists, if they do, load them
    if os.path.exists(f"{out_path}/study_DeltaMeta.pkl"):
        study_deltameta = joblib.load(f"{out_path}/study_DeltaMeta.pkl")
    else:
        study_deltameta = optuna.create_study()

    if os.path.exists(f"{out_path}/study_Meta.pkl"):
        study_meta = joblib.load(f"{out_path}/study_Meta.pkl")
    else:
        study_meta = optuna.create_study(direction="maximize")

    if os.path.exists(f"{out_path}/study_Corr.pkl"):
        study_corr = joblib.load(f"{out_path}/study_Corr.pkl")
    else:
        study_corr = optuna.create_study(direction="maximize")

    for dict_res in list_of_results:
        
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
        csv_study = f'{out_path}/study_{name}.csv'
        df = study.trials_dataframe()
        df.to_csv(csv_study)

        # figures
        fig_meta = f'{out_path}/contour__{name}.png'
        if model == "wongwang":
            optuna.visualization.matplotlib.plot_slice(study, params=["g"])
        else:
            optuna.visualization.matplotlib.plot_contour(study, params=["g", "cs"])
        plt.savefig(fig_meta)
        plt.close()
        
        joblib.dump(study, f"{out_path}/study_{name}.pkl")

        #if short run, we need to run an extra simulation at the best point for each study
        if short_run:
            best_point = study.best_trial
            print(best_point)
            g = best_point.params["g"]
            cs = best_point.params["cs"]
            params["g"] = g
            params["cs"] = cs
            best_result = run_iterations_separate(g, cs, params, out_path, path, metast_emp, fMRI, short_run=False)
            #plot_full_FC(path, best_result["run_dir"], f"{out_path}/{name}", fMRI)
        else:
            best_point = study.best_trial
            # need to call function to plot and save it
            plot_full_FC(path, best_point.user_attrs["run_dir"], f"{out_path}/{name}", fMRI)
if __name__ == "__main__":
    
    # run_subj(path, out_path, tr, simlen, model, short_run)
    run_subj()