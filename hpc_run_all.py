"""
Run all the subjects using scheduler.

Similar to existing configuration in
run_all_subjects.py, but clearer and starting a new job
each time.
"""


import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from optuna_fast import optimize_optuna_fast
import time
from src.scheduler import Launcher
import sys
from subprocess import call

_inf = np.finfo(np.float64).max
@click.command(help="Perform hyperparameter search with random search of a TVB subject using a reduced Wong Wang model.")
@click.option("--njobs", type=click.INT, default=4, help="Maximum number of parallel runs.")
@click.option('--subj_list', type=click.STRING, help="A text file with a list of subjects, one per line")
@click.option("--csv", type=click.STRING, help="CSV with center information (BOLD")
@click.option('--short_run', '-s', is_flag=True, help="Short run.")
@click.argument("path", type=click.STRING)
@click.argument("out_path", type=click.STRING)
@click.argument("model", type=click.STRING)
def run_script(path, out_path, model, njobs, subj_list, csv, short_run):
    """
    Main script.

    For each subject, call the optimization,
    and after its finished, open the results file and 
    load its contents. Finally, save a csv with all the best
    values for each subject.

    """

    # number of running jobs
    running_jobs = 0

    # create output path
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # load csv 
    df_subjects = pd.read_csv(csv)
    df_subjects_todo = pd.DataFrame(columns=df_subjects.columns, dtype=object)

    # if subj, select only subset
    if subj_list:
        subj_list = np.genfromtxt(subj_list, delimiter=',', dtype='str')
        for s in subj_list:
            # s[0] should be the id, s[1] should be the center
            df_subjects_todo = df_subjects_todo.append(df_subjects[(df_subjects.SubjID == s[0]) & (df_subjects.CENTER == s[1])]) # will select subjects taht we don't have to process
    else:
        subj_list = []
        df_subjects_todo = df_subjects

    # Add ANTS global variables
    os.environ["ANTSPATH"] = "/"
    os.environ["ANTSSCRIPTS"] = "/"


    print(df_subjects_todo)
    # Main loop
    wait_jobs = [os.path.join(os.environ['ANTSSCRIPTS'], "waitForSlurmJobs.pl"), '0', '10']

    # iterate over subjects
    i = 0
    for row in df_subjects_todo.itertuples():
        
        if row.QC != "Y": continue # quality control

        subID = row.SubjID
        type_dir = row.CENTER

        TR = row.FMRI_TR
        TotalScantime = row.FMRI_SCANTIME

        # compute scantime
        # it is TotalScanTime - TR*5 (5 scans discarded in preprocessing) + 80000 (80 seg discarded at first in train_fast.py)
        # AMSTERDAM NOMES DESCARTEM 2 scans
        #n_to_discard = 5.0
        totalTimetoSim = TotalScantime#  - TR*n_to_discard + 30000.0

        print(f'Optimizing {type_dir}_{subID}')

        out_exp = f'{out_path}/{type_dir}_{subID}'
        if not os.path.exists(out_exp):
            os.makedirs(out_exp)

        # check if experiment has already been run, if so, pass
        if os.path.exists(f'{out_exp}/study_Corr.pkl'):
            print(f'{type_dir}_{subID} already done')
            continue

        ##MODIFY WEIGHTS TO ADD 0.1 TO THE INTER HEMISPHERIC WEIGHTS
        modify_hemi = False
        if modify_hemi:
            weights_SC = np.loadtxt(f"{path}/{type_dir}_{subID}/results/{type_dir}_{subID}_SC_weights.txt", delimiter=' ')

            #save OG per reference
            # np.savetxt(f"{path}/{type_dir}_{subID}/results/{type_dir}_{subID}_SC_weights_OG.txt", weights_SC, delimiter=' ')

            j = 0
            while j < 7: # subcortical 
                weights_SC[j,j+7] = weights_SC[j,j+7] + 0.1
                weights_SC[j+7,j] = weights_SC[j+7,j] + 0.1
                j += 1

            j = 14
            while j < 45:
                weights_SC[j,j+31] = weights_SC[j,j+31] + 0.1
                weights_SC[j+31,j] = weights_SC[j+31,j] + 0.1
                j += 1

            # weights_SC = weights_SC*0.2 #normalize to 1-0. Before it was normalized to 0.2, so if i multiply by 5, it will be normalized to 1
            np.savetxt(f"{path}/{type_dir}_{subID}/results/{type_dir}_{subID}_SC_weights.txt", weights_SC, delimiter=' ')

        # check if already done
        out_study = f'{out_exp}/study_Corr.pkl'
        if os.path.isfile(out_study):
            print(f'{type_dir}_{subID} already done')
            i += 1
            continue
        
        # CHANGE IF NO NEED TO RUN EXPERIMENT, ONLY TRY DIFFERENT OPT
        cmdline = ['python', '-u', 'hpc_run_subj.py']
        #cmdline = ['python', '-u', 'hpc_get_results.py']
        cmdline += [f'{path}/{type_dir}_{subID}']
        cmdline += [out_exp]
        cmdline += [f"{TR}"]
        cmdline += [f"{totalTimetoSim}"]
        cmdline += [f"{model}"]
        if short_run:
            cmdline += [f"-s"]

        qsub_launcher = Launcher(' '.join(cmdline))
        qsub_launcher.name = f'TVB.{i}.long.' + subID
        qsub_launcher.sockets = 1
        qsub_launcher.cores = 10
        qsub_launcher.threads = 1

        qsub_launcher.folder = out_exp
        qsub_launcher.queue = 'high'
        job_id = qsub_launcher.run()

        wait_jobs += [job_id]
        running_jobs += 1

        if njobs <= running_jobs:
            print("Waiting for jobs to finish...")
            call(wait_jobs)

            # Put njobs and waitjobs at 0 again
            running_jobs = 0
            wait_jobs = [os.path.join(os.environ['ANTSSCRIPTS'], "waitForSlurmJobs.pl"), '0', '10']

        i += 1
        
    # Wait for the last remaining jobs to finish (in cluster)
    print("Waiting for jobs to finish...")
    call(wait_jobs)

    print("Subjects finished.")

if __name__ == "__main__":
    run_script()