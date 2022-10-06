"""
Equivalent to analyze_g_cs.ipynb in jupyter-analysis, but
runs without jupyter. 

It also creates the corresponding FC matrix and its comparison with
the actual one, as it was not created during runtime, and saves it in a separate
folder to have all of them together.
"""

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from statannot import add_stat_annotation
import glob
import statsmodels.api as sm
import statsmodels.formula.api as smf
import joblib
import scipy.stats as stat
from scipy.stats import pearsonr


## paths to experiments
exp_path = sys.argv[1]
study_name = sys.argv[2]
data_path = ""
out_dir = f'{exp_path}/analysis_{study_name}'

do_boxplot_annotation = True

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

value_to_check = sys.argv[3]
best=sys.argv[3]

csv_total = ''
df_total = pd.read_csv(csv_total)

df_results = pd.DataFrame(columns=["SubjID","CENTER","DeltaMeta", "BestMeta", "BestCorr", "g", "cs"], dtype=object)
df_full_study = []
# load the results for each subject manually and add them to the csv
for subj in glob.glob(f'{exp_path}/*'):
    if not os.path.isfile(f"{subj}/{study_name}.pkl"): continue


    study = joblib.load(f"{subj}/{study_name}.pkl")
    df_study = pd.read_csv(f"{subj}/{study_name}.csv")
    df_full_study.append(df_study)
    # study.StudyDirection = "MAXIMIZE"
    CENTER, subID = os.path.basename(subj).split('_', 1)

    # get best results
    best_trial = study.best_trial

    # Print the FC matrix, together with the actual one
    # together with the SC matrix, together with the corresponding histograms,
    # and the correlation with the SC matrix
    i = 0

    fmri_list = []

    """
    fMRI = np.loadtxt(f"{data_path}/{os.path.basename(subj)}/results/r_matrix.csv", delimiter=',')
    SC = np.loadtxt(f"{data_path}/{os.path.basename(subj)}/results/{os.path.basename(subj)}_SC_weights.txt", delimiter=' ')
    
    z_fmri = np.arctanh(fMRI)
    infs = (z_fmri == np.inf).nonzero()
    #replace the infs with 0            
    for idx in range(len(infs[0])):
        z_fmri[infs[0][idx]][infs[1][idx]] = 0
    np.fill_diagonal(z_fmri, 0)

    corr_sc = fmri_corr(z_fmri, SC)

    for fmri_dir in sorted(glob.glob(f'{subj}/runs/{best_trial.params["cs"]}*_{best_trial.params["g"]}*/FCsyn_*.txt')):
        plt.figure(figsize=(20, 15))
        fMRI_syn = np.genfromtxt(f"{fmri_dir}", delimiter=' ')
        print(fmri_dir)
        #compute the zfisher correlation
        z_fmri_syn = np.arctanh(fMRI_syn)
        infs = (z_fmri_syn == np.inf).nonzero()
        #replace the infs with 0            
        for idx in range(len(infs[0])):
            z_fmri_syn[infs[0][idx]][infs[1][idx]] = 0
        np.fill_diagonal(z_fmri_syn, 0)

        # compute correlation of both matrices with SC
        corr = best_trial.user_attrs["Corr"]
        corr_fc = fmri_uncentered_corr(z_fmri, z_fmri_syn)
        fmri_list.append(fMRI_syn)

        fig_dir2 = f'{FC_dir}/{os.path.basename(subj)}_{i}.png'
        plt.subplot(231), plt.imshow(fMRI_syn, interpolation='none'), plt.colorbar(), plt.title(f"Simulated FC\n {corr_fc = :.3f}")
        plt.subplot(232), plt.imshow(fMRI, interpolation='none'), plt.colorbar(), plt.title("Empirical FC")
        plt.subplot(233), plt.imshow(SC, interpolation='none'), plt.colorbar(), plt.title(f"Empirical SC\n {corr_sc = :.3f}")
        plt.subplot(234), plt.hist(fMRI_syn), plt.colorbar(), plt.title(f"Histogram sim. FC")
        plt.subplot(235), plt.hist(fMRI), plt.colorbar(), plt.title(f"Histogram emp. FC")
        plt.subplot(236), plt.hist(SC), plt.colorbar(), plt.title(f"Histogram emp. SC")
        plt.savefig(fig_dir2)
        plt.close()
        i+=1

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
        fig_dir2 = f'{FC_dir}/{os.path.basename(subj)}_avg.png'
        plt.subplot(231), plt.imshow(fmri_avg, interpolation='none'), plt.colorbar(), plt.title(f"Simulated FC\n {corr_fc = :.3f}")
        plt.subplot(232), plt.imshow(fMRI, interpolation='none'), plt.colorbar(), plt.title("Empirical FC")
        plt.subplot(233), plt.imshow(SC, interpolation='none'), plt.colorbar(), plt.title(f"Empirical SC\n {corr_sc = :.3f}")
        plt.subplot(234), plt.hist(fmri_avg), plt.colorbar(), plt.title(f"Histogram sim. FC")
        plt.subplot(235), plt.hist(fMRI), plt.colorbar(), plt.title(f"Histogram emp. FC")
        plt.subplot(236), plt.hist(SC), plt.colorbar(), plt.title(f"Histogram emp. SC")
        plt.savefig(fig_dir2)
        plt.close()
    """

    data_to_append = {"SubjID": subID, 
                      "CENTER": CENTER, 
                      "DeltaMeta": best_trial.user_attrs["DeltaMeta"], 
                      "BestMeta": best_trial.user_attrs["Meta"], 
                      "BestCorr": best_trial.user_attrs["Corr"],
                      # "CorrSC": corr_sc,
                      # "CorrFC": corr_fc,
                      "g": best_trial.params["g"], 
                      "cs": best_trial.params["cs"]}
    df_results = df_results.append(data_to_append, ignore_index = True)


df_full_results = pd.concat(df_full_study)
df_full_results.to_csv(f'{exp_path}/{study_name}_full_results_all.csv')

csv_results = f'{exp_path}/{study_name}_results_all.csv'
df_results.to_csv(csv_results)

"""
# merge results and total
df_merged = df_results.merge(df_total, on=["SubjID", "CENTER"])

print(df_merged["GROUP"].describe())
df_merged.head()

### Distribution check
# Visualize distribution of the g and cs in the groups
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,5))
plt.suptitle("Distribution plot on coupling (G) and conduction speed (cs)", y=1.2)
sns.kdeplot(data=df_merged, x="g", hue="GROUP", ax=ax1)
sns.kdeplot(data=df_merged, x="cs", hue="GROUP", ax=ax2)
plt.savefig(f'{out_dir}/distribution_plot.png')
plt.close()

###################### COUPLING ###########################
## Create various boxplots with different groups
# and check for significance with two tailed t-test

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15,5))
# plt.suptitle("Comparisons on coupling (G)", y=1.2)

# other test is Mann-Whitney
test = 't-test_ind'
# test = 'Mann-Whitney'

### BY GROUPS
sns.set(style="whitegrid")
sns.boxplot(data=df_merged, x="GROUP", y="g", order=["HC", "RRMS", "SPMS", "PPMS"], ax=ax1)
sns.swarmplot(data=df_merged, x="GROUP", y="g", order=["HC", "RRMS", "SPMS", "PPMS"], ax=ax1, color=".2")
if do_boxplot_annotation:
    add_stat_annotation(ax1, data=df_merged, x="GROUP", y="g", order=["HC", "RRMS", "SPMS", "PPMS"],
                        box_pairs=[("HC", "RRMS"), ("HC", "SPMS"), ("HC", "PPMS"), ("RRMS", "SPMS"), ("RRMS", "PPMS"), ("SPMS", "PPMS")],
                        test=test, comparisons_correction=None, text_format='star', loc='outside', verbose=1)
# BY HC - MS
df_merged['disease'] = np.where(df_merged['GROUP']== 'HC', "HC", "MS")
sns.set(style="whitegrid")
sns.boxplot(data=df_merged, x="disease", y="g", ax=ax2)
sns.swarmplot(data=df_merged, x="disease", y="g", ax=ax2, color=".2")
if do_boxplot_annotation:
    add_stat_annotation(ax2, data=df_merged, x="disease", y="g",
                        box_pairs=[("HC", "MS")], comparisons_correction=None,
                        test=test, text_format='star', loc='outside', verbose=1)
# BY EDSS
df_merged['EDSSbin'] = np.where(df_merged['EDSS'] < 3, "EDSS<3", "EDSS>=3")
sns.set(style="whitegrid")
sns.boxplot(data=df_merged, x="EDSSbin", y="g", ax=ax3)
sns.swarmplot(data=df_merged, x="EDSSbin", y="g", ax=ax3, color=".2")
if do_boxplot_annotation:
    add_stat_annotation(ax3, data=df_merged, x="EDSSbin", y="g",
                        box_pairs=[("EDSS<3", "EDSS>=3")], comparisons_correction=None,
                        test=test, text_format='star', loc='outside', verbose=1)
# BY SEX
sns.set(style="whitegrid")
sns.boxplot(data=df_merged, x="SEX", y="g", ax=ax4)
sns.swarmplot(data=df_merged, x="SEX", y="g", ax=ax4, color=".2")
if do_boxplot_annotation:
    add_stat_annotation(ax4, data=df_merged, x="SEX", y="g",
                        box_pairs=[("M", "F")], comparisons_correction=None,
                        test=test, text_format='star', loc='outside', verbose=1)
plt.savefig(f'{out_dir}/g_boxplot.png', bbox_inches="tight")
plt.close()


### Scatterplot G
## Compare different values 
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,5))
sns.regplot(data=df_merged, x="g", y="EDSS", ax=ax1)
sns.regplot(data=df_merged, x="g", y="SDMT", ax=ax2)
sns.regplot(data=df_merged, x="g", y="DD", ax=ax3)
plt.tight_layout()
plt.savefig(f'{out_dir}/g_scatterplot.png')
plt.close()

###################### CONDUCTION SPEED ###########################
## Create various boxplots with different groups
# and check for significance with two tailed t-test
# same as above, but with conduction speed differences

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,5))
# plt.suptitle("Comparisons on conduction speed (cs)", y=1.2)

# other test is Mann-Whitney
# test = 't-test_ind'
test = 'Mann-Whitney'

### BY GROUPS
sns.set(style="whitegrid")
sns.boxplot(data=df_merged, x="GROUP", y="cs", order=["HC", "RRMS", "SPMS", "PPMS"], ax=ax1)
sns.swarmplot(data=df_merged, x="GROUP", y="cs", order=["HC", "RRMS", "SPMS", "PPMS"], ax=ax1, color=".2")
if do_boxplot_annotation:
    add_stat_annotation(ax1, data=df_merged, x="GROUP", y="cs", order=["HC", "RRMS", "SPMS", "PPMS"],
                        box_pairs=[("HC", "RRMS"), ("HC", "SPMS"), ("HC", "PPMS"), ("RRMS", "SPMS"), ("RRMS", "PPMS"), ("SPMS", "PPMS")],
                        test=test, comparisons_correction=None, text_format='star', loc='outside', verbose=1)
# BY HC - MS
df_merged['disease'] = np.where(df_merged['GROUP']== 'HC', "HC", "MS")
sns.set(style="whitegrid")
sns.boxplot(data=df_merged, x="disease", y="cs", ax=ax2)
sns.swarmplot(data=df_merged, x="disease", y="cs", ax=ax2, color=".2")
if do_boxplot_annotation:
    add_stat_annotation(ax2, data=df_merged, x="disease", y="cs",
                        box_pairs=[("HC", "MS")], comparisons_correction=None,
                        test=test, text_format='star', loc='outside', verbose=1)
# BY EDSS
df_merged['EDSSbin'] = np.where(df_merged['EDSS'] <= 3, "EDSS<=3", "EDSS>3")
sns.set(style="whitegrid")
sns.boxplot(data=df_merged, x="EDSSbin", y="cs", ax=ax3)
sns.swarmplot(data=df_merged, x="EDSSbin", y="cs", ax=ax3, color=".2")
if do_boxplot_annotation:
    add_stat_annotation(ax3, data=df_merged, x="EDSSbin", y="cs",
                        box_pairs=[("EDSS<=3", "EDSS>3")], comparisons_correction=None,
                        test=test, text_format='star', loc='outside', verbose=1)
# BY SEX
sns.set(style="whitegrid")
sns.boxplot(data=df_merged, x="SEX", y="cs", ax=ax4)
sns.swarmplot(data=df_merged, x="SEX", y="cs", ax=ax4, color=".2")
if do_boxplot_annotation:
    add_stat_annotation(ax4, data=df_merged, x="SEX", y="cs",
                        box_pairs=[("M", "F")], comparisons_correction=None,
                        test=test, text_format='star', loc='outside', verbose=1)
plt.savefig(f'{out_dir}/cs_boxplot.png', bbox_inches="tight")
plt.close()

## SCATTERPLOT CS
fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(20,5))
sns.regplot(data=df_merged, x="cs", y="EDSS", ax=ax1)
sns.regplot(data=df_merged, x="cs", y="SDMT", ax=ax2)
sns.regplot(data=df_merged, x="cs", y="DD", ax=ax3)
plt.savefig(f'{out_dir}/cs_scatterplot.png')
plt.close()


###################### METASTABILITY ###########################
## Create various boxplots with different groups
# and check for significance with two tailed t-test
# same as above, but with conduction speed differences

fig, ((ax1, ax2, ax3, ax4)) = plt.subplots(1, 4, figsize=(20,5))
# plt.suptitle(f"Comparisons on {value_to_check}", y=1.2)

# other test is Mann-Whitney
test = 't-test_ind'
# test = 'Mann-Whitney'

### BY GROUPS
sns.set(style="whitegrid")
sns.boxplot(data=df_merged, x="GROUP", y=best, order=["HC", "RRMS", "SPMS", "PPMS"], ax=ax1)
sns.swarmplot(data=df_merged, x="GROUP", y=best, order=["HC", "RRMS", "SPMS", "PPMS"], ax=ax1, color=".2")
if do_boxplot_annotation:
    add_stat_annotation(ax1, data=df_merged, x="GROUP", y=best, order=["HC", "RRMS", "SPMS", "PPMS"],
                        box_pairs=[("HC", "RRMS"), ("HC", "SPMS"), ("HC", "PPMS"), ("RRMS", "SPMS"), ("RRMS", "PPMS"), ("SPMS", "PPMS")],
                        test=test, comparisons_correction=None, text_format='star', loc='outside', verbose=1)
# BY HC - MS
df_merged['disease'] = np.where(df_merged['GROUP']== 'HC', "HC", "MS")
sns.set(style="whitegrid")
sns.boxplot(data=df_merged, x="disease", y=best, ax=ax2)
sns.swarmplot(data=df_merged, x="disease", y=best, ax=ax2, color=".2")
if do_boxplot_annotation:
    add_stat_annotation(ax2, data=df_merged, x="disease", y=best,
                        box_pairs=[("HC", "MS")], comparisons_correction=None,
                        test=test, text_format='star', loc='outside', verbose=1)
# BY EDSS
df_merged['EDSSbin'] = np.where(df_merged['EDSS'] <= 3, "EDSS<=3", "EDSS>3")
sns.set(style="whitegrid")
sns.boxplot(data=df_merged, x="EDSSbin", y=best, ax=ax3)
sns.swarmplot(data=df_merged, x="EDSSbin", y=best, ax=ax3, color=".2")
if do_boxplot_annotation:
    add_stat_annotation(ax3, data=df_merged, x="EDSSbin", y=best,
                        box_pairs=[("EDSS<=3", "EDSS>3")], comparisons_correction=None,
                        test=test, text_format='star', loc='outside', verbose=1)
# BY SEX
sns.set(style="whitegrid")
sns.boxplot(data=df_merged, x="SEX", y=best, ax=ax4)
sns.swarmplot(data=df_merged, x="SEX", y=best, ax=ax4, color=".2")
if do_boxplot_annotation:
    add_stat_annotation(ax4, data=df_merged, x="SEX", y=best,
                        box_pairs=[("M", "F")], comparisons_correction=None,
                        test=test, text_format='star', loc='outside', verbose=1)
plt.savefig(f"{out_dir}/meta_boxplot.png", bbox_inches="tight")
plt.close()

## LINK BETWEEN CS AND G
sns.scatterplot(data=df_merged, x="cs", y="g", alpha=0.8, s=75)
plt.savefig(f"{out_dir}/cs_g_scatter.png", bbox_inches="tight")
plt.close()


###BOXPLOT CORRELATIONS
sns.set(style="whitegrid")
plt.figure(figsize=(15, 10))
plt.boxplot([df_merged["BestMeta"], df_merged["BestCorr"]], labels=["BestMeta", "BestCorr"])
plt.savefig(f'{out_dir}/metacorr_boxplot.png', bbox_inches="tight")
plt.close()

### CURVES
# they are in the results section, no need to create them
# todo: if needed, we can addd the univariatespline 

# those are not the curves that we want to plot
# plot the curves over all the experiments, not just the best ones
plt.figure(figsize=(15, 10))
sns.lineplot(data=df_merged, x="cs", y=value_to_check, hue="GROUP")
plt.savefig(f"{out_dir}/cs_vs_GROUP_curve.png")
plt.close()

df_merged['disease'] = np.where(df_merged['GROUP']== 'HC', "HC", "MS")
plt.figure(figsize=(15, 10))
sns.lineplot(data=df_merged, x="cs", y=value_to_check, hue="disease")
plt.savefig(f"{out_dir}/cs_vs_disease_curve.png")
plt.close()

df_merged['EDSSbin'] = np.where(df_merged['EDSS'] < 3, "EDSS<3", "EDSS>=3")
plt.figure(figsize=(15, 10))
sns.lineplot(data=df_merged, x="cs", y=value_to_check, hue="EDSSbin")
plt.savefig(f"{out_dir}/cs_vs_EDSS_curve.png")
plt.close()
"""