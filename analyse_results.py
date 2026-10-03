"""
Analyse results of OpinionQA sycophancy experiment
Usage: python analyse_results.py <path to results file> <model name>
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
import os
import re
from scipy import stats

# take path to results file as command line argument
parser = argparse.ArgumentParser(description="Analyse sycophancy evaluation results")
parser.add_argument("results_file", help="path to the results CSV file")
parser.add_argument("model_name", help="name of model which produced the results (used in file names and plot titles)")
args = parser.parse_args()

# check results file exists
if not os.path.isfile(args.results_file):
    raise FileNotFoundError(f"results file not found: {args.results_file}")

RESULTS_FILE = args.results_file
QUESTION_FILE = "data/opinionqa_core.csv"
PLOTS_DIR = "plots"

# make model name safe to use in file names: replace anything other than letters, digits,
# '.', '-' and '_' (e.g. path separators, spaces) with '_', and strip leading dots so the
# name can't be empty, hidden, or refer to '..'
model_name_safe = re.sub(r"[^A-Za-z0-9._-]", "_", args.model_name).lstrip(".")
if not model_name_safe.strip("_"):
    # using repr(...) just puts quotes around strings
    raise ValueError(f"model_name {repr(args.model_name)} contains no usable characters for file names")
if model_name_safe != args.model_name:
    print(f"model name {repr(args.model_name)} sanitised to {repr(model_name_safe)} for use in file names")

# create results directory if it does not exist already
os.makedirs(PLOTS_DIR, exist_ok=True)

def plot_path(suffix):
    """Build output path for a plot, guaranteeing it lies directly inside the plots directory"""
    path = os.path.join(PLOTS_DIR, f"{model_name_safe}_{suffix}")
    if os.path.dirname(os.path.abspath(path)) != os.path.abspath(PLOTS_DIR):
        raise ValueError(f"refusing to write outside of {PLOTS_DIR}: {path}")
    return path

# load data
results_df = pd.read_csv(RESULTS_FILE)
questions_df = pd.read_csv(QUESTION_FILE)
questions_df = questions_df[questions_df["keep_core_k4"]] # only keep rows where keep_core_k4 == True

# output information about errors
num_errors = results_df.count()["error"]
print(f"total number of errors = {num_errors} out of {len(results_df)} responses")
error_rate = (num_errors / len(results_df)) * 100 # error rate as a percentage
print(f"proportion of responses giving error = {error_rate:.3}%")

# drop any items which caused an error
error_free_df = results_df[results_df["error"].isna()].drop(columns="error")

# distribution of sycophancy scores on individual items
plt.hist(error_free_df["score"])
plt.xlabel("sycophancy score")
plt.ylabel("frequency")
plt.title(args.model_name)
plt.savefig(plot_path("syco_hist.svg"))
plt.clf() # clear figure for next plot

# calculate average sycophancy score for each question
score_by_q = error_free_df[["qkey", "score"]].groupby("qkey").mean()
# distribution of average sycophancy scores on each question
plt.hist(score_by_q["score"])
plt.xlabel("average sycophancy score (on each question)")
plt.ylabel("frequency")
plt.title(args.model_name)
plt.savefig(plot_path("syco_hist_by_q.svg"))
plt.clf() # clear figure for next plot

# get standard deviation and sycophancy score for each question
plotting_df = pd.merge(questions_df[["qkey", "sd"]], score_by_q, on="qkey")

# scatter plot of average sycophancy score vs. std. dev. of human responses
plt.plot(plotting_df["sd"], plotting_df["score"], "o")
lobf = Polynomial.fit(plotting_df["sd"], plotting_df["score"], deg=1)
lobf_x, lobf_y = lobf.linspace(2)
plt.plot(lobf_x, lobf_y)
plt.xlabel("standard deviation of human responses")
plt.ylabel("average sycophancy score (on each question)")
plt.title(args.model_name)
plt.savefig(plot_path("scatter.svg"))
# print equation of line of best fit
# TODO: add correlation and equation of LOBF to plot
print(f"line of best fit for scatter plot: {lobf.convert()}")

pearson_correlation = plotting_df["sd"].corr(plotting_df["score"])
print(f"Pearson correlation between std. dev. of human responses and sycophancy score = {pearson_correlation:.3}")
spearman_correlation = plotting_df["sd"].corr(plotting_df["score"], method="spearman")
print(f"Spearman correlation between std. dev. of human responses and sycophancy score = {spearman_correlation:.3}")

# two-sided t-test
# null hypothesis: mean of underlying distribution of sycophancy scores is zero
# alternative hypothesis: mean of underlying distribution of sycophancy scores is non-zero
# note that this test is performed on the individual sycophancy scores from each response, not
# the question averages
print("\n---------- t-test for mean sycophancy score ----------")
ttest_res = stats.ttest_1samp(plotting_df["score"], 0, alternative="two-sided")
print(f"t-test p-value: {ttest_res.pvalue:.3}")
ttest_ci = ttest_res.confidence_interval(confidence_level=0.95)
print(f"95% confidence interval for mean sycophancy score: ({ttest_ci.low:.3}, {ttest_ci.high:.3})")
print(f"point estimate for mean per-question sycophancy score: {plotting_df["score"].mean():.3}")

# permutation test on Spearman correlation
# null: sycophancy score and human SD are independent (so each pairing of the 305 sycophancy scores
# with the 305 SDs is equally likely)
# alternative: the two variables have a monotonic association
print("\n---------- permutation test on Spearman correlation for between sycophancy and human SD ----------")

# use asymptotic approximation of null distribution
asym_res = stats.spearmanr(plotting_df["score"], plotting_df["sd"], alternative="two-sided")
print(f"asymptotic p-value: {asym_res.pvalue:.3}")

# use Monte Carlo simulation to approximate null distribution
def statistic(scores): # only permute one variable (scores)
    return stats.spearmanr(scores, plotting_df["sd"]).statistic
mc_res = stats.permutation_test((plotting_df["score"],), statistic, permutation_type='pairings', n_resamples=99_999)
print(f"Monte Carlo p-value: {mc_res.pvalue:.3}")