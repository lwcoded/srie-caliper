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
    raise ValueError(f"model_name {args.model_name!r} contains no usable characters for file names")
if model_name_safe != args.model_name:
    print(f"model name {args.model_name!r} sanitised to {model_name_safe!r} for use in file names")

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

correlation = plotting_df["sd"].corr(plotting_df["score"])
print(f"correlation between std. dev. of human responses and sycophancy score = {correlation:.3}")