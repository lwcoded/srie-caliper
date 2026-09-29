"""
Analyse results of OpinionQA sycophancy experiment
Usage: python analyse_results.py <path to results file>
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
import os

# take path to results file as command line argument
parser = argparse.ArgumentParser(description="Analyse sycophancy evaluation results")
parser.add_argument("results_file", help="path to the results CSV file")
args = parser.parse_args()

# check results file exists
if not os.path.isfile(args.results_file):
    raise FileNotFoundError(f"results file not found: {args.results_file}")

RESULTS_FILE = args.results_file
QUESTION_FILE = "data/opinionqa_core.csv"

# create results directory if it does not exist already
os.makedirs("plots", exist_ok=True)

# load data
results_df = pd.read_csv(RESULTS_FILE)
questions_df = pd.read_csv(QUESTION_FILE)
questions_df = questions_df[questions_df["keep_core_k4"]] # only keep rows where keep_core_k4 == True

# output information about errors
num_errors = results_df.count()["error"]
print(f"total number of errors = {num_errors} out of {len(results_df)} responses")
error_rate = (num_errors / len(results_df)) * 100 # error rate as a percentage
print(f"proportion of responses giving error = {error_rate:.3}%")

# TODO: should we completely drop questions which have an error on any response, or only drop that specific response?
error_free_df = results_df[results_df["error"].isna()].drop(columns="error")

# distribution of sycophancy scores on individual items
# TODO: add axis labels to all plots
plt.hist(error_free_df["score"])
# TODO: change filenames according to which model's results we are using
plt.savefig("plots/syco_hist.svg")
plt.clf() # clear figure for next plot

# calculate average sycophancy score for each question
score_by_q = error_free_df[["qkey", "score"]].groupby("qkey").mean()
# distribution of average sycophancy scores on each question
plt.hist(score_by_q["score"])
plt.savefig("plots/syco_hist_by_q.svg")
plt.clf() # clear figure for next plot

# get standard deviation and sycophancy score for each question
plotting_df = pd.merge(questions_df[["qkey", "sd"]], score_by_q, on="qkey")

# scatter plot of average sycophancy score vs. std. dev. of human responses
plt.plot(plotting_df["sd"], plotting_df["score"], "o")
lobf = Polynomial.fit(plotting_df["sd"], plotting_df["score"], deg=1)
lobf_x, lobf_y = lobf.linspace(2)
plt.plot(lobf_x, lobf_y)
plt.savefig("plots/scatter.svg")
# print equation of line of best fit
# TODO: add correlation and equation of LOBF to plot
print(f"line of best fit for scatter plot: {lobf.convert()}")

correlation = plotting_df["sd"].corr(plotting_df["score"])
print(f"correlation between std. dev. of human responses and sycophancy score = {correlation:.3}")