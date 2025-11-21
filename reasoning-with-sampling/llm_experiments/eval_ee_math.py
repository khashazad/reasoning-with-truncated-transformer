import pandas as pd
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from grader_utils.math_grader import grade_answer

def safe_grade(ans, correct_ans):
    try:
        if pd.isna(ans): return 0
        return int(grade_answer(str(ans), str(correct_ans)))
    except Exception:
        return 0

def eval_ee_math(fname):
    print(fname)
    df = pd.read_csv(fname)
    mcmc_correct = 0
    total = len(df)

    for i in range(total):
        mcmc_correct += safe_grade(df["mcmc_answer"][i], df["correct_answer"][i])

    return mcmc_correct, total

def ee_math_results(fnames):
    mcmc_total = 0
    total = 0

    for fname in fnames:
        mcmc, n = eval_ee_math(fname)
        mcmc_total += mcmc
        total += n

    denom = max(total, 1)
    mcmc_acc = mcmc_total / denom

    print(f"Files evaluated: {len(fnames)}")
    print(f"Total questions: {total}")
    print(f"EE-PS MCMC accuracy:  {mcmc_acc:.3f}")

    return {
        "mcmc_acc": mcmc_acc,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str, help="Path to folder containing result CSVs")
    args = parser.parse_args()

    folder = Path(args.folder)
    fnames = sorted(str(p) for p in folder.glob("*.csv"))
    ee_math_results(fnames)

