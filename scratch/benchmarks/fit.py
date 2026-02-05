#!/usr/bin/env python3

################################################################################
# Copyright (c) 2023-2026, National Research Foundation (SARAO)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

"""Fit a model to benchmark calibration results."""

import argparse

import numpy as np  # noqa: F401 (flake8 can't see inside the patsy formula)
import pandas as pd
import statsmodels.api as sm
from patsy import build_design_matrices, dmatrices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("--plot", action="store_true", help="Plot the fitted model")
    args = parser.parse_args()

    orig_df = pd.read_csv(args.filename, sep=" ", names=["rate", "successes", "trials", "errors"], index_col="rate")
    # Split each entry back into an original set of trials with 0/1 outcomes
    rates = []
    ind = []
    for rate, row in orig_df.iterrows():
        rates.extend([rate] * row["trials"])
        ind.extend([1] * row["successes"])
        ind.extend([0] * (row["trials"] - row["successes"]))
    df = pd.DataFrame({"ind": ind, "rate": rates})

    y, X = dmatrices("ind ~ np.log(rate)", data=df, return_type="dataframe")  # noqa: N806
    model = sm.Logit(y, X)
    result = model.fit()
    print(result.params)

    if args.plot:
        import matplotlib.pyplot as plt

        (z,) = build_design_matrices([X.design_info], {"rate": orig_df.index})
        p = model.predict(result.params, z)
        (orig_df["successes"] / orig_df["trials"]).plot()
        plt.plot(orig_df.index, p)
        plt.legend(["observed", "predicted"])
        plt.show()


if __name__ == "__main__":
    main()
