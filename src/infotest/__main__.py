from __future__ import annotations
import argparse
import numpy as np
from sklearn.linear_model import LinearRegression
from .infotest import infotest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IM test on a toy regression.")
    parser.add_argument("--white", action="store_true", help="Include White's test")
    args = parser.parse_args()

    # Tiny demo with random data (for quick smoke test)
    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 2))
    beta = np.array([1.0, -2.0])
    y = 3.0 + X @ beta + rng.normal(scale=1.0, size=200)

    m = LinearRegression().fit(X, y)
    infotest(m, X, y, white=args.white, verbose=True)


if __name__ == "__main__":
    main()
