import numpy as np
from sklearn.linear_model import LinearRegression
from infotest import infotest


def test_infotest_runs_and_returns_dict():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(150, 3))
    y = 1.0 + X @ np.array([0.5, -0.3, 0.8]) + rng.normal(scale=1.0, size=150)

    model = LinearRegression().fit(X, y)
    out = infotest(model, X, y, white=True, verbose=False)

    assert isinstance(out, dict)
    assert "decomposition" in out
    d = out["decomposition"]
    for key in ["heteroskedasticity", "skewness", "kurtosis", "total"]:
        assert key in d
        for stat in ["chi2", "df", "p"]:
            assert stat in d[key]
