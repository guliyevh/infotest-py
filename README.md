# infotest-py

`infotest` provides the Information Matrix (IM) test for linear regression with Cameron & Trivedi's decomposition (heteroskedasticity, skewness, kurtosis) and an optional White (1980) heteroskedasticity test.

## Install
```bash
!pip install "git+https://github.com/guliyevh/infotest-py.git"
```

## Quick example
```python
import numpy as np
from sklearn.linear_model import LinearRegression
from infotest import infotest

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 2))
beta = np.array([1.0, -2.0])
y = 3.0 + X @ beta + rng.normal(scale=1.0, size=200)

m = LinearRegression().fit(X, y)
out = infotest(m, X, y, white=True)  # prints summary and returns dict
```

## CLI
```bash
infotest-demo --white
```

## License
MIT © Hasraddin Guliyev
