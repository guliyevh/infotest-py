import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings

def infotest(model, X, y, white=True):
    """
    Python implementation of R's infotest function
    
    Parameters:
    -----------
    model : LinearRegression
        Fitted sklearn LinearRegression model
    X : array-like
        The feature matrix used to train the model
    y : array-like
        The target vector used to train the model
    white : bool, default True
        Whether to perform White's test
        
    Returns:
    --------
    dict : Test results
    """
    
    # Check if the model is from LinearRegression
    if not isinstance(model, LinearRegression):
        raise ValueError("infotest is only possible after linear regression (LinearRegression)")
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y).flatten()
    
    # Get residuals and predictions
    y_pred = model.predict(X)
    res = y - y_pred
    n = len(res)
    s2 = np.sum(res**2) / n
    
    # Handle 1D case
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    
    nrhs = X.shape[1]
    
    if nrhs == 0:
        print("(infotest not allowed without covariates)")
        return None
    
    # Center the data (equivalent to removing intercept effects)
    X_centered = X - np.mean(X, axis=0)
    
    # Create squared terms (x_i * x_j) - centered version
    rhs2 = None
    k = 0
    
    for i in range(nrhs):
        for j in range(i + 1):  # Include diagonal (i,i) terms
            if rhs2 is None:
                rhs2 = (X_centered[:, i] * X_centered[:, j]).reshape(-1, 1)
            else:
                rhs2 = np.column_stack([rhs2, X_centered[:, i] * X_centered[:, j]])
            k += 1
    
    results = {}
    
    # White's original test
    if white:
        y_white = res**2
        
        # Prepare design matrix for White's test
        if rhs2 is not None:
            X_white = np.column_stack([X, rhs2])
        else:
            X_white = X
        
        # Add intercept
        X_white = np.column_stack([np.ones(n), X_white])
        
        # Fit White's auxiliary regression
        white_model = LinearRegression(fit_intercept=False)
        white_model.fit(X_white, y_white)
        y_white_pred = white_model.predict(X_white)
        
        # Calculate R-squared
        ss_res = np.sum((y_white - y_white_pred)**2)
        ss_tot = np.sum((y_white - np.mean(y_white))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        chi2_stat = n * r2
        df = X_white.shape[1] - 1  # minus intercept
        p_val = chi2.sf(chi2_stat, df)
        
        results['white'] = {
            'chi2': chi2_stat,
            'df': df,
            'p': p_val
        }
    
    # Cameron & Trivedi's decomposition
    
    # Heteroskedasticity component
    y_h = res**2 - s2
    y_h_sq = y_h**2
    uss_h = np.sum(y_h_sq)
    
    # Prepare design matrix for heteroskedasticity component
    if rhs2 is not None:
        X_h = np.column_stack([X, rhs2])
    else:
        X_h = X
    X_h = np.column_stack([np.ones(n), X_h])
    
    model_h = LinearRegression(fit_intercept=False)
    model_h.fit(X_h, y_h)
    rss_h = np.sum((y_h - model_h.predict(X_h))**2)
    
    chi2_h = n * (1 - rss_h / uss_h) if uss_h != 0 else 0
    df_h = X_h.shape[1] - 1
    p_h = chi2.sf(chi2_h, df_h)
    
    # Skewness component
    y_s = res**3 - 3 * s2 * res
    y_s_sq = y_s**2
    uss_s = np.sum(y_s_sq)
    
    X_s = np.column_stack([np.ones(n), X])
    model_s = LinearRegression(fit_intercept=False)
    model_s.fit(X_s, y_s)
    rss_s = np.sum((y_s - model_s.predict(X_s))**2)
    
    chi2_s = n * (1 - rss_s / uss_s) if uss_s != 0 else 0
    df_s = X_s.shape[1] - 1
    p_s = chi2.sf(chi2_s, df_s)
    
    # Kurtosis component
    y_k = res**4 - 6 * s2 * res**2 + 3 * s2**2
    y_k_sq = y_k**2
    uss_k = np.sum(y_k_sq)
    
    X_k = np.ones(n).reshape(-1, 1)
    model_k = LinearRegression(fit_intercept=False)
    model_k.fit(X_k, y_k)
    rss_k = np.sum((y_k - model_k.predict(X_k))**2)
    
    chi2_k = n * (1 - rss_k / uss_k) if uss_k != 0 else 0
    df_k = 1
    p_k = chi2.sf(chi2_k, df_k)
    
    # Total
    chi2_t = chi2_s + chi2_k + chi2_h
    df_t = df_s + df_k + df_h
    p_t = chi2.sf(chi2_t, df_t)
    
    # Store results
    results['decomposition'] = {
        'heteroskedasticity': {'chi2': chi2_h, 'df': df_h, 'p': p_h},
        'skewness': {'chi2': chi2_s, 'df': df_s, 'p': p_s},
        'kurtosis': {'chi2': chi2_k, 'df': df_k, 'p': p_k},
        'total': {'chi2': chi2_t, 'df': df_t, 'p': p_t}
    }
    
    # Print results (mimicking R output)
    if white:
        print("\nWhite's (1980) heteroskedasticity test")
        print("H0: Homoskedasticity")
        print("Ha: Unrestricted heteroskedasticity\n")
        print(f"chi2({results['white']['df']}) = {results['white']['chi2']:.4f}")
        print(f"Prob > chi2 = {results['white']['p']:.4f}\n")
    
    print("Cameron & Trivedi's (1990) decomposition of IM-test\n")
    
    # Create and print table
    tab_data = {
        'Source': ['Heteroskedasticity', 'Skewness', 'Kurtosis', 'Joint Test'],
        'chi2': [chi2_h, chi2_s, chi2_k, chi2_t],
        'df': [df_h, df_s, df_k, df_t],
        'p': [p_h, p_s, p_k, p_t]
    }
    
    print("----------------------------------------")
    print(f"{'Source':<20} {'chi2':>10} {'df':>6} {'p':>10}")
    print("----------------------------------------")
    
    for i in range(len(tab_data['Source'])):
        source = tab_data['Source'][i]
        chi2_val = tab_data['chi2'][i]
        df_val = tab_data['df'][i]
        p_val = tab_data['p'][i]
        print(f"{source:<20} {chi2_val:>10.2f} {df_val:>6} {p_val:>10.4f}")
    
    print("----------------------------------------")
    
    return results