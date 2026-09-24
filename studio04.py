import numpy as np

def bootstrap_sample(data, compute_stat, n_bootstrap=1000):
    """
    Generate the bootstrap distribution of a statistic

    Parameters
    ----------
    data : array-like
        original sample (for regression: 2D array with columns [x, y])

    compute_stat : callable
        function that computes a univariate statistic from data
    
    n_bootstrap : int, default 1000
        number of bootstrap replicates to generate

    Returns
    -------
    numpy.ndarray
        Array of bootstrap statistics, length n_bootstrap

    Raises
    ------
    ValueError
        If data is empty, n_bootstrap < 1, or data has wrong shape
    TypeError
        If compute_stat is not callable
    

    Example
    -------
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> data = np.random.normal(size=50)
    >>> stats = bootstrap_sample(data, np.mean, n_bootstrap=500)
    >>> stats.shape
    (500,)

    """
    if not callable(compute_stat):
        raise TypeError("compute_stat must be callable")

    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, (int, np.integer)):
        raise TypeError("n_bootstrap must be an integer")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1")

    data = np.asarray(data)
    if data.ndim not in (1, 2):
        raise ValueError("data must be a 1D array or a 2D array of shape (n, p)")
    n = data.shape[0]
    if data.size == 0 or n == 0:
        raise ValueError("data must not be empty")

    # Resample rows with replacement; for 2D data this keeps (x, y) pairs together
    stats = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = np.random.randint(0, n, size=n)
        stats[b] = compute_stat(data[idx])
    return stats

def bootstrap_ci(bootstrap_stats, alpha=0.05):
    """
    Calculate a CI from bootstrap distribution

    Parameters
    ----------
    bootstrap_stats : numpy.ndarray
        bootstrap statistics from bootstrap_sample(...)

    alpha : float, default 0.05
        significance level

    Returns
    -------
    tuple
        (lower_bound, upper_bound) of the CI

    Raises
    ------
    ValueError
        If alpha not in (0, 1) or if bootstrap_stats is empty
    
    Example
    -------
    TBA
    
    """

def r_squared(data):
    """
    Calculate R^2 from a linear regression

    Parameters
    ----------
    data : array-like, shape (n, 2)
        Data with columns [x, y]

    Returns
    -------
    float
        R-squared value between 0 and 1

    Raises
    ------
    ValueError
        If data doesn't have exactly 2 columns or < 2 rows
    """