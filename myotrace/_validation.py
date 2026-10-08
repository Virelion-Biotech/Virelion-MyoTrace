import numpy as np


def bootstrap_parameters(n_boot, alpha):
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 100:
        raise ValueError("n_boot must be an integer >= 100")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
