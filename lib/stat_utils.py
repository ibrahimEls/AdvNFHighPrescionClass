import os
root_dir = os.getcwd()
import torch
import numpy as np
import ast
import numpy as np
from scipy.interpolate import SmoothBivariateSpline,interp1d
from scipy.optimize import minimize, curve_fit
import json
def string_to_tuple_str(s: str) -> tuple:
    """
    Converts a stringified tuple into an actual tuple of strings.
    
    Supports tuples of length 2 and 3.
    
    Parameters:
    - s (str): The string representation of the tuple, e.g., "('A1', 'B2')" or "('A1', 'B2', 'C3')"
    
    Returns:
    - tuple: A tuple of strings, e.g., ('A1', 'B2') or ('A1', 'B2', 'C3')
    
    Raises:
    - ValueError: If the string does not represent a tuple of length 2 or 3.
    """
    try:
        # Safely evaluate the string to a Python tuple
        parsed = ast.literal_eval(s)
        
        # Check if the result is a tuple with length 2 or 3
        if isinstance(parsed, tuple) and len(parsed) in [2, 3]:
            # Convert each element to string
            return tuple(str(element) for element in parsed)
        else:
            raise ValueError("The string does not represent a tuple of length 2 or 3.")
    
    except (SyntaxError, ValueError) as e:
        print(f"Error parsing string to tuple: {e}")
        return None


def prior_theta(theta, mu_theta=1.0, sigma_theta=0.01):
    """Gaussian prior on theta."""
    return (
        1.0 / (np.sqrt(2 * np.pi) * sigma_theta)
        * np.exp(-0.5 * ((theta - mu_theta) / sigma_theta) ** 2)
    )
    
def compute_mu_nuan_2NP_class(test_data_2j,test_data_1j,dnn_model,bin_splines_S,bin_splines_BG):
    """
    Perform a simultaneous MLE of the global signal fraction `f_s`
    and a single nuisance parameter `theta`, using three categories
    (0-jet, 1-jet, 2-jet).

    Arguments
    ---------
    model : torch.nn.Module
        Your trained model (used for classifier scores).
    test_data_{0,1,2} : torch datasets
        The data for each category (0-jet, 1-jet, 2-jet).
    hist_dict_{0,1,2} : dict
        Each must provide the S and B template histograms keyed by theta values:
            hist_dict_c[theta] = (S_hist, B_hist)

    Returns
    -------
    A tuple (f_s_hat, theta_hat)
    """

    nbins = 200
    bins = np.linspace(0, 1, num=nbins) 
    bin_widths = np.diff(bins)
    print("Performing Classfication")
    with torch.no_grad():
        scores_2j = torch.sigmoid(dnn_model(test_data_2j,2)).cpu().numpy()
        scores_1j = torch.sigmoid(dnn_model(test_data_1j,1)).cpu().numpy()

    total_score = np.concatenate([scores_2j,scores_1j])
    hist_data, _ = np.histogram(total_score, bins=bins)

    N_total = len(total_score)

    # -- 3) Define the negative log-likelihood
    def neg_log_likelihood(params):
        # params = (f_s, nu1, nu2, nu3)
        f_s, nu1, nu2 = params

        # Check bounds (L-BFGS-B also will do this, but let's be explicit).
        if not (0 <= f_s <= 1):
            return np.inf
        # Suppose we allow nu1, nu2, nu3 in [-3, 3], or whichever range is appropriate.
        if abs(nu1) > 1.1 or abs(nu2) > 1.1:
            return np.inf

        # 3a) Morph signal and background histograms at (nu1, nu2, nu3).
        S = morph_histogram_2D_spline([nu1, nu2], bin_splines_S)
        B = morph_histogram_2D_spline([nu1, nu2], bin_splines_BG)


        E = N_total * (f_s * S + (1 - f_s) * B) * bin_widths
        # Avoid zeros in E to prevent log(0)
        E = np.clip(E, a_min=1e-10, a_max=None)
        # Negative log-likelihood
        nll = np.sum(E - hist_data * np.log(E))

        p1 = prior_theta(nu1, mu_theta=1, sigma_theta=.01)
        p2 = prior_theta(nu2, mu_theta=1, sigma_theta=.01)
        prior_val = p1 * p2   # assume independence

        nll_prior = -np.log(prior_val + 1e-40)  # add small epsilon to avoid log(0)

        return nll + nll_prior

    # -- 4) Minimize the NLL
    param_bounds = [
        (0, 1),    # f_s in [0, 1]
        (.9, 1.1),   # nu1 in [-3, 3]
        (.9, 1.1),   # nu2 in [-3, 3]
    ]

    # Initial guess
    initial_params = [0.002, 1, 1]  # f_s ~ 1%, all NPs ~ 0

    print("Performing MLE")
    # We'll use L-BFGS-B
    opt_result = minimize(
        neg_log_likelihood,
        x0=initial_params,
        method='L-BFGS-B',
        bounds=param_bounds
    )
    f_s_hat, nu1_hat, nu2_hat = opt_result.x

    print("===== Fit Results =====")
    print(f"f_s   = {f_s_hat:.6g}")
    print(f"nu1   = {nu1_hat:.6g}")
    print(f"nu2   = {nu2_hat:.6g}")
    print(f"Converged? {opt_result.success}, {opt_result.message}")

    return f_s_hat/.002


def fit_2D_splines_bin_by_bin_from_dict(param_hist_dict, s=0, kx=3, ky=3):
    """
    Fits a SmoothBivariateSpline for each bin (univariate in x,y) using
    the dictionary:
        {
            ("0.9","0.9"): hist_array,   # shape (nbins,)
            ("1.0","1.1"): hist_array,
            ...
        }
    The first element of the tuple is nu1, the second is nu2.
    Each hist_array has the bin contents (length nbins) at that (nu1, nu2).

    We treat each bin b as a function z_b(nu1, nu2), and fit a 2D spline.

    Parameters
    ----------
    param_hist_dict : dict
        Keys: (str, str) => e.g. ("0.9","1.0")
        Values: np.ndarray (nbins,) => histogram bin contents
    s : float, optional
        Smoothing factor for the spline (0 => interpolate exactly). 
        Increase if you have noise and want a smoother fit.
    kx, ky : int, optional
        Spline degrees in x and y (e.g. 1=linear, 3=cubic).

    Returns
    -------
    bin_splines : list of SmoothBivariateSpline
        bin_splines[b] is the spline fitted to bin b's content as a function 
        of (nu1, nu2).
    """

    # 1) Parse dictionary into arrays: param_array of shape (M,2), hist_array of shape (M, nbins)
    param_list = []
    hist_list = []
    
    for key_tuple, hist_data in param_hist_dict.items():
        # key_tuple might be ("0.9","1.0"), so convert to float
        nu_values = [float(x) for x in key_tuple]  # [nu1, nu2]
        param_list.append(nu_values)
        hist_list.append(hist_data)               # shape (nbins,)

    param_array = np.array(param_list)   # shape (M, 2)
    hist_array = np.array(hist_list)     # shape (M, nbins)

    M, nbins = hist_array.shape
    # param_array[:, 0] => all nu1 values
    # param_array[:, 1] => all nu2 values

    xvals = param_array[:, 0]
    yvals = param_array[:, 1]

    # 2) Fit a 2D spline per bin
    bin_splines = []
    for b in range(nbins):
        # z-values for bin b across all M points
        zvals = hist_array[:, b]

        # Fit a bivariate spline, z = f(x,y)
        # *Note*: SmoothBivariateSpline expects 1D arrays x, y, z (scattered points).
        spline = SmoothBivariateSpline(xvals, yvals, zvals, kx=kx, ky=ky, s=s)
        bin_splines.append(spline)

    return bin_splines

def morph_histogram_2D_spline(params, bin_splines):
    """
    Evaluate the 2D spline morphing at (nu1, nu2).
    bin_splines[b] is a SmoothBivariateSpline for bin b.
    
    Parameters
    ----------
    nu1, nu2 : float
        The new parameter values at which to evaluate the histogram.
    bin_splines : list of SmoothBivariateSpline
        The fitted splines from fit_2D_splines_bin_by_bin_from_dict().
    
    Returns
    -------
    morphed_hist : np.ndarray, shape (nbins,)
        The bin contents at (nu1, nu2).
    """
    nu1,nu2 = params
    nbins = len(bin_splines)
    morphed = np.zeros(nbins, dtype=float)

    # Evaluate each bin's spline
    for b in range(nbins):
        # SmoothBivariateSpline.__call__(x, y, grid=False) returns a small 2D array 
        # if x,y are arrays. With scalars, we get a shape (1,1). We'll take [0,0].
        val_2d = bin_splines[b](nu1, nu2, grid=False)
        morphed[b] = val_2d# extract the single scalar

    # Clip negative bin contents if appropriate
    morphed = np.clip(morphed, 0, None)
    return morphed


def load_bias_data(path):
    with open(path, 'r', encoding='utf-8') as file:
        MLE_ratio_arr = json.load(file)
        
    # Example data (replace with your actual data)
    mu_real_values = np.sort(np.array(list(MLE_ratio_arr.keys()))) # Your mu_real keys
    mu_obs_distributions = {mu_real: np.array(MLE_ratio_arr[mu_real],dtype=float) for mu_real in mu_real_values}

    mu_obs_means = []
    mu_obs_stds = []

    for mu_real in mu_real_values:
        mu_obs = mu_obs_distributions[mu_real]
        mu_obs_means.append(np.mean(mu_obs))
        mu_obs_stds.append(np.std(mu_obs))

    mu_obs_means = np.array(mu_obs_means)
    mu_obs_stds = np.array(mu_obs_stds)
    mu_real_values = np.array(mu_real_values,dtype=float)

    def bias_func(mu_real, a, b):
        return a * mu_real + b

    params, _ = curve_fit(bias_func, mu_real_values, mu_obs_means)
    a, b = params

    mu_obs_stds_corrected = mu_obs_stds / abs(a) 
    std_corrected_interp = interp1d(mu_real_values, mu_obs_stds_corrected, kind='linear', fill_value="extrapolate")

    return std_corrected_interp,a,b

def inverse_bias_func(mu_obs_mean,a,b):
    return (mu_obs_mean - b) / a

def compute_posterior(mu_obs, mu_real_range,std_corrected_interp,a,b):
    mu_obs_corrected = inverse_bias_func(mu_obs,a,b)
    likelihood = np.exp(-0.5 * ((mu_obs_corrected - mu_real_range) / std_corrected_interp(mu_real_range))**2)
    posterior = likelihood  
    posterior /= np.trapz(posterior, mu_real_range) 
    return mu_obs_corrected,posterior

def get_confidence_interval(mu_obs,std_corrected_interp,a,b):
    mu_real_range = np.linspace(0, 3, 1000)
    mu_obs_corrected,posterior = compute_posterior(mu_obs, mu_real_range,std_corrected_interp,a,b)
    cdf = np.cumsum(posterior)
    cdf /= cdf[-1]  
    lower_idx = np.searchsorted(cdf, 0.16)
    upper_idx = np.searchsorted(cdf, 0.84)
    return mu_obs_corrected,mu_real_range[lower_idx], mu_real_range[upper_idx]