# map <leader><leader> :wall<cr>:!python %<cr>

from copy import deepcopy
import numpy as np
from scipy.io import loadmat
from signal_sum import signal_sum

def filter_change_z_signal(h):
    h_ = deepcopy(h)
    h_[1 : :2] = -h_[1 : :2]
    return h_

def locations_null_coefficients(v, tol):
    v_null = [np.abs(v[i]) <= tol for i in range(0, len(v))]
    k = np.where(v_null)[0]
    return k

def qmf_filters_validator(h0, h1, g0, g1, tol = 1e-6):
    # Check the aliasing term
    # We need 1/2 * (H0(-z)G0(z) + H1(-z)G1(z)) = 0.
    h0_ = filter_change_z_signal(h0)
    h1_ = filter_change_z_signal(h1)
    h0_g0 = np.convolve(h0_, g0)
    h1_g1 = np.convolve(h1_, g1)
    aliasing_term = 0.5 * signal_sum(h0_g0, h1_g1)
    k = locations_null_coefficients(aliasing_term, tol)
    if len(k) < len(aliasing_term):
        aliasing_term_null = False
    else:
        aliasing_term_null = True
    # Check the LTI term
    # We need 1/2 * (H0(z)G0(z) + H1(z)G1(z)) = Az^(-d).
    h0_g0 = np.convolve(h0, g0)
    h1_g1 = np.convolve(h1, g1)
    lti_term = 0.5 * signal_sum(h0_g0, h1_g1)
    k = locations_null_coefficients(lti_term, tol)
    d = np.nan
    A = np.nan
    if len(k) == len(lti_term) - 1:
        lti_term_valid = True
        n = np.arange(0, len(lti_term))
        d = np.setdiff1d(n, k)[0]
        print("Atraso:")
        print(d)
        A = lti_term[d]
    else:
        lti_term_valid = False
    valid_filters = lti_term_valid and aliasing_term_null
    return valid_filters, lti_term_valid, aliasing_term_null, A, d

if __name__ == '__main__':
    h0 = np.array([1, 1])
    h1 = np.array([1, -1])
    g0 = np.array([1, 1])
    g1 = np.array([-1, 1])
    valid_filters, lti_term_valid, aliasing_term_null, A, d = \
    qmf_filters_validator(h0, h1, g0, g1)
    print(valid_filters)
    print(lti_term_valid)
    print(aliasing_term_null)
    print(A)
    print(d)
    M = loadmat('wfilters.mat')
    print(M.keys())
    chosen_wfilter = 44
    print(M['filters_families'][0,chosen_wfilter])
    h0 = M['h0']
    h1 = M['h1']
    g0 = M['g0']
    g1 = M['g1']
    h0 = h0[0, chosen_wfilter][0]
    h1 = h1[0, chosen_wfilter][0]
    g0 = g0[0, chosen_wfilter][0]
    g1 = g1[0, chosen_wfilter][0]
    valid_filters, lti_term_valid, aliasing_term_null, A, d = \
    qmf_filters_validator(h0, h1, g0, g1, tol = 1e-4)
    print(valid_filters)
    print(lti_term_valid)
    print(aliasing_term_null)
    print(A)
    print(d)
