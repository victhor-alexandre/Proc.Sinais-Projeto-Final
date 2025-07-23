from scipy.io import loadmat

def extract_filters(chosen_wfilter):
    M = loadmat('wfilters.mat')
    h0 = M['h0']
    h1 = M['h1']
    g0 = M['g0']
    g1 = M['g1']
    h0 = h0[0, chosen_wfilter][0]
    h1 = h1[0, chosen_wfilter][0]
    g0 = g0[0, chosen_wfilter][0]
    g1 = g1[0, chosen_wfilter][0]
    return h0, h1, g0, g1
