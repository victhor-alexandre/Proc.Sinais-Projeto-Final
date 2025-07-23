from matplotlib import pyplot as plt

def font_configuration():
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        # "font.serif": ["Palatino"],
        "font.size": 12,
    })
