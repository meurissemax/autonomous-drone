"""
Settings used to generate plots with LaTeX style.
"""

###########
# Imports #
###########

import matplotlib as mpl
import matplotlib.pyplot as plt


############
# Settings #
############

# Tool bar
mpl.rcParams['toolbar'] = 'None'

# Axes
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.linewidth'] = 0.5

# Font size
plt.rcParams['font.size'] = 14
plt.rcParams['legend.fontsize'] = 'small'

# Figure style
plt.rcParams['figure.autolayout'] = True
plt.rcParams['savefig.transparent'] = True

# LaTeX
if mpl.checkdep_usetex(True):
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['font.serif'] = ['Computer Modern']
    plt.rcParams['text.usetex'] = True
