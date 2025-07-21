import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter
from matplotlib.lines import Line2D

# Helper function for y-axis ticks
def log_format_func(x, pos):
    exponent = int(np.log10(x))
    return r'$10^{{{}}}$'.format(exponent)

# Algorithm line styles and probabilities/colors
algos = ['GD', 'HB', 'NAG', 'GM']
linestyles = {'GD': '-', 'HB': '--', 'NAG': '-.', 'GM': ':'}
probs = [5, 10, 50]
colors = {5: 'b', 10: 'r', 50: 'g'}

# Load data
cost_data = {}
for prob in probs:
    for algo in algos:
        filename = f"cost_diff_{algo}_prob={prob}.npy"
        cost_data[(algo, prob)] = np.load(filename)

# Step 1: Find the maximum length
max_len = max(len(arr) for arr in cost_data.values())

# Step 2: Pad each array with zeros to match max_len
for key in cost_data:
    arr = cost_data[key]
    if len(arr) < max_len:
        pad_width = max_len - len(arr)
        cost_data[key] = np.concatenate([arr, np.zeros(pad_width)])


# Time vector
dt = 1
tf = max_len-1
tvec = np.arange(0, tf + dt, dt)

# Plot settings
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['text.usetex'] = True
plt.figure(figsize=(9, 6))

# Plot all curves
for algo in algos:
    for prob in probs:
        style = linestyles[algo]
        color = colors[prob]
        plt.semilogy(tvec, cost_data[(algo, prob)][:len(tvec)],
                     linestyle=style, color=color, linewidth=4)

# Custom legend handles
algo_lines = [Line2D([0], [0], color='k', linestyle=linestyles[algo], linewidth=2, label=algo) for algo in algos]
prob_lines = [Line2D([0], [0], color=colors[prob],linestyle=' ',marker='.', markersize=12, label=f'$p={prob/100}$') for prob in probs]

# Axes and labels
plt.xlabel(r'Iterations, $k$', fontsize=18)
plt.ylabel(r'$\|f(x) - f(x^\star)\|$', fontsize=18)
plt.xlim([0, 14000])
plt.grid(True, which='both')
plt.tick_params(axis='both', labelsize=18)  # Set tick label size for both axes

# Log y-axis formatting
ax = plt.gca()
ax.yaxis.set_major_locator(LogLocator(base=10.0))
ax.yaxis.set_major_formatter(FuncFormatter(log_format_func))

# First legend: Algorithm
legend1 = plt.legend(handles=algo_lines,
                     title=r'\textbf{Algorithm}',
                     loc='upper right',
                     bbox_to_anchor=(1.0, 1.0),
                     fontsize=16,
                     title_fontsize=14)

# Add first legend to axes
plt.gca().add_artist(legend1)

# ax = plt.gca()
# ax.text(0.7, 0.91, 'Update/Communication\n Probability',
#         transform=ax.transAxes,
#         fontsize=10,
#         ha='center', va='bottom',zorder=10)

# Second legend: Update/Communicatio Probability, shifted left by x units
legend2 = plt.legend(handles=prob_lines,
                     title=r'\textbf{Probability}',
                     loc='upper right',
                     bbox_to_anchor=(0.8, 1.0),  # Shift this left (smaller x)
                     fontsize=16,
                     title_fontsize=14)

plt.tight_layout()
plt.savefig("Alg_comparison_dual_legend.pdf", format='pdf', bbox_inches='tight')
plt.show()
