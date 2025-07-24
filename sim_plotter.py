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

# Setup figure with broken x-axis
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['text.usetex'] = True
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 6), gridspec_kw={'width_ratios': [4, 1],'wspace': 0.04})

# Plot all algorithms on left plot (0–14,000)
for algo in algos:
    for prob in probs:
        ax1.semilogy(tvec[:12000], cost_data[(algo, prob)][:12000],
                     linestyle=linestyles[algo], color=colors[prob], linewidth=4)

# Plot only GD from 25,000 onward on right plot
for prob in probs:
    ax2.semilogy(tvec[18000:], cost_data[('GD', prob)][18000:],
                 linestyle=linestyles['GD'], color=colors[prob], linewidth=4)

# Set limits
ax1.set_xlim(0, 12000)
ax2.set_xlim(18000, 25000)

# Axis labels and ticks
for ax in [ax1, ax2]:
    ax.grid(True, which='both')
    ax.tick_params(axis='both', labelsize=18)
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(FuncFormatter(log_format_func))

ax1.set_xlabel(r'Iterations, $k$', fontsize=18)
# ax2.set_xlabel(r'Iterations, $k$', fontsize=18)
ax1.set_ylabel(r'$\|f(x) - f(x^\star)\|$', fontsize=18)

# Hide spines between axes
ax1.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax1.tick_params(labelright=False)
ax2.yaxis.tick_right()

# Diagonal break line size
d = 0.007  # smaller value = shorter diagonal

# Get bounding boxes of axes in figure coordinates
bbox1 = ax1.get_position()
bbox2 = ax2.get_position()

# Coordinates for top and bottom breaks
y0 = bbox1.y0
y1 = bbox1.y1

# Function to add a diagonal line
def add_diag(x0, x1, y0, y1):
    line = Line2D([x0, x1], [y0, y1], transform=fig.transFigure,
                  color='k', linewidth=1.2, clip_on=False)
    fig.add_artist(line)

# Top break (stays the same)
add_diag(bbox1.x1 - d, bbox1.x1 + d, y1 - d, y1 + d)  # ax1 top right
add_diag(bbox2.x0 - d, bbox2.x0 + d, y1 - d, y1 + d)  # ax2 top left

# Bottom break (corrected!)
add_diag(bbox1.x1 - d, bbox1.x1 + d, y0 - d, y0 + d)  # ax1 bottom right
add_diag(bbox2.x0 - d, bbox2.x0 + d, y0 - d, y0 + d)  # ax2 bottom left

# Custom legends
algo_lines = [Line2D([0], [0], color='k', linestyle=linestyles[algo], linewidth=2, label=algo) for algo in algos]
prob_lines = [Line2D([0], [0], color=colors[prob], linestyle=' ', marker='.', markersize=12, label=f'$p={prob/100}$') for prob in probs]

legend1 = ax2.legend(handles=algo_lines, title=r'\textbf{Algorithm}', loc='upper left', bbox_to_anchor=(-1.0, 1.0), fontsize=14, title_fontsize=14)
ax2.add_artist(legend1)

legend2 = ax2.legend(handles=prob_lines, title=r'\textbf{Probability}', loc='upper left', bbox_to_anchor=(-0.05, 1.0), fontsize=14, title_fontsize=14)

plt.tight_layout()
plt.subplots_adjust(wspace=0.08)  # adjust spacing between axes
plt.savefig("Alg_comparison_convergence.pdf", format='pdf', bbox_inches='tight')
plt.show()

