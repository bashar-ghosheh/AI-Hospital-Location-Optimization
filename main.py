import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import math
import random

import matplotlib
matplotlib.use('TkAgg') # Forces Matplotlib to open pop-up windows

# ==========================================
# 0. Load Data
# ==========================================
try:
    metrics_df = pd.read_csv('metrics.csv')
    pop_df     = pd.read_csv('population.csv')
    cand_df    = pd.read_csv('candidates.csv')
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please ensure your C program has generated the CSV files in the same directory.")
    exit(1)

# Derived column: strip out the building cost to isolate pure travel distance
# TravelDistance = TotalCost - (λ × |HC|)
metrics_df['TravelDistance'] = metrics_df['TotalCost'] - (metrics_df['Lambda'] * metrics_df['Hospitals'])

# Convert Lambda to an ordered categorical string so x-axis ticks show
# exactly {1, 10, 50, 100} evenly spaced — avoids the linear-scale distortion
# that would make the 1→10 gap look identical to the 50→100 gap.
lambda_order = ['1', '10', '50', '100']
metrics_df['Lambda_label'] = metrics_df['Lambda'].apply(lambda v: str(int(v)))
metrics_df['Lambda_label'] = pd.Categorical(metrics_df['Lambda_label'],
                                             categories=lambda_order, ordered=True)

# Global style
sns.set_theme(style="whitegrid", palette="muted")
SAVE_DPI = 150  # consistent resolution for all saved figures

# ==========================================
# 1. Lambda vs Number of Hospitals
# ==========================================
plt.figure(figsize=(8, 5))
sns.lineplot(data=metrics_df, x='Lambda_label', y='Hospitals',
             hue='Algorithm', marker='o', errorbar=None)
plt.title('Impact of Penalty (λ) on Number of Hospitals Built')
plt.xlabel('λ (Hospital Cost Parameter)')
plt.ylabel('Number of Open Hospitals')
plt.tight_layout()
plt.savefig('1_lambda_vs_hospitals.png', dpi=SAVE_DPI)
plt.show()

# ==========================================
# 2. Lambda vs Travel Distance
# ==========================================
plt.figure(figsize=(8, 5))
sns.lineplot(data=metrics_df, x='Lambda_label', y='TravelDistance',
             hue='Algorithm', marker='o', errorbar=None)
plt.title('Impact of Penalty (λ) on Patient Travel Distance')
plt.xlabel('λ (Hospital Cost Parameter)')
plt.ylabel('Total Weighted Travel Distance')
plt.tight_layout()
plt.savefig('2_lambda_vs_distance.png', dpi=SAVE_DPI)
plt.show()

# ==========================================
# 3. Lambda vs Final Cost
# ==========================================
plt.figure(figsize=(8, 5))
sns.lineplot(data=metrics_df, x='Lambda_label', y='TotalCost',
             hue='Algorithm', marker='o', errorbar=None)
plt.title('Objective Cost Comparison across λ')
plt.xlabel('λ (Hospital Cost Parameter)')
plt.ylabel('Final Objective Cost')
plt.tight_layout()
plt.savefig('3_lambda_vs_cost.png', dpi=SAVE_DPI)
plt.show()

# ==========================================
# 4. Runtime Comparison
# ==========================================
plt.figure(figsize=(7, 5))
sns.barplot(data=metrics_df, x='Algorithm', y='Time', errorbar='sd', capsize=0.1)
plt.title('Average Execution Time (± 1 SD, all λ combined)')
plt.xlabel('Algorithm')
plt.ylabel('Execution Time (seconds)')
plt.tight_layout()
plt.savefig('4_runtime_comparison.png', dpi=SAVE_DPI)
plt.show()

# ==========================================
# 5. Stability / Variance Across All Lambda Values
# ==========================================
# Facet grid: one box plot per lambda so variance across ALL conditions is visible,
# not just lambda = 50.
g = sns.FacetGrid(metrics_df, col='Lambda_label', col_order=lambda_order,
                  height=4, aspect=0.8, sharey=False)
g.map_dataframe(sns.boxplot,  x='Algorithm', y='TotalCost')
g.map_dataframe(sns.stripplot, x='Algorithm', y='TotalCost',
                color='.25', size=5, alpha=0.6)
g.set_axis_labels('Algorithm', 'Final Cost Distribution')
g.set_titles(col_template='λ = {col_name}')
g.figure.suptitle('Algorithm Stability across Multiple Runs (all λ)', y=1.03)
g.tight_layout()
g.savefig('5_stability_boxplot.png', dpi=SAVE_DPI)
plt.show()

# ==========================================
# 6. Geographic Distribution of Selected Hospitals (λ = 50)
# ==========================================
fig, ax = plt.subplots(figsize=(10, 8))

# Population points — size proportional to weight
ax.scatter(pop_df['X'], pop_df['Y'],
           s=pop_df['Weight'] * 10, c='gray', alpha=0.5,
           label='Population (size ∝ weight)')

# Candidate sites not chosen by either algorithm
unselected = cand_df[(cand_df['HC_Open'] == 0) & (cand_df['SA_Open'] == 0)]
ax.scatter(unselected['X'], unselected['Y'],
           s=30, c='black', marker='x', alpha=0.3,
           label='Candidate Site (Closed)')

# Hill Climbing selections
hc_selected = cand_df[cand_df['HC_Open'] == 1]
ax.scatter(hc_selected['X'], hc_selected['Y'],
           s=150, c='blue', marker='^', edgecolors='black',
           label='HC Hospital')

# Simulated Annealing selections
# Offset by +1 so overlapping HC/SA selections remain individually visible
sa_selected = cand_df[cand_df['SA_Open'] == 1]
ax.scatter(sa_selected['X'] + 1, sa_selected['Y'] + 1,
           s=150, c='red', marker='s', edgecolors='black',
           label='SA Hospital')

ax.set_title('Geographic Distribution of Selected Hospitals')
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.legend(loc='upper right', bbox_to_anchor=(1.30, 1))
ax.grid(True, linestyle='--', alpha=0.6)
# Subtitle clarifies which lambda this snapshot was taken at
fig.text(0.5, -0.01,
         'Note: hospital selections shown for λ = 50 (last run)',
         ha='center', fontsize=9, color='gray')
plt.tight_layout()
plt.savefig('6_map_visualization.png', dpi=SAVE_DPI, bbox_inches='tight')
plt.show()

# ==========================================
# 7. Parameter Tuning — Effect of SA Cooling Rate (α)
# ==========================================
# The project requires at least one parameter tuning analysis per algorithm.
# Here we simulate SA with three different cooling rates (α) on a fresh
# problem instance and plot cost vs. iteration to illustrate the trade-off
# between exploration speed and solution quality.

def sa_simulate(alpha, T0=1000.0, T_min=0.0001, seed=42):
    """
    Lightweight Python re-implementation of SA on the same problem instance
    loaded from CSV, used purely for the parameter tuning visualisation.
    Returns a list of (iteration, best_cost_so_far) pairs.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Rebuild problem arrays from the CSV data
    pop   = list(zip(pop_df['X'], pop_df['Y']))
    w     = list(pop_df['Weight'])
    cands = list(zip(cand_df['X'], cand_df['Y']))
    lam   = 50.0   # fixed lambda for fair comparison
    M_    = len(cands)
    N_    = len(pop)

    def cost(state):
        total = 0.0
        open_idx = [j for j in range(M_) if state[j]]
        if not open_idx:
            return 9_999_999.0
        for i in range(N_):
            d = min(math.sqrt((pop[i][0]-cands[j][0])**2 + (pop[i][1]-cands[j][1])**2)
                    for j in open_idx)
            total += w[i] * d
        total += lam * len(open_idx)
        return total

    # Random initial state (~15% open)
    state = [1 if random.random() < 0.15 else 0 for _ in range(M_)]
    cur   = cost(state)
    best  = cur
    T     = T0
    history = []
    it = 0

    while T > T_min:
        idx = random.randrange(M_)
        state[idx] ^= 1
        nb = cost(state)
        de = nb - cur
        if de <= 0 or random.random() < math.exp(-de / T):
            cur = nb
            if cur < best:
                best = cur
        else:
            state[idx] ^= 1
        T *= alpha
        it += 1
        history.append((it, best))

    return history


# ==========================================
# 8. Parameter Tuning — Effect of HC Initialization Rate
# ==========================================
# The project requires at least one parameter tuning analysis per algorithm.
# Here we simulate Hill Climbing with three different initial opening rates
# to see how starting state sparsity affects convergence and final quality.

def hc_simulate(init_prob, max_iter=500, seed=42):
    """
    Lightweight Python re-implementation of Hill Climbing on the same problem instance,
    used purely for the parameter tuning visualization.
    Returns a list of (iteration, best_cost_so_far) pairs.
    """
    random.seed(seed)
    np.random.seed(seed)

    pop   = list(zip(pop_df['X'], pop_df['Y']))
    w     = list(pop_df['Weight'])
    cands = list(zip(cand_df['X'], cand_df['Y']))
    lam   = 50.0   # fixed lambda for fair comparison
    M_    = len(cands)
    N_    = len(pop)

    def cost(state):
        total = 0.0
        open_idx = [j for j in range(M_) if state[j]]
        if not open_idx:
            return 9_999_999.0
        for i in range(N_):
            d = min(math.sqrt((pop[i][0]-cands[j][0])**2 + (pop[i][1]-cands[j][1])**2)
                    for j in open_idx)
            total += w[i] * d
        total += lam * len(open_idx)
        return total

    # Initialise state based on the parameter tuning probability
    state = [1 if random.random() < init_prob else 0 for _ in range(M_)]
    cur   = cost(state)
    best  = cur
    history = [(0, best)]

    for it in range(1, max_iter + 1):
        idx = random.randrange(M_)
        state[idx] ^= 1  # flip bit
        nb = cost(state)

        if nb < cur:
            cur = nb
            if cur < best:
                best = cur
        else:
            state[idx] ^= 1  # undo flip

        history.append((it, best))

    return history

probs       = [0.10, 0.25, 0.50]
hc_colors   = ['#8e44ad', '#d35400', '#16a085']
hc_labels   = [f'Init Open Rate = {int(p*100)}%' for p in probs]

plt.figure(figsize=(9, 5))
for prob, color, label in zip(probs, hc_colors, hc_labels):
    hist = hc_simulate(prob)
    iters, costs = zip(*hist)
    plt.plot(iters, costs, color=color, label=label, linewidth=1.5)

plt.title('Hill Climbing Parameter Tuning: Effect of Initialization Rate on Convergence\n'
          '(λ = 50, single run per rate)')
plt.xlabel('Iteration')
plt.ylabel('Best Cost Found So Far')
plt.legend()
plt.tight_layout()
plt.savefig('8_hc_tuning.png', dpi=SAVE_DPI)
plt.show()

print("Plot 8 (Hill Climbing Tuning) generated and saved successfully!")


alphas      = [0.99, 0.999, 0.9999]
colors      = ['#e74c3c', '#2980b9', '#27ae60']
labels      = [f'α = {a}' for a in alphas]

plt.figure(figsize=(9, 5))
for alpha, color, label in zip(alphas, colors, labels):
    hist = sa_simulate(alpha)
    iters, costs = zip(*hist)
    # Downsample to at most 500 points so the plot is not over-crowded
    step = max(1, len(iters) // 500)
    plt.plot(iters[::step], costs[::step], color=color, label=label, linewidth=1.5)

plt.title('SA Parameter Tuning: Effect of Cooling Rate (α) on Solution Quality\n'
          '(λ = 50, single run per α)')
plt.xlabel('Iteration')
plt.ylabel('Best Cost Found So Far')
plt.legend()
plt.tight_layout()
plt.savefig('7_sa_alpha_tuning.png', dpi=SAVE_DPI)
plt.show()

print("All 7 plots generated and saved successfully!")