import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

stats = [
    {'med': 454.20, 'q1': 203.86, 'q3': 605.60, 'whislo': 0, 'whishi': 928.27, 'label': 'Proportional Controller'},
    {'med': 260.98, 'q1': 79.52, 'q3': 435.43, 'whislo': 0, 'whishi': 770.01, 'label': 'Non-recurrent Agent'},
    {'med': 254.19, 'q1': 68.76, 'q3': 439.45, 'whislo': 0, 'whishi': 751.16, 'label': 'Frame Stacking Agent'},
    {'med': 521.36, 'q1': 312.11, 'q3': 633.30, 'whislo': 0, 'whishi': 875.54, 'label': 'Recurrent Agent'},
]

plt.rcParams['figure.dpi'] = 600
plt.figure(figsize=(10, 6))

ax = plt.gca() # Get current axes
ax.bxp(stats, showfliers=False);

# ax.set_xticks([1, 2])
# ax.set_xticklabels(("Proportional Controller", "Recurrent Agent (LSTM)"))

ax.set_ylabel("Effective Survey Coverage (deg²/h)", fontsize=14, fontname="Palatino Linotype")
plt.title("Comparative Performance: DRL vs. Traditional Control", fontsize=16, fontname="Palatino Linotype")
ax.tick_params(
    axis='both',       # Apply to both x and y axis
    which='major',     # Apply to major ticks
    labelsize=12,      # Set the desired font size (e.g., 12pt)
    labelcolor='black', # Optional: set color
    labelfontfamily='Palatino Linotype'
)

plt.grid(True)           # Add a grid for better readability
plt.tight_layout()       # Adjust layout to prevent labels from overlapping

def custom_scientific_formatter(x, pos):
    # """Custom function to format numbers as 'X.XX × 10^Y'."""
    if x == 0:
        return "0"
    
    # Add a 0 before the decimal point (e.g., 0.1)
    if abs(x) < 1 and abs(x) > 0:
        return f"{x:0.2f}".replace('.', '.')
        
    # Add commas for five or more digits (thousands separator)
    if abs(x) >= 10000:
        return f"{x:,.0f}"
        
    return f"{x:.2f}" # Default formatting

# ax.xaxis.set_major_formatter(ticker.FuncFormatter(custom_scientific_formatter))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(custom_scientific_formatter))
plt.savefig('boxplot.png', dpi=600, format='png', bbox_inches='tight')