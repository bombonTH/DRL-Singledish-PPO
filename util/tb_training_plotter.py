import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# --- Configuration: CHANGE THESE VALUES ---
FILE_PATH = 'Vanilla vs Framestack vs LSTM.csv'  # 1. Replace with the path to your CSV file
X1_COLUMN = 'StepV'
X2_COLUMN = 'StepF'
X3_COLUMN = 'StepL'
Y1_COLUMN = 'ValueV'
Y2_COLUMN = 'ValueF'
Y3_COLUMN = 'ValueL'
TITLE = 'Effective Survey Coverage During Training (LSTM vs. Baselines)'       # 4. Set the title for your plot
X_LABEL = 'Training Steps'         # 5. Set the label for the X-axis
Y_LABEL = "Effective Survey Coverage (deg²/h)"      # 6. Set the label for the Y-axis
# -----------------------------------------

try:
    # 1. Load the CSV file into a pandas DataFrame
    # 'header=0' assumes the first row is the column names
    # 'skipinitialspace=True' is good practice for CSVs
    df = pd.read_csv(FILE_PATH, header=0, skipinitialspace=True)

    # --- Data Preparation (Optional but recommended) ---
    
    # Check if the required columns exist
    if X1_COLUMN not in df.columns or Y1_COLUMN not in df.columns or Y2_COLUMN not in df.columns:
        print(f"Error: One of the required columns ('{X1_COLUMN}' or '{X2_COLUMN} or '{X3_COLUMN}' '{Y1_COLUMN}' or '{Y2_COLUMN} or '{Y3_COLUMN}') was not found in the CSV.")
        print("Available columns are:", list(df.columns))
    else:
        # Convert columns to numeric types, coercing errors to NaN
        # This handles cases where data might be read as a string
        df[X1_COLUMN] = pd.to_numeric(df[X1_COLUMN], errors='coerce')
        df[X2_COLUMN] = pd.to_numeric(df[X2_COLUMN], errors='coerce')
        df[X3_COLUMN] = pd.to_numeric(df[X3_COLUMN], errors='coerce')
        df[Y1_COLUMN] = pd.to_numeric(df[Y1_COLUMN], errors='coerce')
        df[Y2_COLUMN] = pd.to_numeric(df[Y2_COLUMN], errors='coerce')
        df[Y3_COLUMN] = pd.to_numeric(df[Y3_COLUMN], errors='coerce')
        
        # Calculating moving average
        window_size = 30
        df['MA1'] = df[Y1_COLUMN].rolling(window=window_size).mean()
        # df['MA1'] = np.multiply(3.2569, np.power(df[X_COLUMN], -0.165))
        df['MA2'] = df[Y2_COLUMN].rolling(window=window_size).mean()
        df['MA3'] = df[Y3_COLUMN].rolling(window=window_size).mean()
        # df['MA2'] = np.multiply(6.1674, np.power(df[X_COLUMN], -0.192))

        # Drop rows with NaN values in the plotting columns (clean data)
        df.dropna(subset=[X1_COLUMN, X2_COLUMN, X3_COLUMN, Y1_COLUMN, Y2_COLUMN, Y3_COLUMN], inplace=True)

        # 2. Plot the data
        plt.rcParams['figure.dpi'] = 600
        plt.figure(figsize=(10, 6)) # Set the size of the plot

        # Create the plot (e.g., a line plot or a scatter plot)
        plt.plot(df[X1_COLUMN], df[Y1_COLUMN], 
                 # label=Y1_COLUMN, 
                 # marker='o',         # Use small circles for points
                 linestyle='-',      # Connect points with a line
                 markersize=0,
                 linewidth=1,
                 alpha=0.3,
                 color='#ff7f0e',)
                 
        plt.plot(df[X2_COLUMN], df[Y2_COLUMN], 
                 # label=Y2_COLUMN, 
                 # marker='x',         # Use small circles for points
                 linestyle='-',      # Connect points with a line
                 markersize=0,
                 linewidth=1,
                 alpha=0.3,
                 color='#006400',) 
                 
        plt.plot(df[X3_COLUMN], df[Y3_COLUMN], 
                 # label=Y3_COLUMN, 
                 # marker='s',         # Use small circles for points
                 linestyle='-',      # Connect points with a line
                 markersize=0,
                 linewidth=1,
                 alpha=0.3,
                 color='blue',)
                 
        plt.plot(df[X1_COLUMN], df['MA1'], 
                 label='Non-recurrent Agent', 
                 marker='',         # Use small circles for points
                 linestyle='-',      # Connect points with a line
                 markersize=3,
                 linewidth=2,
                 color='#ff7f0e',)
                 
        plt.plot(df[X2_COLUMN], df['MA2'], 
                 label='Frame stacking Agent', 
                 marker='',         # Use small circles for points
                 linestyle='-',      # Connect points with a line
                 markersize=3,
                 linewidth=2,
                 color='#006400',)
                 
        plt.plot(df[X3_COLUMN], df['MA3'], 
                 label='Recurrent Agent (LSTM)', 
                 marker='',         # Use small circles for points
                 linestyle='-',      # Connect points with a line
                 markersize=3,
                 linewidth=2,
                 color='blue',)

        # 3. Customize and display the plot
        plt.title(TITLE, fontsize=16, fontname="Palatino Linotype")
        plt.xlabel(X_LABEL, fontsize=14, fontname="Palatino Linotype")
        plt.ylabel(Y_LABEL, fontsize=14, fontname="Palatino Linotype")
        plt.ylim(0,800)
        
        plt.legend()             # Show the legend for the data line
        plt.grid(True)           # Add a grid for better readability
        plt.tight_layout()       # Adjust layout to prevent labels from overlapping
        
        # 4. Enforce Scientific and Number Formatting Rules
        ax = plt.gca() # Get current axes
        ax.tick_params(
        axis='both',       # Apply to both x and y axis
        which='major',     # Apply to major ticks
        labelsize=12,      # Set the desired font size (e.g., 12pt)
        labelcolor='black', # Optional: set color
        labelfontfamily='Palatino Linotype'
)
        
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
        
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(custom_scientific_formatter))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(custom_scientific_formatter))
        plt.savefig('3.2.png', dpi=600, format='png', bbox_inches='tight')
        # plt.show()
        
except FileNotFoundError:
    print(f"Error: The file '{FILE_PATH}' was not found.")
except pd.errors.EmptyDataError:
    print(f"Error: The file '{FILE_PATH}' is empty.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")