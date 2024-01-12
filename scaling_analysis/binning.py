import sys
import numpy as np
import matplotlib.pyplot as plt


def min_max_scale(data):
    min_val = np.min(data)
    max_val = np.max(data)
    print("min", min_val)
    print("max", max_val)
    scaled_data = (data - min_val) / (max_val - min_val)
    return scaled_data

def load_data(file_path):
    try:
        data = np.loadtxt(file_path, dtype=int)
        return min_max_scale(np.abs(data))
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data from '{file_path}': {e}")
        sys.exit(1)

def binning_analysis(data1, data2, num_bins=10):
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.hist(data1, bins=num_bins, alpha=0.5, color='blue', label='File 1', density=True)
    plt.title('Binning Analysis - File 1')
    plt.xlabel('Value')
    plt.ylabel('Normalized Frequency')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.hist(data2, bins=num_bins, alpha=0.5, color='orange', label='File 2', density=True)
    plt.title('Binning Analysis - File 2')
    plt.xlabel('Value')
    plt.ylabel('Normalized Frequency')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python binning_analysis.py <file1> <file2>")
        sys.exit(1)

    file1_path = sys.argv[1]
    file2_path = sys.argv[2]

    data1 = load_data(file1_path)
    data2 = load_data(file2_path)

    binning_analysis(data1, data2)
