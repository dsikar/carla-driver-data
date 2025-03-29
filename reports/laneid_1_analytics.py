import numpy as np

def extract_steering_angles(filename):
    """
    Extracts steering angles from a file where each line follows the pattern:
    [prefix]_steering_[angle].jpg
    
    Args:
        filename (str): Path to the input file (e.g., 'laneid_1.txt').
    
    Returns:
        numpy.ndarray: Array of steering angles in the order they appear in the file.
    """
    steering_angles = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace
            if not line:
                continue  # Skip empty lines
            
            # Split the line into parts and extract the steering value
            parts = line.split('_')
            angle_str = parts[-1].replace('.jpg', '')  # Get the part before .jpg
            steering_angles.append(float(angle_str))
    
    return np.array(steering_angles)  # Convert to NumPy array (or return as list)

# Example usage:
steering_angles = extract_steering_angles('laneid_1.txt')
print(steering_angles[:10])  # Print first 10 angles for verification

import matplotlib.pyplot as plt
import numpy as np

def plot_steering_angles(steering_angles, title="Steering Angles Over Time"):
    """
    Plots steering angles with indices as x-axis and inverts the y-axis.
    
    Args:
        steering_angles (numpy.ndarray or list): Array of steering angles.
        title (str): Optional title for the plot.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(steering_angles, color='blue', linewidth=1)
    
    # Invert y-axis to match typical driving visualization
    # (negative = left turn, positive = right turn)
    plt.gca().invert_yaxis()
    
    # Label axes and add title
    plt.xlabel("Frame Index (Time)")
    plt.ylabel("Steering Angle (Inverted: Left = - , Right = +)")
    plt.title(title)
    
    # Add a horizontal line at 0 for reference (neutral steering)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.grid(True)
    #plt.show()
    plt.savefig('steering_angles_town04.png', dpi=300, bbox_inches='tight')

# Example usage:
steering_angles = extract_steering_angles("laneid_1.txt")  # From previous function
plot_steering_angles(steering_angles)

import numpy as np

def print_steering_stats(steering_angles):
    """
    Prints statistics about steering angles including min, max, mean, std,
    and distribution of left/right/neutral turns.

    Args:
        steering_angles (numpy.ndarray): Array of steering angles.
    """
    if len(steering_angles) == 0:
        print("No steering angles to analyze!")
        return

    stats = {
        'min': np.min(steering_angles),
        'max': np.max(steering_angles),
        'mean': np.mean(steering_angles),
        'median': np.median(steering_angles),
        'std': np.std(steering_angles),
        'abs_mean': np.mean(np.abs(steering_angles)),
        'left_turns': np.sum(steering_angles < -0.001),  # Small threshold to account for floating point
        'right_turns': np.sum(steering_angles > 0.001),
        'neutral': np.sum(np.abs(steering_angles) <= 0.001),
        'total_samples': len(steering_angles)
    }

    # Print basic statistics
    print("Steering Angle Statistics:")
    print(f"- Minimum: {stats['min']:.6f}")
    print(f"- Maximum: {stats['max']:.6f}")
    print(f"- Mean: {stats['mean']:.6f}")
    print(f"- Median: {stats['median']:.6f}")
    print(f"- Standard Deviation: {stats['std']:.6f}")
    print(f"- Mean Absolute Value: {stats['abs_mean']:.6f}\n")

    # Print turn distribution
    print("Turn Direction Distribution:")
    print(f"- Left turns (< -0.001): {stats['left_turns']} ({stats['left_turns']/stats['total_samples']*100:.1f}%)")
    print(f"- Right turns (> 0.001): {stats['right_turns']} ({stats['right_turns']/stats['total_samples']*100:.1f}%)")
    print(f"- Neutral (straight): {stats['neutral']} ({stats['neutral']/stats['total_samples']*100:.1f}%)")
    print(f"- Total samples: {stats['total_samples']}")

# Example usage:
steering_angles = extract_steering_angles("laneid_1.txt")  # From previous function
print_steering_stats(steering_angles)

import numpy as np
import matplotlib.pyplot as plt

def plot_steering_histogram_log(steering_angles):
    """
    Plots a histogram of steering angles with 15 bins from -0.7 to +0.7,
    with logarithmic y-axis scale to better visualize small counts.
    
    Args:
        steering_angles (numpy.ndarray): Array of steering angles
    """
    plt.figure(figsize=(10, 6))
    
    # Create bins - 15 equally spaced bins from -0.7 to 0.7
    bins = np.linspace(-0.7, 0.7, 16)  # 16 edges for 15 bins
    
    # Plot histogram with log y-axis
    n, bins, patches = plt.hist(steering_angles, bins=bins, edgecolor='black', 
                               alpha=0.7, log=True)
    
    # Highlight the bin around zero (typically -0.05 to 0.05)
    zero_bin_index = len(bins) // 2  # Middle bin
    patches[zero_bin_index].set_facecolor('green')
    patches[zero_bin_index].set_alpha(0.8)
    
    # Add labels and title
    plt.xlabel('Steering Angle (Negative = Left, Positive = Right)')
    plt.ylabel('Frequency (log scale)')
    plt.title('Logarithmic Distribution of Steering Angles (15 bins from -0.7 to 0.7)')
    
    # Add vertical line at zero
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1)
    
    # Add text showing percentage of values in zero bin
    zero_bin_width = bins[1] - bins[0]
    zero_bin_range = (bins[zero_bin_index], bins[zero_bin_index + 1])
    zero_count = np.sum((steering_angles >= zero_bin_range[0]) & 
                        (steering_angles < zero_bin_range[1]))
    zero_percent = zero_count / len(steering_angles) * 100
    
    plt.text(0.1, max(n)*0.7,  # Adjusted y-position for log scale
             f'{zero_percent:.1f}% of values\nin {zero_bin_range[0]:.2f} to {zero_bin_range[1]:.2f} range',
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.grid(True, alpha=0.3, which='both')  # Show both major and minor grid lines
    plt.yscale('log')  # Ensure logarithmic scale is set
    #plt.show()
    plt.savefig('steering_histogram_log.png', dpi=300, bbox_inches='tight')
# Example usage:
steering_angles = extract_steering_angles("laneid_1.txt")  # From previous function
plot_steering_histogram_log(steering_angles)
