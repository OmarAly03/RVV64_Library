import matplotlib.pyplot as plt

# Data from your table
sizes_str = ["1x1", "2x2", "4x4", "8x8", "16x16", "32x32", "64x64", "128x128", "256x256", "320x320"]
scalar_counts = [101, 143, 290, 848, 3080, 12008, 47721, 190569, 761961, 1190505]
vector_counts = [107, 105, 109, 125, 221, 606, 2177, 8561, 34097, 53249]

# Calculate total elements (S = M * N) for the x-axis
total_elements = [int(s.split('x')[0]) * int(s.split('x')[1]) for s in sizes_str]

# Create the plot
plt.figure(figsize=(10, 6))

# Plot scalar and vector counts
plt.plot(total_elements, scalar_counts, marker='o', linestyle='-', label='Scalar Count')
plt.plot(total_elements, vector_counts, marker='s', linestyle='-', label='Vectorized Count')

# Set log-log scale to properly visualize scaling
plt.xscale('log')
plt.yscale('log')

# Add labels and title
plt.xlabel('Total Elements (S = M x N)')
plt.ylabel('Cycle Count')
plt.title('Performance Scaling: Cycle Count vs. Problem Size (Log-Log Scale)')

# Add legend and grid
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.6)

# Save the plot to a file
plt.savefig('relu_scaling_plot.png')

# Display the plot
print("Plot saved as relu_scaling_plot.png")
plt.show()