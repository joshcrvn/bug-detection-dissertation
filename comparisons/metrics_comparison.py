import matplotlib.pyplot as plt
import numpy as np

# Metrics data from evaluation report
metrics = {
    'Raw Model': {
        'Accuracy': 0.5071,
        'Precision': 0.5597,
        'Recall': 0.1882,
        'F1 Score': 0.2816,
        'AUC-ROC': 0.5433,
        'AUC-PR': 0.5475
    },
    'Cleaned Model': {
        'Accuracy': 0.5581,
        'Precision': 0.5602,
        'Recall': 0.6490,
        'F1 Score': 0.6014,
        'AUC-ROC': 0.5652,
        'AUC-PR': 0.5527
    }
}

# Set up the plot
plt.figure(figsize=(12, 6))

# Set width of bars
barWidth = 0.35

# Set positions of the bars on X axis
r1 = np.arange(len(metrics['Raw Model']))
r2 = [x + barWidth for x in r1]

# Create the bars
bars1 = plt.bar(r1, list(metrics['Raw Model'].values()), width=barWidth, label='Raw Model', color='blue', alpha=0.8)
bars2 = plt.bar(r2, list(metrics['Cleaned Model'].values()), width=barWidth, label='Cleaned Model', color='red', alpha=0.8)

# Add labels and title
plt.xlabel('Metrics', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Model Performance Comparison', fontsize=14, pad=15)

# Add xticks on the middle of the group bars
plt.xticks([r + barWidth/2 for r in range(len(metrics['Raw Model']))], 
           list(metrics['Raw Model'].keys()), 
           rotation=45, ha='right')

# Add a legend
plt.legend()

# Add value labels inside the bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.,
                height/2,  # Position text in middle of bar
                f'{height:.3f}',
                ha='center',
                va='center',
                color='white',
                fontweight='bold',
                fontsize=9)

add_value_labels(bars1)
add_value_labels(bars2)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Add grid for better readability
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Save the plot
plt.savefig('dissertation/metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close() 