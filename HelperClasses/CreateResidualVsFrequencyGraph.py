import matplotlib.pyplot as plt
import pandas as pd

# List of CSV file names
file_names = ["Other_results_pinn_frequency_SGD.csv", "Other_results_pinn_frequency_Adagrad.csv", "Other_results_pinn_frequency_Nesterov.csv", "Other_results_pinn_frequency_SGDM.csv", "Other_results_pinn_frequency_ADAM.csv"]
names = ["SGD", "Adagrad", "Nesterov", "SGDM", "Adam"]

# Initialize the plot
plt.figure(figsize=(10, 6))

# Iterate through each file and plot the data
for i, file_name in enumerate(file_names, start=0):
    # Read the CSV file
    data = pd.read_csv(file_name)

    # Extract frequency and Average L2 Loss r
    frequency = data['Frequency (a)']
    l2_loss_r = data['Average L2 Loss r']

    # Plot the data
    plt.plot(frequency, l2_loss_r, label=names[i], linewidth=6)

# Customize the plot
plt.xlabel('Frequency (a)', fontsize=32)
plt.ylabel('Residual L2 Loss', fontsize=32)
# plt.title('Frequency vs Residual L2 Loss', fontsize=24)
# plt.yscale('log')
plt.xlim(10, 40)
plt.legend(title="Gradient descent methods", loc="upper left", fontsize=24, title_fontsize=16)
plt.grid(True)
# plt.grid(True, which="both", linestyle="--", linewidth=0.8)

plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

# Show the plot
plt.tight_layout()
plt.show()