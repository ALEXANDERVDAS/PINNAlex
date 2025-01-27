import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

results_df = pd.read_csv("experimental_results_batch_vs_error.csv")


# palette = sns.color_palette("Spectral", as_cmap=True)
# sns.set_palette(palette)
sns.lineplot(data=results_df, x='nn', y='error_u', hue='a', marker='o') #, palette="husl"
plt.title("Error_u vs Batch Size for Different Frequencies (a)")
plt.xlabel("Batch Size (nn)")
plt.ylabel("Relative L2 Error in u")
plt.yscale('log')
plt.legend(title="Frequency (a)")
plt.show()

sns.lineplot(data=results_df, x='nn', y='error_r', hue='a', marker='o') #, palette="husl"
plt.title("Error_r vs Batch Size for Different Frequencies (a)")
plt.xlabel("Batch Size (nn)")
plt.ylabel("Relative L2 Error in r")
plt.yscale('log')
plt.legend(title="Frequency (a)")
plt.show()

