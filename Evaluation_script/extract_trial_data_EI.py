import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy.stats import wilcoxon
from scipy.stats import pearsonr, linregress

plt.rcParams['svg.fonttype'] = 'none'

folder_containing_data = sys.argv[1]
inhibitory_silent = 'no' #Choose: yes or no
normalize_area = 'yes'   #Choose: yes or no
#Set parameters for figures
title_fontsize = 24
axis_label_fontsize = 20
axis_line_thickness = 2
annotation_font_size = 14

# Define regex patterns to extract required values
seed_for_trial = re.compile(r"Seed#:\s*(\d+)")
exc_attenuating_pattern = re.compile(r"Exc % attenuating\s*(\d+\.\d+)")
inh_attenuating_pattern = re.compile(r"Inh % attenuating\s*(\d+\.\d+)")
exc_avg_area_e1_pattern = re.compile(r"Avg area diff Exc close to E1\s*([\d\-.]+)\s*([\d\-.]+)")
exc_avg_area_midpoint_pattern = re.compile(r"Avg area diff Exc close to midpoint\s*([\d\-.]+)\s*([\d\-.]+)")
exc_pct_neurons_in_bins_pattern = re.compile(r"Percentage of Exc neurons close to E1, midpoint\s*([\d\-.]+)\s*([\d\-.]+)")
inh_avg_area_e1_pattern = re.compile(r"Avg area diff Inh close to E1\s*([\d\-.]+)\s*([\d\-.]+)")
inh_avg_area_midpoint_pattern = re.compile(r"Avg area diff Inh close to midpoint\s*([\d\-.]+)\s*([\d\-.]+)")
inh_pct_neurons_in_bins_pattern = re.compile(r"Percentage of Inh neurons close to E1, midpoint\s*([\d\-.]+)\s*([\d\-.]+)")
exc_true_area_pattern = re.compile(r"Exc True area differences \(min, max\)\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)")
exc_norm_area_pattern = re.compile(r"Exc Normalized area differences \(min, max\)\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)")

# Read the file containing all trials
with open(folder_containing_data+"/Terminal_Saved_Output_NEW.txt", "r") as file:
    data = file.read()

# Split data into individual trials
trials = data.split("-- N E S T --")  # Adjust if there's a better delimiter
trials = [trial.strip() for trial in trials if trial.strip()]  # Remove empty strings
trials = trials[1:] if len(trials) > 1 else [] # Remove first line (0th trial does not exist)

exc_attenuating_vals = []
inh_attenuating_vals = []
exc_avg_area_e1_vals_pos = []
exc_avg_area_midpoint_vals_pos = []
exc_avg_area_e1_vals_neg = []
exc_avg_area_midpoint_vals_neg = []
exc_pct_neurons_e1 = []
exc_pct_neurons_mid = []
inh_avg_area_e1_vals_pos = []
inh_avg_area_midpoint_vals_pos = []
inh_avg_area_e1_vals_neg = []
inh_avg_area_midpoint_vals_neg = []
inh_pct_neurons_e1 = []
inh_pct_neurons_mid = []
exc_true_area_pos = []
exc_norm_area_pos = []
exc_true_area_neg = []
exc_norm_area_neg = []

# Process each trial and print values
for i, trial in enumerate(trials, start=1):
    seed = seed_for_trial.search(trial)
    exc_attenuating = exc_attenuating_pattern.search(trial)
    inh_attenuating = inh_attenuating_pattern.search(trial)
    exc_avg_area_e1 = exc_avg_area_e1_pattern.search(trial)
    exc_avg_area_midpoint = exc_avg_area_midpoint_pattern.search(trial)
    exc_pct_per_bin = exc_pct_neurons_in_bins_pattern.search(trial)
    inh_avg_area_e1 = inh_avg_area_e1_pattern.search(trial)
    inh_avg_area_midpoint = inh_avg_area_midpoint_pattern.search(trial)
    inh_pct_per_bin = inh_pct_neurons_in_bins_pattern.search(trial)
    exc_true_area = exc_true_area_pattern.search(trial)
    exc_norm_area = exc_norm_area_pattern.search(trial)

    # Extract values or assign None if not found
    seed_value = seed.group(1) if seed else None
    exc_attenuating_value = exc_attenuating.group(1) if exc_attenuating else None
    inh_attenuating_value = inh_attenuating.group(1) if inh_attenuating else None
    exc_avg_area_e1_values = exc_avg_area_e1.groups() if exc_avg_area_e1 else None
    exc_avg_area_midpoint_values = exc_avg_area_midpoint.groups() if exc_avg_area_midpoint else None
    exc_pct_per_bin_values = exc_pct_per_bin.groups() if exc_pct_per_bin else None
    inh_avg_area_e1_values = inh_avg_area_e1.groups() if inh_avg_area_e1 else None
    inh_avg_area_midpoint_values = inh_avg_area_midpoint.groups() if inh_avg_area_midpoint else None
    inh_pct_per_bin_values = inh_pct_per_bin.groups() if inh_pct_per_bin else None
    exc_true_area_values = exc_true_area.groups() if exc_true_area else None
    exc_norm_area_values = exc_norm_area.groups() if exc_norm_area else None

    if exc_attenuating_value is not None and inh_attenuating_value is not None:
        exc_attenuating_vals.append(float(exc_attenuating_value))
        inh_attenuating_vals.append(float(inh_attenuating_value))
    
    if exc_avg_area_e1_values[0] is not None and exc_avg_area_midpoint_values[0] is not None:
        exc_avg_area_e1_vals_pos.append(float(exc_avg_area_e1_values[0]))
        exc_avg_area_midpoint_vals_pos.append(float(exc_avg_area_midpoint_values[0]))
    
    if exc_avg_area_e1_values[1] is not None and exc_avg_area_midpoint_values[1] is not None:
        exc_avg_area_e1_vals_neg.append(float(exc_avg_area_e1_values[1]))
        exc_avg_area_midpoint_vals_neg.append(float(exc_avg_area_midpoint_values[1]))
        
    if exc_pct_per_bin_values[0] is not None:
        exc_pct_neurons_e1.append(float(exc_pct_per_bin_values[0]))
    
    if exc_pct_per_bin_values[1] is not None:
        exc_pct_neurons_mid.append(float(exc_pct_per_bin_values[1]))
    
    if inh_avg_area_e1_values[0] is not None and inh_avg_area_midpoint_values[0] is not None:
        inh_avg_area_e1_vals_pos.append(float(inh_avg_area_e1_values[0]))
        inh_avg_area_midpoint_vals_pos.append(float(inh_avg_area_midpoint_values[0]))
    
    if inh_avg_area_e1_values[1] is not None and inh_avg_area_midpoint_values[1] is not None:
        inh_avg_area_e1_vals_neg.append(float(inh_avg_area_e1_values[1]))
        inh_avg_area_midpoint_vals_neg.append(float(inh_avg_area_midpoint_values[1]))
        
    if inh_pct_per_bin_values[0] is not None:
        inh_pct_neurons_e1.append(float(inh_pct_per_bin_values[0]))
    
    if inh_pct_per_bin_values[1] is not None:
        inh_pct_neurons_mid.append(float(inh_pct_per_bin_values[1])) 
    
    if exc_true_area_values[1] is not None:
        exc_true_area_pos.append(float(exc_true_area_values[1]))
    
    if exc_true_area_values[0] is not None:
        exc_true_area_neg.append(float(exc_true_area_values[0]))
    
    if exc_norm_area_values[1] is not None:
        exc_norm_area_pos.append(float(exc_norm_area_values[1]))
    
    if exc_norm_area_values[0] is not None:
        exc_norm_area_neg.append(float(exc_norm_area_values[0]))

#Covert avg areas from normalized to true values        
if normalize_area == 'no':
    exc_true_area_pos = [float(x) for x in exc_true_area_pos]
    exc_avg_area_e1_vals_pos = [exc_true_area_pos[i] * exc_avg_area_e1_vals_pos[i] for i in range(len(exc_true_area_pos))]
    exc_avg_area_e1_vals_neg = [-1 * exc_true_area_neg[i] * exc_avg_area_e1_vals_neg[i] for i in range(len(exc_true_area_neg))]
    exc_avg_area_midpoint_vals_pos = [exc_true_area_pos[i] * exc_avg_area_midpoint_vals_pos[i] for i in range(len(exc_true_area_pos))]
    exc_avg_area_midpoint_vals_neg = [-1 * exc_true_area_neg[i] * exc_avg_area_midpoint_vals_neg[i] for i in range(len(exc_true_area_neg))]
        
# Create figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(24, 12))

mean_exc_attenuating = np.mean(exc_attenuating_vals)
std_exc_attenuating = np.std(exc_attenuating_vals)

mean_inh_attenuating = np.mean(inh_attenuating_vals)
std_inh_attenuating = np.std(inh_attenuating_vals)

# Boxplot for Exc vs. Inh Attenuating
sns.boxplot(data=[exc_attenuating_vals, inh_attenuating_vals], ax=axes[0], width=0.6)
sns.stripplot(data=[exc_attenuating_vals, inh_attenuating_vals], ax=axes[0], color='black', jitter=True, size=5)
axes[0].set_ylabel("Percentage of Neurons Attenuated", fontsize=axis_label_fontsize)
axes[0].set_yticklabels(axes[0].get_yticks(), fontsize=axis_label_fontsize)
axes[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.1f}'))
axes[0].set_ylim(0,100)
axes[0].set_xticklabels(["Exc Attenuating", "Inh Attenuating"], fontsize=axis_label_fontsize)
axes[0].set_title("Comparison of Exc vs. Inh Attenuating", fontsize=title_fontsize)

# Boxplot for Exc Avg Area E1 vs. Midpoint
sns.boxplot(data=[exc_avg_area_e1_vals_pos, exc_avg_area_midpoint_vals_pos, exc_avg_area_e1_vals_neg, exc_avg_area_midpoint_vals_neg], ax=axes[1], width=0.6)
sns.stripplot(data=[exc_avg_area_e1_vals_pos, exc_avg_area_midpoint_vals_pos, exc_avg_area_e1_vals_neg, exc_avg_area_midpoint_vals_neg], ax=axes[1], color='black', jitter=True, size=5)
axes[1].set_xticklabels(["Enhanced E1", "Enhanced Midpoint", "Attenuated E1", "Attenuated Midpoint"], rotation=45, fontsize=axis_label_fontsize)
axes[1].set_yticklabels(axes[1].get_yticks(), fontsize=axis_label_fontsize)
axes[1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.1f}'))
axes[1].set_ylim(np.min(exc_avg_area_e1_vals_neg)*1.5,np.max(exc_avg_area_midpoint_vals_pos)*1.2)
axes[1].set_ylabel("Normalized Difference in Area", fontsize=axis_label_fontsize)
axes[1].set_title("Magnitude Modulation of Exc Neurons Close to E1 vs Midpoint", fontsize=title_fontsize)

# Calculate means and standard deviations
exc_mean_pos_e1 = np.mean(exc_avg_area_e1_vals_pos)
exc_std_pos_e1 = np.std(exc_avg_area_e1_vals_pos)

exc_mean_pos_mid = np.mean(exc_avg_area_midpoint_vals_pos)
exc_std_pos_mid = np.std(exc_avg_area_midpoint_vals_pos)

exc_mean_neg_e1 = np.mean(exc_avg_area_e1_vals_neg)
exc_std_neg_e1 = np.std(exc_avg_area_e1_vals_neg)

exc_mean_neg_mid = np.mean(exc_avg_area_midpoint_vals_neg)
exc_std_neg_mid = np.std(exc_avg_area_midpoint_vals_neg)

# Statistical test for significance
exc_stat_pos, exc_p_value_pos = wilcoxon(exc_avg_area_e1_vals_pos, exc_avg_area_midpoint_vals_pos)
exc_stat_neg, exc_p_value_neg = wilcoxon(exc_avg_area_e1_vals_neg, exc_avg_area_midpoint_vals_neg)

# Determine significance stars based on p-value
exc_sig_label_pos = "ns"
if exc_p_value_pos < 0.05:
    exc_sig_label_pos = "*"
if exc_p_value_pos < 0.01:
    exc_sig_label_pos = "**"
if exc_p_value_pos < 0.001:
    exc_sig_label_pos = "***"

exc_sig_label_neg = "ns"
if exc_p_value_neg < 0.05:
    exc_sig_label_neg = "*"
if exc_p_value_neg < 0.01:
    exc_sig_label_neg = "**"
if exc_p_value_neg < 0.001:
    exc_sig_label_neg = "***"

# Annotate significance
x1, x2, x3, x4 = 0, 1, 2, 3  # x-coordinates of the two groups
y_pos, h = max(max(exc_attenuating_vals), max(inh_attenuating_vals)) * 1.05, 0.025  # Adjust height for annotation

# Format annotation text
mean_label_exc = f"mean: {mean_exc_attenuating:.2f}, SD: {std_exc_attenuating:.2f}"
mean_label_inh = f"mean: {mean_inh_attenuating:.2f}, SD: {std_inh_attenuating:.2f}"
axes[0].text((x1)*.5, y_pos+h, mean_label_exc, ha='center', va='bottom', fontsize=annotation_font_size, zorder=6)
axes[0].text((x3)*.5, y_pos+h, mean_label_inh, ha='center', va='bottom', fontsize=annotation_font_size, zorder=6)

y_pos, h = max(max(exc_avg_area_e1_vals_pos), max(exc_avg_area_midpoint_vals_pos)) * 1.05, 0.025  # Adjust height for annotation
exc_sig_label_pos_full = f"{exc_sig_label_pos} (p = {exc_p_value_pos:.2e})\n(mean: {exc_mean_pos_e1:.2f}, SD: {exc_std_pos_e1:.2f}) vs. \n(mean: {exc_mean_pos_mid:.2f}, SD: {exc_std_pos_mid:.2f})"
exc_sig_label_neg_full = f"{exc_sig_label_neg} (p = {exc_p_value_neg:.2e})\n(mean: {exc_mean_neg_e1:.2f}, SD: {exc_std_neg_e1:.2f}) vs. \n(mean: {exc_mean_neg_mid:.2f}, SD: {exc_std_neg_mid:.2f})"

# Draw the line for positive significance annotation with zorder for proper layering
axes[1].plot([x1, x1, x2, x2], [y_pos, y_pos+h, y_pos+h, y_pos], lw=1.5, c='black', zorder=5)
axes[1].text((x1+x2)*.5, y_pos+2*h, exc_sig_label_pos_full, ha='center', va='bottom', fontsize=annotation_font_size, zorder=6)

# Draw the line for negative significance annotation with zorder for proper layering
axes[1].plot([x3, x3, x4, x4], [y_pos, y_pos+h, y_pos+h, y_pos], lw=1.5, c='black', zorder=5)
axes[1].text((x3+x4)*.5, y_pos+2*h, exc_sig_label_neg_full, ha='center', va='bottom', fontsize=annotation_font_size, zorder=6)

# Adjust axis line thickness
for axis in ['top','bottom','left','right']:
    axes[0].spines[axis].set_linewidth(axis_line_thickness)
    axes[1].spines[axis].set_linewidth(axis_line_thickness)

plt.tight_layout()

output_file = folder_containing_data + '/vc_output_metrics.pdf'
plt.savefig(output_file, format='pdf')
output_file = folder_containing_data + '/vc_output_metrics.svg'
plt.savefig(output_file, format='svg')   

# Compute correlation coefficient
corr_coef_exc_e1, _ = pearsonr(exc_attenuating_vals, exc_pct_neurons_e1)
corr_coef_exc_mid, _ = pearsonr(exc_attenuating_vals, exc_pct_neurons_mid)
corr_coef_inh_e1, _ = pearsonr(inh_attenuating_vals, inh_pct_neurons_e1)
corr_coef_inh_mid, _ = pearsonr(inh_attenuating_vals, inh_pct_neurons_mid)

#Comparison plot location vs attenuation - excitatory only
plt.figure(figsize=(16, 12))
plt.scatter(exc_attenuating_vals, exc_pct_neurons_e1, color='darkblue', alpha=0.7, edgecolors='k', label='Neurons close to E1')
plt.scatter(exc_attenuating_vals, exc_pct_neurons_mid, color='lightblue', alpha=0.7, edgecolors='k', label='Neurons close to the midpoint')
plt.xlabel('Exc Attenuating Values', fontsize=axis_label_fontsize)
plt.ylabel('Percentage of Exc Neurons per Location', fontsize=axis_label_fontsize)
plt.xticks(fontsize=axis_label_fontsize)
plt.yticks(fontsize=axis_label_fontsize)
plt.xlim(0,100)
plt.ylim(0,100)
plt.title('% of attenuating Exc neurons vs.\n % of Exc neurons close to the midpoint', fontsize=title_fontsize)
plt.grid(True)

# Compute and plot linear regression line
slope_exc_e1, intercept_exc_e1, _, _, _ = linregress(exc_attenuating_vals, exc_pct_neurons_e1)
slope_exc_mid, intercept_exc_mid, _, _, _ = linregress(exc_attenuating_vals, exc_pct_neurons_mid)
x_vals = np.linspace(min(exc_attenuating_vals), max(exc_attenuating_vals), 100)
y_vals_exc_e1 = slope_exc_e1 * x_vals + intercept_exc_e1
y_vals_exc_mid = slope_exc_mid * x_vals + intercept_exc_mid
plt.plot(x_vals, y_vals_exc_e1, color='darkblue', linestyle='dashed', linewidth=2, label='Linear Fit E1')
plt.plot(x_vals, y_vals_exc_mid, color='lightblue', linestyle='dashed', linewidth=2, label='Linear Fit Mid')
legend = plt.legend(fontsize=axis_label_fontsize)
plt.gca().add_artist(legend)
plt.text(0.7, 0.05, f'Pearson r (E1): {corr_coef_exc_e1:.2f}\nPearson r (Mid):{corr_coef_exc_mid:.2f}', fontsize=axis_label_fontsize, color='red', transform=plt.gca().transAxes)

output_file = folder_containing_data + '/exc_location_vs_attenuation.pdf'
plt.savefig(output_file, format='pdf')
output_file = folder_containing_data + '/exc_location_vs_attenuation.svg'
plt.savefig(output_file, format='svg') 

if inhibitory_silent == 'no':
#Comparison plot location vs attenuation - inhibitory only
    plt.figure(figsize=(16, 12))
    plt.scatter(inh_attenuating_vals, inh_pct_neurons_e1, color='darkred', alpha=0.7, edgecolors='k', label='Neurons close to E1')
    plt.scatter(inh_attenuating_vals, inh_pct_neurons_mid, color='pink', alpha=0.7, edgecolors='k', label='Neurons close to the midpoint')
    plt.xlabel('Inh Attenuating Values', fontsize=axis_label_fontsize)
    plt.ylabel('Percentage of Inh Neurons per Location', fontsize=axis_label_fontsize)
    plt.xticks(fontsize=axis_label_fontsize)
    plt.yticks(fontsize=axis_label_fontsize)
    plt.xlim(0,100)
    plt.ylim(0,100)
    plt.title('% of attenuating Inh neurons vs.\n % of Inh neurons close to the midpoint', fontsize=title_fontsize)
    plt.grid(True)

    # Compute and plot linear regression line
    slope_inh_e1, intercept_inh_e1, _, _, _ = linregress(inh_attenuating_vals, inh_pct_neurons_e1)
    slope_inh_mid, intercept_inh_mid, _, _, _ = linregress(inh_attenuating_vals, inh_pct_neurons_mid)
    x_vals = np.linspace(min(inh_attenuating_vals), max(inh_attenuating_vals), 100)
    y_vals_inh_e1 = slope_inh_e1 * x_vals + intercept_inh_e1
    y_vals_inh_mid = slope_inh_mid * x_vals + intercept_inh_mid
    plt.plot(x_vals, y_vals_inh_e1, color='darkred', linestyle='dashed', linewidth=2, label='Linear Fit E1')
    plt.plot(x_vals, y_vals_inh_mid, color='pink', linestyle='dashed', linewidth=2, label='Linear Fit Mid')
    legend = plt.legend(fontsize=axis_label_fontsize)
    plt.gca().add_artist(legend)
    plt.text(0.7, 0.05, f'Pearson r (E1): {corr_coef_inh_e1:.2f}\nPearson r (Mid):{corr_coef_inh_mid:.2f}', fontsize=axis_label_fontsize, color='red', transform=plt.gca().transAxes)

    output_file = folder_containing_data + '/inh_location_vs_attenuation.pdf'
    plt.savefig(output_file, format='pdf')
    output_file = folder_containing_data + '/inh_location_vs_attenuation.svg'
    plt.savefig(output_file, format='svg') 

plt.show()

