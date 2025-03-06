#!/usr/bin/env python

import nest
import numpy as np
import sys
import pylab
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import random
import time
import start_simulation as ss
import pickle, yaml
import pandas as pd
from scipy import stats
from scipy.integrate import simpson
import set_network_params as netparams
import population_functions as popfunc
ss.nest_start()
nn=netparams.neural_network() 

import create_vc_network_spatial as vc
import calculate_metrics as calc

plt.rcParams['svg.fonttype'] = 'none'

#Create the visual cortex network  
vc1 = vc.create_vc_network()
	
print("Seed#: ",nn.rng_seed)
print("VC (exc, inh): ",nn.exc_tonic_count,nn.inh_tonic_count)

init_time=50
nest.Simulate(init_time)
num_steps = int(nn.sim_time/nn.time_resolution)
t_start = time.perf_counter()
for i in range(int(num_steps/10)-init_time):
    nest.Simulate(nn.time_resolution*10)
    print("t = " + str(nest.biological_time),end="\r")        
                
t_stop = time.perf_counter()    
print('Simulation completed. It took ',round(t_stop-t_start,2),' seconds.')

spike_count_array = []

#Read spike data 
senders_exc_tonic1,spiketimes_exc_tonic1 = popfunc.read_spike_data(vc1.spike_detector_vc_exc_tonic)
senders_inh_tonic1,spiketimes_inh_tonic1 = popfunc.read_spike_data(vc1.spike_detector_vc_inh_tonic)

#Calculate balance
vc1_exc_tonic_weight = popfunc.calculate_weighted_balance(vc1.vc_exc_tonic,vc1.spike_detector_vc_exc_tonic)
vc1_inh_tonic_weight = popfunc.calculate_weighted_balance(vc1.vc_inh_tonic,vc1.spike_detector_vc_inh_tonic)
weights_per_pop = [vc1_exc_tonic_weight,vc1_inh_tonic_weight]
absolute_weights_per_pop = [vc1_exc_tonic_weight,abs(vc1_inh_tonic_weight)]
vc1_balance_pct = (sum(weights_per_pop)/sum(absolute_weights_per_pop))*100
print('VC balance %: ',round(vc1_balance_pct,2),' >0 skew excitatory; <0 skew inhibitory')

#Create Rate Coded Output
if nn.rate_coded_plot==1:
    t_start = time.perf_counter()
    
    #Create smoothed individual neuron output
    spike_bins_array_exc, min_bin_array_exc, max_bin_array_exc = popfunc.rate_code_spikes_indiv_neurons(nn.exc_tonic_count,spiketimes_exc_tonic1)
    spike_bins_array_inh, min_bin_array_inh, max_bin_array_inh = popfunc.rate_code_spikes_indiv_neurons(nn.inh_tonic_count,spiketimes_inh_tonic1)
    
    exc_area_diff, exc_percentage_attentuating = popfunc.compare_area_under_curve('Exc',spike_bins_array_exc)
    inh_area_diff, inh_percentage_attentuating = popfunc.compare_area_under_curve('Inh',spike_bins_array_inh)

    t_stop = time.perf_counter()
    print('Spike binning complete, taking ',int(t_stop-t_start),' seconds.')

x_ticks = np.arange(0, nn.sim_time/nn.time_resolution, 10000)
x_tick_labels = np.arange(0, nn.sim_time, 1000)
#Plot rate-coded output
if nn.rate_coded_plot==1:
    #Plot of individual neuron smoothed output
    spike_bins_array_exc = np.array(spike_bins_array_exc)
    spike_bins_array_inh = np.array(spike_bins_array_inh)
    mean_activity_exc = np.mean(spike_bins_array_exc, axis=0)
    mean_activity_inh = np.mean(spike_bins_array_inh, axis=0)
    
    mean_during_e1 = np.concatenate([mean_activity_exc[25000:30000],mean_activity_exc[50000:55000]])
    mean_during_e1_e2 = np.concatenate([mean_activity_exc[125000:130000],mean_activity_exc[150000:155000]])
    area1 = simpson(mean_during_e1)
    area2 = simpson(mean_during_e1_e2)
    print('Mean difference in Exc area: ',round(abs(area1-area2),2))
    
    fig, ax = plt.subplots(2,sharex='all')
    t = np.arange(0,len(spike_bins_array_exc[0]),1)
    for neuron_activity in spike_bins_array_exc:
        ax[0].plot(t, neuron_activity, color="lightblue", linewidth=2)
    ax[0].plot(t, mean_activity_exc, color="darkblue", linewidth=2, label="Avg Exc Firing")
    for neuron_activity in spike_bins_array_inh:
        ax[1].plot(t, neuron_activity, color="lightcoral", linewidth=2)
    ax[1].plot(t, mean_activity_inh, color="darkred", linewidth=2, label="Avg Inh Firing")
    for i in range(1):
        ax[i].set_xticks([])
        ax[i].set_xlim(0,len(spike_bins_array_exc[0]))
    ax[1].set_xlabel('Time (ms)')
    ax[1].set_xticks(x_ticks)
    ax[1].set_xticklabels(x_tick_labels, rotation=45)
    ax[1].set_xlim(0,len(spike_bins_array_exc[0]))
    ax[0].legend(loc='upper right',fontsize='x-small') 
    ax[1].legend(loc='upper right',fontsize='x-small') 
    ax[0].set_title("Individual neuron output (Exc)")
    ax[1].set_title("Individual neuron output (Inh)")
    ax[0].text(t[-50],np.mean(max_bin_array_exc),f'{exc_percentage_attentuating:.1f}% attenuating', 
           fontsize=15, color='darkred', ha='center', va='bottom')
    ax[1].text(t[-50],np.mean(max_bin_array_inh),f'{inh_percentage_attentuating:.1f}% attenuating', 
           fontsize=15, color='darkred', ha='center', va='bottom')
    figure = plt.gcf() # get current figure
    figure.set_size_inches(8, 6)
    plt.tight_layout()
    if nn.args['save_results'] and nn.args['save_as_svg']==0: plt.savefig(nn.pathFigures + '/' + 'rate_coded_output_indiv_neurons.pdf',bbox_inches="tight")
    if nn.args['save_results'] and nn.args['save_as_svg']: plt.savefig(nn.pathFigures + '/' + 'rate_coded_output_indiv_neurons.svg',bbox_inches="tight")
        
#Plot neuron positions
plt.figure(figsize=(12, 8))
plt.scatter(vc1.exc_positions[:, 0], vc1.exc_positions[:, 1], color='blue', label='Excitatory Neurons', alpha=0.6)
plt.scatter(vc1.inh_positions[:, 0], vc1.inh_positions[:, 1], color='red', label='Inhibitory Neurons', alpha=0.6)
plt.scatter(vc1.e1_position[0], vc1.e1_position[1], s=200, color='darkorange', label='Electrode 1', alpha=0.6)
plt.scatter(vc1.e2_position[0], vc1.e2_position[1], s=200, color='lightsalmon', label='Electrode 2', alpha=0.6)
plt.xlabel("X Distance (um)")
plt.ylabel("Y Distance (um)")
plt.title("Neuron Positions")
plt.legend(loc='upper left')
xticks = plt.xticks()[0]  # Extract tick positions
yticks = plt.yticks()[0]
plt.xticks(xticks, np.round(xticks * 200).astype(int))
plt.yticks(yticks, np.round(yticks * 200).astype(int))
plt.grid(True)
if nn.args['save_results'] and nn.args['save_as_svg']==0: plt.savefig(nn.pathFigures + '/' + 'neuron_positions.pdf',bbox_inches="tight")
if nn.args['save_results'] and nn.args['save_as_svg']: plt.savefig(nn.pathFigures + '/' + 'neuron_positions.svg',bbox_inches="tight")

#Plot weight connectivity
vmin, vmax = min(vc1.color_vals), max(vc1.color_vals)
plt.figure(figsize=(12, 8))
sc = plt.scatter(vc1.x_vals, vc1.y_vals, c=vc1.color_vals, cmap='BrBG', s=80)
plt.scatter(vc1.x_common, vc1.y_common, facecolors='none', edgecolors='red', s=120, linewidths=2, label="Connected to E1 & E2")
plt.scatter(vc1.e1_position[0], vc1.e1_position[1], s=200, color='darkorange', label='Electrode 1', alpha=0.6)
plt.scatter(vc1.e2_position[0], vc1.e2_position[1], s=200, color='lightsalmon', label='Electrode 2', alpha=0.6)
#plt.scatter(vc1.mid_x, vc1.mid_y, s=200, color='peachpuff', label='Midpoint', alpha=0.6)
cbar = plt.colorbar(sc)
cbar.set_label('Normalized Weight (Electrode to Neuron)')
cbar.set_ticks([vmin, 0, vmax])
cbar.set_ticklabels(['E2 Max', 0, 'E1 Max'])
plt.xlabel("X Distance (um)")
plt.ylabel("Y Distance (um)")
plt.legend(loc='upper left')
plt.title('Neuron Stimulation Heatmap')
xticks = plt.xticks()[0]  # Extract tick positions
yticks = plt.yticks()[0]
plt.xticks(xticks, np.round(xticks * 200).astype(int))
plt.yticks(yticks, np.round(yticks * 200).astype(int))
plt.grid(True)
if nn.args['save_results'] and nn.args['save_as_svg']==0: plt.savefig(nn.pathFigures + '/' + 'weight_vs_distance.pdf',bbox_inches="tight")
if nn.args['save_results'] and nn.args['save_as_svg']: plt.savefig(nn.pathFigures + '/' + 'weight_vs_distance.svg',bbox_inches="tight") 

if nn.rate_coded_plot==1:
    
    plt.figure(figsize=(12, 8))

    # Scatter plot with colormap based on firing rates
    sc = plt.scatter(vc1.exc_positions[:, 0], vc1.exc_positions[:, 1], 
                     c=exc_area_diff, cmap='PRGn', s=50, alpha=0.75, vmin=-1, vmax=1, marker='^', label="Excitatory")

    plt.scatter(vc1.inh_positions[:, 0], vc1.inh_positions[:, 1], 
                c=inh_area_diff, cmap='PRGn', s=50, alpha=0.75, vmin=-1, vmax=1, marker='o', label="Inhibitory")

    # Plot electrodes
    plt.scatter(vc1.e1_position[0], vc1.e1_position[1], s=200, color='darkorange', label='Electrode 1', alpha=0.6)
    plt.scatter(vc1.e2_position[0], vc1.e2_position[1], s=200, color='lightsalmon', label='Electrode 2', alpha=0.6)
    plt.scatter(vc1.mid_x, vc1.mid_y, s=200, color='peachpuff', label='Midpoint', alpha=0.6)
    # Add colorbar to indicate firing rate intensity
    cbar = plt.colorbar(sc)
    cbar.set_label('Normalized Firing (Area under the curve)')
    
    # Create legend handles manually for Excitatory and Inhibitory markers
    exc_marker = mlines.Line2D([], [], color='black', marker='^', linestyle='None', markersize=8, label='Excitatory')
    inh_marker = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=8, label='Inhibitory')
    e1_marker = mlines.Line2D([], [], color='darkorange', marker='o', linestyle='None', markersize=10, label='Electrode 1')
    e2_marker = mlines.Line2D([], [], color='lightsalmon', marker='o', linestyle='None', markersize=10, label='Electrode 2')
    mid_marker = mlines.Line2D([], [], color='peachpuff', marker='o', linestyle='None', markersize=10, label='Midpoint')

    # Labels and formatting
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Neuron Activity Heatmap")
    plt.legend(handles=[exc_marker, inh_marker, e1_marker, e2_marker, mid_marker], loc='upper left')
    #plt.legend(loc='upper left')
    plt.grid(True)
    
    # Save if needed
    if nn.args['save_results'] and nn.args['save_as_svg']==0: 
        plt.savefig(nn.pathFigures + '/' + 'neuron_activity_heatmap.pdf', bbox_inches="tight")
    if nn.args['save_results'] and nn.args['save_as_svg']: plt.savefig(nn.pathFigures + '/' + 'neuron_activity_heatmap.svg',bbox_inches="tight")    
    
    #Create plot of distance to the electrodes versus change in spiking rate
    exc_distances_midpoint = vc1.distances["exc_to_mid"]
    inh_distances_midpoint = vc1.distances["inh_to_mid"]
    
    proximity_e1_thresh, proximity_mid_thresh_low, proximity_mid_thresh_high = popfunc.compare_distance_of_magnitude_changes('Exc',vc1.e1_to_mid, exc_distances_midpoint, exc_area_diff)
    popfunc.compare_distance_of_magnitude_changes('Inh',vc1.e1_to_mid, inh_distances_midpoint, inh_area_diff)
   
    plt.figure(figsize=(12, 8))
    plt.scatter(exc_distances_midpoint, exc_area_diff, color='blue', label='Excitatory Neurons', alpha=0.6)
    plt.scatter(inh_distances_midpoint, inh_area_diff, color='red', label='Inhibitory Neurons', alpha=0.6)
    plt.gca().set_xticks([vc1.e1_to_mid,0,vc1.e2_to_mid])
    plt.gca().set_xticklabels(['E1','Midpoint','E2'], rotation=45)
    plt.xlabel("Distance to electrodes")
    plt.ylabel("Change in area under the curve")
    plt.title("Neuron Firing vs Distance to Electrodes")
    plt.legend(loc='upper left')
    plt.grid(True)
    if nn.args['save_results'] and nn.args['save_as_svg']==0: plt.savefig(nn.pathFigures + '/' + 'neuron_firing_rate_vs_distance.pdf',bbox_inches="tight")
    if nn.args['save_results'] and nn.args['save_as_svg']: plt.savefig(nn.pathFigures + '/' + 'neuron_firing_rate_vs_distance.svg',bbox_inches="tight")     
        
    import statsmodels.api as sm  # For LOWESS
    #Create plot of distance to the electrodes versus change in spiking rate with trend lines
    exc_area_diff = np.array(exc_area_diff)
    inh_area_diff = np.array(inh_area_diff)
    
    exc_mask = exc_area_diff != 0
    inh_mask = inh_area_diff != 0

    exc_distances_midpoint, exc_area_diff = exc_distances_midpoint[exc_mask], exc_area_diff[exc_mask]
    inh_distances_midpoint, inh_area_diff = inh_distances_midpoint[inh_mask], inh_area_diff[inh_mask]

    # Split excitatory neurons (positive and negative values)
    exc_pos_mask = exc_area_diff > 0
    exc_neg_mask = exc_area_diff < 0
    exc_dist_pos, exc_data_pos = exc_distances_midpoint[exc_pos_mask], exc_area_diff[exc_pos_mask]
    exc_dist_neg, exc_data_neg = exc_distances_midpoint[exc_neg_mask], exc_area_diff[exc_neg_mask]

    # Split inhibitory neurons (positive and negative values)
    inh_pos_mask = inh_area_diff > 0
    inh_neg_mask = inh_area_diff < 0
    inh_dist_pos, inh_data_pos = inh_distances_midpoint[inh_pos_mask], inh_area_diff[inh_pos_mask]
    inh_dist_neg, inh_data_neg = inh_distances_midpoint[inh_neg_mask], inh_area_diff[inh_neg_mask]

    # Scatter plot
    plt.figure(figsize=(12, 8))
    plt.scatter(exc_distances_midpoint, exc_area_diff, color='blue', label='Excitatory Neurons', alpha=0.6)
    plt.scatter(inh_distances_midpoint, inh_area_diff, color='red', label='Inhibitory Neurons', alpha=0.6)

    # Fit and plot LOWESS for Excitatory Neurons (Positive)
    fraction_of_data=0.5
    if len(exc_dist_pos) > 0:
        lowess_exc_pos = sm.nonparametric.lowess(exc_data_pos, exc_dist_pos, frac=fraction_of_data)
        plt.plot(lowess_exc_pos[:, 0], lowess_exc_pos[:, 1], color='blue', linewidth=2, linestyle='dashed', label="Excitatory Positive Trend")

    # Fit and plot LOWESS for Excitatory Neurons (Negative)
    if len(exc_dist_neg) > 0:
        lowess_exc_neg = sm.nonparametric.lowess(exc_data_neg, exc_dist_neg, frac=fraction_of_data)
        plt.plot(lowess_exc_neg[:, 0], lowess_exc_neg[:, 1], color='blue', linewidth=2, linestyle='solid', label="Excitatory Negative Trend")

    # Fit and plot LOWESS for Inhibitory Neurons (Positive)
    if len(inh_dist_pos) > 0:
        lowess_inh_pos = sm.nonparametric.lowess(inh_data_pos, inh_dist_pos, frac=fraction_of_data)
        plt.plot(lowess_inh_pos[:, 0], lowess_inh_pos[:, 1], color='red', linewidth=2, linestyle='dashed', label="Inhibitory Positive Trend")

    # Fit and plot LOWESS for Inhibitory Neurons (Negative)
    if len(inh_dist_neg) > 0:
        lowess_inh_neg = sm.nonparametric.lowess(inh_data_neg, inh_dist_neg, frac=fraction_of_data)
        plt.plot(lowess_inh_neg[:, 0], lowess_inh_neg[:, 1], color='red', linewidth=2, linestyle='solid', label="Inhibitory Negative Trend")

    # Format plot
    plt.axvspan(np.min(exc_distances_midpoint),proximity_e1_thresh, color='blue', alpha=0.1)
    plt.axvspan(proximity_mid_thresh_low, proximity_mid_thresh_high, color='blue', alpha=0.1)
    plt.gca().set_xticks([vc1.e1_to_mid, 0, vc1.e2_to_mid])
    plt.gca().set_xticklabels(['E1', 'Midpoint', 'E2'], rotation=45)
    plt.xlabel("Distance to electrodes")
    plt.ylabel("Change in area under the curve")
    plt.title("Neuron Firing vs Distance to Electrodes")
    plt.legend(loc='upper left')
    plt.grid(True)

    # Save if needed
    if nn.args['save_results'] and nn.args['save_as_svg']==0: 
        plt.savefig(nn.pathFigures + '/' + 'neuron_firing_rate_vs_distance_trend_lines.pdf', bbox_inches="tight")
    if nn.args['save_results'] and nn.args['save_as_svg']: plt.savefig(nn.pathFigures + '/' + 'neuron_firing_rate_vs_distance_trend_lines.svg',bbox_inches="tight")   
    
    
#plt.show()