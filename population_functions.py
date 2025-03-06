#!/usr/bin/env python

#include <static_connection.h>
import nest
import nest.raster_plot
import numpy as np
import sys
import pylab
import math
import matplotlib.pyplot as pyplot
import pickle, yaml
import random
import scipy
import scipy.fftpack
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks, peak_widths, peak_prominences
from scipy.integrate import simpson
import time
import copy
import set_network_params as netparams

nn=netparams.neural_network()

def update_neuronal_characteristic(update_charac,neuron_population,leakage_value):
	neuron_charac = update_charac
	for neuron in neuron_population:
	    nest.SetStatus(neuron, {neuron_charac: leakage_value})
	new_val = nest.GetStatus(neuron_population, keys=neuron_charac)[0]
	return new_val

def read_spike_data(spike_detector):
	senders = []
	spiketimes = []
	spike_detector = spike_detector
	senders += [spike_detector.get('events', 'senders')]
	spiketimes += [spike_detector.get('events', 'times')]
	return senders,spiketimes

def read_membrane_potential(multimeter,pop_size,neuron_num):
	mm = nest.GetStatus(multimeter,keys="events")[0]
	vm =  mm['V_m']
	t_vm = mm['times']
	vm = vm[neuron_num::pop_size]
	t_vm = t_vm[neuron_num::pop_size]
	return vm,t_vm

def count_indiv_spikes(total_neurons,neuron_id_data,calc_freq):
        total_spikes_per_second = 6 if math.isnan(calc_freq) else int(calc_freq*2) #Spiking 2 times per period
        spike_count_array = [len(neuron_id_data[0][i]) for i in range(total_neurons)]
        sparse_count_max = total_spikes_per_second*(nn.sim_time/1000)	
        sparse_firing_count = [i for i, count in enumerate(spike_count_array) if count>=1 and count<=sparse_count_max]
        silent_neuron_count = [i for i, count in enumerate(spike_count_array) if count==0]
        neuron_to_sample = sparse_firing_count[1] if len(sparse_firing_count) > 1 else 0
        #print('Max for sparse firing for this trial: ',sparse_count_max)
        return spike_count_array,neuron_to_sample,len(sparse_firing_count),len(silent_neuron_count) 

def save_spike_data(num_neurons,population,neuron_num_offset):
	spike_time = []
	all_spikes = []
	for i in range(num_neurons):
	    spike_data = population[0][i]
	    neuron_num = [i+neuron_num_offset]*spike_data.shape[0]
	    for j in range(spike_data.shape[0]):
	        spike_time.append(spike_data[j])    
	    indiv_spikes = list(zip(neuron_num,spike_time))
	    all_spikes.extend(indiv_spikes)  
	    spike_time = []     
	return all_spikes

def single_neuron_spikes(neuron_number,population):
	spike_time = [0]*int(nn.sim_time/nn.time_resolution)
	spike_data = population[0][neuron_number]
	for j in range(spike_data.shape[0]):
	    spike_time_index = int(spike_data[j]*(1/nn.time_resolution))-1
	    spike_time[spike_time_index]=spike_data[j]        
	return spike_time

def single_neuron_spikes_binary(neuron_number,population):
	spike_time = [0]*int(nn.sim_time/nn.time_resolution)
	spike_data = population[0][neuron_number]
	for j in range(spike_data.shape[0]):
	    spike_time_index = int(spike_data[j]*(1/nn.time_resolution))-1
	    spike_time[spike_time_index]=1        
	return spike_time

def rate_code_spikes(neuron_count, output_spiketimes):
	# Initialize the spike bins array as a 2D array
	bins=np.arange(0, nn.sim_time+nn.time_resolution,nn.time_resolution)
	# Loop over each neuron
	for i in range(neuron_count):
	    t_spikes = output_spiketimes[0][i]
	    # Use numpy's histogram function to assign each spike to its corresponding time bin index
	    spikes_per_bin,bin_edges=np.histogram(t_spikes, bins)
	    # Add the spike counts to the `spike_bins_current` array
	    if i == 0:
	        spike_bins_current = spikes_per_bin
	    else:
	        spike_bins_current += spikes_per_bin
	spike_bins_current = sliding_time_window(spike_bins_current,nn.time_window) #Applies a time window to smooth the output        
	smoothed_spike_bins = gaussian_filter(spike_bins_current, nn.convstd_rate) #Applies a filter to smooth the high frequency noise
	if nn.chop_edges_amount > 0.0:
	    smoothed_spike_bins = smoothed_spike_bins[int(nn.chop_edges_amount):int(-nn.chop_edges_amount)]
	return smoothed_spike_bins

def rate_code_spikes_indiv_neurons(neuron_count, output_spiketimes):
    # Precompute bins for histogram
    bins = np.arange(0, nn.sim_time + nn.time_resolution, nn.time_resolution)
    spike_bins_array = []
    
    # Initialize variables for tracking overall min and max
    spike_bin_array_min_neuron = None
    spike_bin_array_max_neuron = None
    overall_min = float('inf')
    overall_max = float('-inf')

    # Loop over each neuron
    for i in range(neuron_count):
        t_spikes = output_spiketimes[0][i]
        
        # Compute spikes per bin
        spikes_per_bin, _ = np.histogram(t_spikes, bins)
        
        # Apply sliding time window and smoothing
        spike_bins_current = sliding_time_window(spikes_per_bin, nn.time_window_indiv_neurons)
        smoothed_spike_bins = gaussian_filter(spike_bins_current, nn.convstd_rate)
        
        # Chop edges if specified
        if nn.chop_edges_amount > 0.0:
            chop = int(nn.chop_edges_amount)
            smoothed_spike_bins = smoothed_spike_bins[chop:-chop]

        # Calculate current max within the first half of the smoothed bins
        mid_point = len(smoothed_spike_bins) // 2
        current_max = np.max(smoothed_spike_bins[:mid_point])
        
        # Update overall min/max and corresponding neuron bins
        if 0 < current_max < overall_min:
            spike_bin_array_min_neuron = smoothed_spike_bins
            overall_min = current_max
        if current_max > overall_max:
            spike_bin_array_max_neuron = smoothed_spike_bins
            overall_max = current_max
        
        # Append the processed bins
        spike_bins_array.append(smoothed_spike_bins)

    return spike_bins_array, spike_bin_array_min_neuron, spike_bin_array_max_neuron

def sliding_time_window(signal, window_size):
	windows = np.lib.stride_tricks.sliding_window_view(signal, window_size)
	return np.sum(windows, axis=1)

def sliding_time_window_matrix(signal, window_size):
	result = []
	for row in signal:
	    windows = np.lib.stride_tricks.sliding_window_view(row, window_size)
	    row_sum = np.sum(windows, axis=1)
	    result.append(row_sum)
	return np.array(result)

def smooth(data, sd):
	data = copy.copy(data)       
	from scipy.signal import gaussian
	from scipy.signal import convolve
	n_bins = data.shape[1]
	w = n_bins - 1 if n_bins % 2 == 0 else n_bins
	window = gaussian(w, std=sd)
	for j in range(data.shape[0]):
	    data[j,:] = convolve(data[j,:], window, mode='same', method='auto') 
	return data

def convolve_spiking_activity(population_size,population):
	time_steps = int(nn.sim_time/nn.time_resolution) 
	binary_spikes = np.vstack([single_neuron_spikes_binary(i, population) for i in range(population_size)])
	binned_spikes = sliding_time_window_matrix(binary_spikes,nn.time_window)
	smoothed_spikes = smooth(binned_spikes, nn.convstd_pca)
	if nn.chop_edges_amount > 0.0:
	    smoothed_spikes = smoothed_spikes[:,int(nn.chop_edges_amount):int(-nn.chop_edges_amount)]
	if nn.remove_mean:
	    smoothed_spikes = (smoothed_spikes.T - np.mean(smoothed_spikes, axis=1)).T
	if nn.high_pass_filtered:
	    #print('High pass filtering the output.')            
	    from scipy.signal import butter, sosfilt, filtfilt, sosfiltfilt
	    # Same used as in Linden et al, 2022 paper
	    b, a = butter(3, .1, 'highpass', fs=1000)		#high pass freq was previously 0.3Hz
	    smoothed_spikes = filtfilt(b, a, smoothed_spikes)
	if nn.downsampling_convolved:
	    from scipy.signal import decimate
	    smoothed_spikes = decimate(smoothed_spikes, int(1/nn.time_resolution), n=2, ftype='iir', zero_phase=True)
	smoothed_spikes = smoothed_spikes[:, :-nn.time_window+1] #truncate array by the width of the time window
	return smoothed_spikes

def inject_current(neuron_population,current):
	for neuron in neuron_population:
	    nest.SetStatus([neuron],{"I_e": current})
	updated_current = nest.GetStatus(neuron_population, keys="I_e")[0]
	return updated_current
	
def normalize_rows(matrix):
    max_values = np.max(matrix, axis=1, keepdims=True)
    normalized_matrix = matrix / max_values
    return normalized_matrix

def calculate_synapse_percentage():
        all_connections = len(nest.GetConnections())
        percentage_of_connections = [x//all_connections for x in num_of_synapses]       
        print('Total connections: ',all_connections)
        print('Name of connections: ',name_of_pops)
        print('Local connections: ',num_of_synapses)
    
def sum_weights_per_source(population):
    synapse_data = nest.GetConnections(population).get(['source', 'weight'])
    weights_per_source = {}
    for connection in synapse_data:
        source_neuron = synapse_data['source']
        weights = synapse_data['weight']
        for s in set(source_neuron):
            if s not in weights_per_source:
                weights_per_source[s] = sum([w for i, w in enumerate(weights) if source_neuron[i] == s])
            else:
                weights_per_source[s] += sum([w for i, w in enumerate(weights) if source_neuron[i] == s])
    return weights_per_source

def count_spikes_per_source(spike_detector):
    sender_counts = {}
    spike_data = spike_detector.get('events', 'senders')
    #print('Sender data: ',spike_data)
    for sender_list in spike_data:
        for sender in sender_list:
            if sender not in sender_counts:
                sender_counts[sender] = 1
            else:
                sender_counts[sender] += 1
    return sender_counts

def calculate_weighted_balance(pop1,spike_detector):
    total_weight = 0 
    weights_by_source = sum_weights_per_source(pop1)
    sender_counts = count_spikes_per_source(spike_detector)
    #print('Count per neuron ID: ',sender_counts)        
    #print('Weights by source: ',weights_by_source)
    for source in weights_by_source:
        #print('Neuron ID: ',source)
        if source in sender_counts:
            weighted_weight = weights_by_source[source] * sender_counts[source]
        else:
            weighted_weight = 0
        total_weight += weighted_weight
    #total_weight = total_weight*2 if total_weight < 0 else total_weight*.2
    total_weight = total_weight*2.9 if total_weight < 0 else total_weight
    return round(total_weight,2) 

def analyze_neuron_peak_heights(spike_bins_array):
    """
    Analyze average peak heights for single and double electrodes for individual neurons.

    Parameters:
    - spike_times_array: np.ndarray
        2D array where each row contains spike times for a neuron.
    - nn_params: object
        An object or dictionary containing necessary neural network parameters
        (e.g., sim_time, time_resolution, time_window_indiv_neurons, convstd_rate, chop_edges_amount).

    Returns:
    - results: list of dict
        Each entry contains information for a neuron: average peak heights, max difference, etc.
    """
    # Initialize results lists
    results = []
    peak_difference_single_to_double = []
    attenuating_neurons_found = []
    negative_max_diff_count = 0

    for neuron_idx, max_bin_array_exc in enumerate(spike_bins_array):
        # Determine midpoint
        midpoint = len(max_bin_array_exc) // 2

        # Compute max firing rates for single and double electrodes
        max_firing_neuron_single_electrode = np.max(max_bin_array_exc[:midpoint + 10000])
        max_firing_neuron_double_electrode = np.max(max_bin_array_exc[midpoint + 10000:])

        # Find peaks for single electrode
        x_avg_max_firing_neuron_single_electrode, properties_single = find_peaks(
            max_bin_array_exc[:midpoint + 10000],
            height=int(max_firing_neuron_single_electrode / 2),
            distance=10000
        )

        # Find peaks for double electrode
        x_avg_max_firing_neuron_double_electrode, properties_double = find_peaks(
            max_bin_array_exc[midpoint + 10000:],
            height=int(max_firing_neuron_double_electrode / 2),
            distance=10000
        )

        # Compute the average of peak heights
        y_avg_max_firing_neuron_single_electrode = np.mean(properties_single["peak_heights"]) if len(properties_single["peak_heights"]) > 0 else 0
        y_avg_max_firing_neuron_double_electrode = np.mean(properties_double["peak_heights"]) if len(properties_double["peak_heights"]) > 0 else 0

        # Maximum difference
        max_difference = max_firing_neuron_double_electrode - max_firing_neuron_single_electrode 
        peak_difference_single_to_double.append(max_difference)
        
        # Check if max difference is negative
        if max_difference < 0:
            negative_max_diff_count += 1
            attenuating_neurons_found.append(spike_bins_array[neuron_idx])

        # Append the results for the current neuron
        results.append({
            "neuron_idx": neuron_idx,
            "avg_peak_height_single_electrode": y_avg_max_firing_neuron_single_electrode,
            "avg_peak_height_double_electrode": y_avg_max_firing_neuron_double_electrode,
            "max_difference": max_difference
        })
        
    # Calculate the percentage of neurons with a negative max difference
    total_neurons = len(spike_bins_array)
    print('Neuron count (total, decrease): ',total_neurons,negative_max_diff_count)
    negative_max_diff_percentage = round((negative_max_diff_count / total_neurons) * 100,2)
    print('Percentage with a decrease in spiking',negative_max_diff_percentage,'%')

    return peak_difference_single_to_double, negative_max_diff_percentage, attenuating_neurons_found

def compare_area_under_curve(pop_type,pop_output):
    pop_output = np.array([np.array(neuron_spikes) for neuron_spikes in pop_output])
    spike_bins_during_e1 = np.concatenate([pop_output[:, 25000:30000],pop_output[:, 50000:55000]], axis=1)
    spike_bins_during_e1_e2 = np.concatenate([pop_output[:, 125000:130000],pop_output[:, 150000:155000]], axis=1)
    
    area1 = simpson(spike_bins_during_e1)
    area2 = simpson(spike_bins_during_e1_e2)
    
    area_differences = area2 - area1
    print(pop_type,'True area differences (min, max)',round(np.min(area_differences),2),round(np.max(area_differences),2))
    # Normalize to [-1, 1]
    max_abs_diff = np.max(np.abs(area_differences))
    if max_abs_diff != 0:
        area_differences = area_differences / max_abs_diff
    percentage_attenuation = np.sum(area_differences < -0.01) / len(area_differences) * 100
    print(pop_type,'Normalized area differences (min, max)',round(np.min(area_differences),2),round(np.max(area_differences),2))
    print(pop_type,'% attenuating',percentage_attenuation)
    
    return area_differences, percentage_attenuation

def compare_distance_of_magnitude_changes(pop_type,distance_electrode_to_midpoint, distances_midpoint, area_diff):
    """
    Bins the distance values from electrode to neuron into three bins (low, middle, high) based on range,
    and assigns corresponding values from the area difference calculations to the same bins using their indices.
    
    Parameters:
    distances_midpoint (list or np.array): Array of distance values.
    area_diff (list or np.array): Array of corresponding area differences.
    
    Returns:
    dict: Dictionary containing binned indices, distance values, and area values.
    """
    if len(distances_midpoint) != len(area_diff):
        raise ValueError("Both input arrays must have the same length.")
    
    distances_midpoint = np.array(distances_midpoint)
    area_diff = np.array(area_diff)
    
    # Define bin thresholds
    proximity_mid_thresh_low = distance_electrode_to_midpoint / 3
    proximity_mid_thresh_high = -1*distance_electrode_to_midpoint / 3 
    proximity_e1_thresh = 2 * distance_electrode_to_midpoint / 3
    
    # Create bins using indices
    low_bin_idx = np.where(distances_midpoint < proximity_e1_thresh)[0]
    mid_bin_idx = np.where((distances_midpoint >= proximity_e1_thresh) & (distances_midpoint < proximity_mid_thresh_low))[0]
    high_bin_idx = np.where((distances_midpoint >= proximity_mid_thresh_low) & (distances_midpoint <= proximity_mid_thresh_high))[0]
    
    areas_low = area_diff[low_bin_idx]
    areas_high = area_diff[high_bin_idx]
    
    positive_low = areas_low[areas_low > 0.01]
    negative_low = areas_low[areas_low < -0.01]
    positive_high = areas_high[areas_high > 0.01]
    negative_high = areas_high[areas_high < -0.01]

    # Compute means, ensuring non-empty arrays to avoid errors
    mean_low_positive = round(np.mean(positive_low), 2) if positive_low.size > 0 else 0
    mean_low_negative = round(np.mean(negative_low), 2) if negative_low.size > 0 else 0
    mean_high_positive = round(np.mean(positive_high), 2) if positive_high.size > 0 else 0
    mean_high_negative = round(np.mean(negative_high), 2) if negative_high.size > 0 else 0
    
    print('Percentage of '+pop_type+' neurons close to E1, midpoint',round(len(low_bin_idx)*100/len(distances_midpoint),2),round(len(high_bin_idx)*100/len(distances_midpoint),2))
    #print('Min and Max '+pop_type+' area diff close to E1',round(np.min(area_diff[low_bin_idx]),2),round(np.max(area_diff[low_bin_idx]),2))
    #print('Min and Max '+pop_type+' area diff close to midpoint',round(np.min(area_diff[high_bin_idx]),2),round(np.max(area_diff[high_bin_idx]),2))
    print('Avg area diff '+pop_type+' close to E1',mean_low_positive,mean_low_negative)
    print('Avg area diff '+pop_type+' close to midpoint',mean_high_positive,mean_high_negative)
    
    return proximity_e1_thresh, proximity_mid_thresh_low, proximity_mid_thresh_high