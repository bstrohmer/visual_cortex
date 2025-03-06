#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
from scipy.signal import find_peaks
from scipy.integrate import simpson

plt.rcParams.update({'font.size': 20})
time_resolution=0.1
print_data_array_bd=[]
print_data_array_phase=[]

def analyze_output(input_1,input_2,pop_type,y_line_bd,y_line_phase):
    print('Stability metric calculation')
    try:
        min_dist = 1000
        pop_data1 = input_1
        pop_data2 = input_2   
        up_bd1,down_bd1,burst_duration1,bd_variance1,coeff_bd_variance1=calculate_burst_duration(pop_data1,y_line_bd)
        up_bd2,down_bd2,burst_duration2,bd_variance2,coeff_bd_variance2=calculate_burst_duration(pop_data2,y_line_bd)          
        
        up_bd1_y = [pop_data1[t_thresh] for t_thresh in up_bd1]
        down_bd1_y = [pop_data1[t_thresh] for t_thresh in down_bd1]
        up_bd2_y = [pop_data2[t_thresh] for t_thresh in up_bd2]         
        down_bd2_y = [pop_data2[t_thresh] for t_thresh in down_bd2]
        
        freq_pop1 = calculate_freq(up_bd1)
        freq_pop2 = calculate_freq(up_bd2)
        avg_freq = (freq_pop1 + freq_pop2)/2
        print('Freq (Exc, Inh)')
        print(round(freq_pop1,2),round(freq_pop2,2))
        
       #Calculate phase using peak to peak
        phase_peak,phase_variance_peak,coeff_phase_variance_peak,avg_freq_peak,pop1_peaks,pop2_peaks=calculate_peak_to_peak_phase(pop_data1,pop_data2,y_line_phase,min_dist)
        phase_peak = 360 - phase_peak if phase_peak > 180 else phase_peak  #Ensure phase always between 0-180 degrees
        phase1_y = [pop_data1[t_peak] for t_peak in pop1_peaks]
        phase2_y = [pop_data2[t_peak] for t_peak in pop2_peaks]

        print(pop_type+' Phase, BD (Exc, Inh): ')
        print(round(phase_peak,2),round(burst_duration1,2),round(burst_duration2,2)) 
        #print(pop_type+' Phase CV, BD CV (Flx, Ext): ')
        #print(round(coeff_phase_variance_peak,2),round(coeff_bd_variance1,2),round(coeff_bd_variance2,2))
        bd_comparison = (burst_duration2-burst_duration1)/burst_duration1
        #print('The extensor BD is ',round(bd_comparison,2),' times the duration of the flexor. (Positive = Larger; Negative = Smaller)')
        
        area1 = np.trapz(pop_data1)
        area2 = np.trapz(pop_data2)
        print(pop_type+' Area under the curve (Exc, Inh): ')
        print(round(area1,2),round(area2,2))
 
    except Exception as e:
        print(f"An error occurred: {str(e)}")  

    return round(avg_freq,2),round(phase_peak,2),round(bd_comparison,2)
    
def calculate_burst_duration(array, value):
    upward_count = 0
    downward_count = 0
    upward_indices = []  # Store indices of upward crossings
    downward_indices = []  # Store indices of downward crossings

    crossing = False

    for index, item in enumerate(array):
        if item >= value and not crossing:
            upward_count += 1
            upward_indices.append(index)
            crossing = True
        elif item < value and crossing:
            downward_count += 1
            downward_indices.append(index)
            crossing = False
    #print('Upward indices',upward_indices)
    #print('Downward indices',downward_indices)
    min_length = min(len(downward_indices), len(upward_indices))
    downward_indices = downward_indices[1:min_length]
    upward_indices = upward_indices[1:min_length]
    burst_duration=(np.subtract(downward_indices,upward_indices))*time_resolution
    bd_variance=np.var(burst_duration)
    coeff_bd_variance=(np.std(burst_duration)/np.mean(burst_duration))*100
    burst_duration=np.mean(burst_duration)
    return upward_indices,downward_indices,round(burst_duration,2),round(bd_variance,2),round(coeff_bd_variance,2)

def calculate_peak_to_peak_phase(spike_bins1, spike_bins2, min_peak_height, min_dist):
    pop1_peaks = find_peaks(spike_bins1, height=min_peak_height, distance=min_dist, prominence=0.1)[0]
    pop2_peaks = find_peaks(spike_bins2, height=min_peak_height, distance=min_dist, prominence=0.1)[0]

    alternating_peaks1 = []
    alternating_peaks2 = []

    i, j = 0, 0
    last_pop = None

    while i < len(pop1_peaks) and j < len(pop2_peaks):
        if last_pop is None or last_pop == 2:
            if pop1_peaks[i] < pop2_peaks[j]:
                alternating_peaks1.append(pop1_peaks[i])
                last_pop = 1
                i += 1
            else:
                #alternating_peaks2.append(pop2_peaks[j])
                #last_pop = 2
                j += 1
        elif last_pop == 1 and (pop2_peaks[j] < pop1_peaks[i]):
            alternating_peaks2.append(pop2_peaks[j])
            last_pop = 2
            j += 1
        else:
            #alternating_peaks1.append(pop1_peaks[i])
            #last_pop = 1
            i += 1

    # Truncate to the shortest length
    min_length = min(len(alternating_peaks1), len(alternating_peaks2))
    alternating_peaks1 = alternating_peaks1[:min_length]
    alternating_peaks2 = alternating_peaks2[:min_length]
    #print('Alternating peaks (1,2)',alternating_peaks1,alternating_peaks2)

    # Calculate time differences
    time_diff = np.subtract(alternating_peaks2, alternating_peaks1)
    period1 = np.mean(np.diff(alternating_peaks1)) * time_resolution
    period2 = np.mean(np.diff(alternating_peaks2)) * time_resolution
    avg_period = (period1 + period2) / 2
    avg_freq = 1000 / avg_period

    phase = (avg_period - (time_diff * time_resolution)) / avg_period
    phase_in_deg = phase * 360
    phase_variance = np.var(phase_in_deg)
    coeff_phase_var = (np.std(phase_in_deg) / np.mean(phase_in_deg)) * 100
    avg_phase_in_deg = np.mean(phase * 360)
    
    # Normalize the phase to the range [0, 360)
    avg_phase_in_deg = (avg_phase_in_deg + 360) % 360

    return round(avg_phase_in_deg, 2), round(phase_variance, 2), round(coeff_phase_var, 2), round(avg_freq, 2), alternating_peaks1, alternating_peaks2

def calculate_freq(arr):
    period = np.mean(np.diff(arr)) * time_resolution
    freq = 1000 / period
    
    return freq
