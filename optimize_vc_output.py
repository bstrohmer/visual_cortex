#!/usr/bin/env python

import nest
import numpy as np
import sys
import pylab
import math
import matplotlib.pyplot as plt
import random
import time
import start_simulation as ss
import pickle, yaml
import pandas as pd
#import elephant
from scipy.signal import find_peaks,correlate
from scipy.fft import fft, fftfreq
import set_network_params as netparams
import population_functions as popfunc
 

import optimize_vc_network as vc
import calculate_metrics as calc

def run_vc_simulation(testing_parameters):
    
    inh_pop_size = testing_parameters[0]
    w_exc = testing_parameters[1]
    w_inh = testing_parameters[2]
    w_selfexc = testing_parameters[3] 
    w_selfinh = testing_parameters[4] 
    sparsity_interconn = testing_parameters[5]
    sparsity_selfconn = testing_parameters[6]
    print('Received parameters: ',testing_parameters)
    
    ss.nest_start()
    nn=netparams.neural_network()

    #Create the visual cortex network
    vc1 = vc.create_vc_network(inh_pop_size,w_exc,w_inh,w_selfexc,w_selfinh,sparsity_interconn,sparsity_selfconn)
    #vc1 = vc.create_vc_network(inh_pop_size,w_inh,w_selfinh,sparsity_interconn,sparsity_selfconn)
    #vc1 = vc.create_vc_network(inh_pop_size,w_exc,w_selfexc,sparsity_interconn,sparsity_selfconn)

    print("Seed#: ",nn.rng_seed)
    print("VC (exc, inh): ",vc1.exc_pop_size,vc1.inh_pop_size)

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
        #Create smoothed population output
        spike_bins_vc_exc_tonic1 = popfunc.rate_code_spikes(vc1.exc_pop_size,spiketimes_exc_tonic1)
        spike_bins_vc_inh_tonic1 = popfunc.rate_code_spikes(vc1.inh_pop_size,spiketimes_inh_tonic1)
        spike_bins_vc1 = spike_bins_vc_exc_tonic1+spike_bins_vc_inh_tonic1
        print('Max network output spike count: ',max(spike_bins_vc1))

        #Create smoothed individual neuron output
        spike_bins_array_exc, min_bin_array_exc, max_bin_array_exc = popfunc.rate_code_spikes_indiv_neurons(vc1.exc_pop_size,spiketimes_exc_tonic1)
        spike_bins_array_inh, min_bin_array_inh, max_bin_array_inh = popfunc.rate_code_spikes_indiv_neurons(vc1.inh_pop_size,spiketimes_inh_tonic1)

        t_stop = time.perf_counter()
        print('Rate coded activity complete, taking ',int(t_stop-t_start),' seconds.')

    x_ticks = np.arange(0, nn.sim_time/nn.time_resolution, 10000)
    x_tick_labels = np.arange(0, nn.sim_time, 1000)
    #Plot rate-coded output
    if nn.rate_coded_plot==1:
        t = np.arange(0,len(spike_bins_vc1),1)
        '''
        fig, ax = plt.subplots(2,sharex='all')
        ax[0].plot(t, spike_bins_vc_exc_tonic1)
        ax[0].plot(t, spike_bins_vc_inh_tonic1)
        ax[1].plot(t, spike_bins_vc1)		
        for i in range(1):
            ax[i].set_xticks([])
            ax[i].set_xlim(0,len(spike_bins_vc1))
        ax[1].set_xlabel('Time (ms)')
        ax[1].set_xticks(x_ticks)
        ax[1].set_xticklabels(x_tick_labels, rotation=45)
        ax[1].set_xlim(0,len(spike_bins_vc1))
        ax[0].legend(['Exc', 'Inh'],loc='upper right',fontsize='x-small') 
        ax[1].legend(['All'],loc='upper right',fontsize='x-small') 
        ax[0].set_title("Population output Exc vs Inh")
        ax[1].set_title("Population output (All)")
        figure = plt.gcf() # get current figure
        figure.set_size_inches(8, 6)
        plt.tight_layout()
        if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + 'rate_coded_output.pdf',bbox_inches="tight")
        '''    
        #Plot of individual neuron smoothed output
        spike_bins_array_exc = np.array(spike_bins_array_exc)
        spike_bins_array_inh = np.array(spike_bins_array_inh)
        mean_activity_exc = np.mean(spike_bins_array_exc, axis=0)
        mean_activity_inh = np.mean(spike_bins_array_inh, axis=0)

        fig, ax = plt.subplots(2,sharex='all')
        t = np.arange(0,len(spike_bins_array_exc[0]),1)
        for neuron_activity in spike_bins_array_exc:
            ax[0].plot(t, neuron_activity, color="lightblue", linewidth=2)
        ax[0].plot(t, min_bin_array_exc, color="darkblue", linewidth=2, label="Min Firing")
        ax[0].plot(t, max_bin_array_exc, color="darkviolet", linewidth=2, label="Max Firing")
        for neuron_activity in spike_bins_array_inh:
            ax[1].plot(t, neuron_activity, color="lightsalmon", linewidth=2)
        ax[1].plot(t, min_bin_array_inh, color="orangered", linewidth=2, label="Min Firing")
        ax[1].plot(t, max_bin_array_inh, color="maroon", linewidth=2, label="Max Firing")
        for i in range(1):
            ax[i].set_xticks([])
            ax[i].set_xlim(0,len(spike_bins_array_exc[0]))
        ax[1].set_xlabel('Time (ms)')
        ax[1].set_xticks(x_ticks)
        ax[1].set_xticklabels(x_tick_labels, rotation=45)
        ax[1].set_xlim(0,len(spike_bins_array_exc[0]))
        ax[0].legend(['Exc'],loc='upper right',fontsize='x-small') 
        ax[1].legend(['Inh'],loc='upper right',fontsize='x-small') 
        ax[0].set_title("Individual neuron output (Exc)")
        ax[1].set_title("Individual neuron output (Inh)")
        figure = plt.gcf() # get current figure
        figure.set_size_inches(8, 6)
        plt.tight_layout()
        if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + 'rate_coded_output_indiv_neurons.pdf',bbox_inches="tight")    

    if nn.args['save_results']:
        # Save rate-coded output
        np.savetxt(nn.pathFigures + '/output_vc_exc.csv',spike_bins_vc_exc_tonic1,delimiter=',')
        np.savetxt(nn.pathFigures + '/output_vc_inh.csv',spike_bins_vc_inh_tonic1,delimiter=',')
    
    midpoint = int(len(max_bin_array_exc)/2)
    max_firing_neuron_single_probe = np.max(max_bin_array_exc[:midpoint])
    max_firing_neuron_double_probe = np.max(max_bin_array_exc[midpoint:])
    score = max_firing_neuron_double_probe-max_firing_neuron_single_probe
    print('Score for this trial',score)
    
    output_for_plotting = [min_bin_array_exc,max_bin_array_exc,min_bin_array_inh,max_bin_array_inh]
    
    return score, output_for_plotting
#Found optimal parameters: [12, -0.030985451646452056, -1.9871176111695257, 0.8298951182430846, 0.3941111567560177]
#testing_parameters = [12, -0.030985451646452056, -1.9871176111695257, 0.8298951182430846, 0.3941111567560177]
#testing_parameters =  = [12,0.6,-0.6,0.6,-0.6,0.5,0.25]
#run_vc_simulation(testing_parameters)
#plt.show()