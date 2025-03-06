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
import time
import copy
from set_network_params import neural_network
netparams = neural_network()

class create_vc_network():
    def __init__(self):
        self.senders = []
        self.spiketimes = []
        self.saved_spiketimes = []
        self.saved_senders = []
        self.time_window = 50		#50*0.1=5ms time window, based on time resolution of 0.1
        self.count = 0
        
        #Create populations for the visual cortex
        self.tonic_neuronparams_exc = {'C_m':nest.random.normal(mean=netparams.C_m_tonic_mean, std=netparams.C_m_tonic_std), 
                                   'g_L':10.,
                                   'E_L':-60.,
                                   'V_th':nest.random.normal(mean=netparams.V_th_mean_tonic, std=netparams.V_th_std_tonic),
                                   'Delta_T':2.,
                                   'tau_w':30., 
                                   'a':3., 
                                   'b':0., 
                                   'V_reset':-58., 
                                   'I_e':0, 
                                   't_ref':nest.random.normal(mean=netparams.t_ref_mean, std=netparams.t_ref_std),
                                   'V_m':nest.random.normal(mean=netparams.V_m_mean, std=netparams.V_m_std),
                                   "tau_syn_rise_I": netparams.tau_syn_i_rise, 
                                   "tau_syn_decay_I": netparams.tau_syn_i_decay, 
                                   "tau_syn_rise_E": netparams.tau_syn_e_rise, 
                                   "tau_syn_decay_E": netparams.tau_syn_e_decay}
        
        self.tonic_neuronparams_inh = {'C_m':nest.random.normal(mean=netparams.C_m_tonic_mean, std=netparams.C_m_tonic_std), 
                                   'g_L':10.,
                                   'E_L':-80.,
                                   'V_th':nest.random.normal(mean=netparams.V_th_mean_tonic, std=netparams.V_th_std_tonic),
                                   'Delta_T':2.,
                                   'tau_w':30., 
                                   'a':3., 
                                   'b':0., 
                                   'V_reset':-58., 
                                   'I_e':0, 
                                   't_ref':nest.random.normal(mean=netparams.t_ref_mean, std=netparams.t_ref_std),
                                   'V_m':nest.random.normal(mean=netparams.V_m_mean, std=netparams.V_m_std),
                                   "tau_syn_rise_I": netparams.tau_syn_i_rise, 
                                   "tau_syn_decay_I": netparams.tau_syn_i_decay, 
                                   "tau_syn_rise_E": netparams.tau_syn_e_rise, 
                                   "tau_syn_decay_E": netparams.tau_syn_e_decay}
        
        pos_exc = nest.spatial.free(pos=nest.random.uniform(min=-0.6, max=0.6),num_dimensions=2) #Keep at same center
        pos_inh = nest.spatial.free(pos=nest.random.uniform(min=-0.6, max=0.6),num_dimensions=2) 
        self.vc_exc_tonic = nest.Create('aeif_cond_beta_aeif_cond_beta_nestml',netparams.exc_tonic_count,self.tonic_neuronparams_exc,positions=pos_exc) 	
        self.vc_inh_tonic = nest.Create('aeif_cond_beta_aeif_cond_beta_nestml',netparams.inh_tonic_count,self.tonic_neuronparams_inh,positions=pos_inh) 
        
        # Extract positions
        self.exc_positions = np.array([nest.GetPosition(n) for n in self.vc_exc_tonic])
        self.inh_positions = np.array([nest.GetPosition(n) for n in self.vc_inh_tonic])
        
        #Create pulse input
        self.amplitude_times_p1 = []
        self.amplitude_times_p2 = []
        current_time = 0.0
        
        #Set pulse times automatically - comment out manual assignment below to use automation
        while current_time < netparams.sim_time:
            self.amplitude_times_p1.append(current_time)                # Start of pulse
            self.amplitude_times_p1.append(current_time + netparams.pulse_duration)  # End of pulse
            current_time += netparams.pulse_duration + netparams.interval_duration  # Move to the next cycle
        self.amplitude_times_p2 = [t + netparams.delay_bt_electrodes for t in self.amplitude_times_p1]
        # If the last time exceeds sim_time, truncate it
        self.amplitude_times_p1 = [t for t in self.amplitude_times_p1 if t <= netparams.sim_time]
        self.amplitude_times_p2 = [t for t in self.amplitude_times_p2 if t > netparams.sim_time/2 and t <= netparams.sim_time] 
        
        #Set pulse times manually
        self.amplitude_times_p1 = [0.0, 500.0, 2500.0, 3000.0, 5000.0, 5500.0, 12500.0, 13000.0, 15000.0, 15500.0]
        self.amplitude_times_p2 = [7500.0, 8000.0, 10000.0, 10500.0, 12500.0, 13000.0, 15000.0, 15500.0]
        self.amplitude_values_p1 = [netparams.I_e_tonic_mean if i % 2 == 0 else 0 for i in range(len(self.amplitude_times_p1))]
        self.amplitude_values_p2 = [netparams.I_e_tonic_mean if i % 2 == 0 else 0 for i in range(len(self.amplitude_times_p2))]

        print("Amplitude times (p1):", self.amplitude_times_p1)
        print("Amplitude times (p2):", self.amplitude_times_p2)
        print("Amplitude values (p1):", self.amplitude_values_p1)
        print("Amplitude values (p2):", self.amplitude_values_p2)

        self.step_current_e1 = nest.Create("step_current_generator",positions=nest.spatial.grid(shape=[1, 1], extent=[2.0, 2.0], center=[0.5, 0.5]))
        nest.SetStatus(self.step_current_e1, {
            "amplitude_times": self.amplitude_times_p1[1:], #skip time 0
            "amplitude_values": self.amplitude_values_p1[1:]
        })
        
        self.step_current_e2 = nest.Create("step_current_generator",positions=nest.spatial.grid(shape=[1, 1], extent=[2.0, 2.0], center=[-1.1, -0.6])) 
        nest.SetStatus(self.step_current_e2, {
            "amplitude_times": self.amplitude_times_p2[1:], #skip time 0
            "amplitude_values": self.amplitude_values_p2[1:]
        })
        
        self.e1_position = nest.GetPosition(self.step_current_e1)
        self.e2_position = nest.GetPosition(self.step_current_e2)
        
        # Compute midpoint between the two electrodes
        self.mid_x = (self.e1_position[0] + self.e2_position[0]) / 2
        self.mid_y = (self.e1_position[1] + self.e2_position[1]) / 2

        # Compute electrode axis vector and normalize it
        dx = self.e2_position[0] - self.e1_position[0]
        dy = self.e2_position[1] - self.e1_position[1]
        electrode_distance = np.sqrt(dx**2 + dy**2)
        print('Electrode distance (um)',electrode_distance*250)
        # Unit vector along electrode axis
        unit_x = dx / electrode_distance
        unit_y = dy / electrode_distance

        # Function to compute signed distances
        def compute_signed_distances(x_positions, y_positions):
            return (x_positions - self.mid_x) * unit_x + (y_positions - self.mid_y) * unit_y
        
        self.e1_to_mid = compute_signed_distances(self.e1_position[0], self.e1_position[1])
        self.e2_to_mid = compute_signed_distances(self.e2_position[0], self.e2_position[1])
        print('Electrode to midpoint distance (1,2) (um)',self.e1_to_mid*250,self.e2_to_mid*250)
        
        # Compute signed distances for excitatory and inhibitory neurons
        exc_signed_distances = compute_signed_distances(self.exc_positions[:, 0], self.exc_positions[:, 1])
        inh_signed_distances = compute_signed_distances(self.inh_positions[:, 0], self.inh_positions[:, 1])
        #print('Exc neuron distance to mid',exc_signed_distances)
        self.distances = {
            "exc_to_mid": exc_signed_distances,
            "inh_to_mid": inh_signed_distances,
            "electrode_distance": electrode_distance  # For reference
        }

        #Create noise
        self.white_noise_tonic = nest.Create("noise_generator",netparams.noise_params_tonic)     
        
        #Create spike detectors (for recording spikes) 
        self.spike_detector_vc_exc_tonic = nest.Create("spike_recorder",netparams.exc_tonic_count)
        self.spike_detector_vc_inh_tonic = nest.Create("spike_recorder",netparams.inh_tonic_count)
                
        #Create multimeters (for recording membrane potential)
        self.mm_vc_exc_tonic = nest.Create("multimeter",netparams.mm_params)
        self.mm_vc_inh_tonic = nest.Create("multimeter",netparams.mm_params)
        
        #Connect current generator to neurons
        electrode_connections = nest.Connect(self.step_current_e1,self.vc_exc_tonic,netparams.conn_dict_electrodes,netparams.electrode_syn_params)
        nest.Connect(self.step_current_e2,self.vc_exc_tonic,netparams.conn_dict_electrodes,netparams.electrode_syn_params)
        if netparams.silence_inhibitory_neurons == 0:
            nest.Connect(self.step_current_e1,self.vc_inh_tonic,netparams.conn_dict_electrodes,netparams.electrode_syn_params)
            nest.Connect(self.step_current_e2,self.vc_inh_tonic,netparams.conn_dict_electrodes,netparams.electrode_syn_params)
        
        ids = nest.GetConnections(electrode_connections)
        
        # Get all connections with source and target info
        electrode_connections = nest.GetStatus(ids, ['source', 'target', 'weight'])

        # Extract neuron IDs
        exc_neurons = nest.GetStatus(self.vc_exc_tonic, 'global_id')
        inh_neurons = nest.GetStatus(self.vc_inh_tonic, 'global_id')
        
        # Create a mapping of neuron ID → (x, y) position
        neuron_positions = {}

        for neuron, pos in zip(exc_neurons, self.exc_positions):
            neuron_positions[neuron] = (pos[0], pos[1])  # Extract only x, y

        for neuron, pos in zip(inh_neurons, self.inh_positions):
            neuron_positions[neuron] = (pos[0], pos[1])  # Extract only x, y
        
        weights_e1 = {}  
        weights_e2 = {}  

        for source, target, weight in electrode_connections:
            if source == 101:  # Electrode ID (E1)
                weights_e1[target] = weight
            elif source == 102:  # Electrode ID (E2)
                weights_e2[target] = weight

        # Step 4: Normalize weights between -1 and 1
        all_targets = set(weights_e1.keys()).union(weights_e2.keys())
        # Find neurons that receive input from both E1 and E2
        common_targets = set(weights_e1.keys()) & set(weights_e2.keys())
        
        # Get min/max weights for normalization
        max_w1 = max(weights_e1.values(), default=1)
        max_w2 = max(weights_e2.values(), default=1)

        normalized_weights = {}
        for target in all_targets:
            w1 = weights_e1.get(target, 0) / max_w1  # Normalize E1 weights
            w2 = weights_e2.get(target, 0) / max_w2  # Normalize E2 weights
            normalized_weights[target] = w1 - w2  # E1 on one side, E2 on the other

        # Step 5: Create heatmap
        self.x_vals = []
        self.y_vals = []
        self.color_vals = []
        
        for target in all_targets:
            if target in neuron_positions:  # Ensure the target neuron has a position
                x, y = neuron_positions[target]
                self.x_vals.append(x)
                self.y_vals.append(y)
                self.color_vals.append(normalized_weights[target])
        
        self.x_common = [neuron_positions[n][0] for n in common_targets if n in neuron_positions]
        self.y_common = [neuron_positions[n][1] for n in common_targets if n in neuron_positions]
        
        #Connect white noise to neurons
        if netparams.noise_removed == 0:
            nest.Connect(self.white_noise_tonic,self.vc_exc_tonic,"all_to_all")
            if netparams.silence_inhibitory_neurons == 0:
                nest.Connect(self.white_noise_tonic,self.vc_inh_tonic,"all_to_all") 
	
        #Connect neurons within vc
        self.inh_syn_params = {"synapse_model":"static_synapse",
            "weight" : nest.random.normal(mean=netparams.w_inh_mean,std=netparams.w_inh_std), #nS            
            "delay" : netparams.synaptic_delay}	#ms
        self.exc_syn_params = {"synapse_model":"static_synapse",
            "weight" : nest.random.normal(mean=netparams.w_exc_mean,std=netparams.w_exc_std), #nS
            "delay" : netparams.synaptic_delay}	#ms
        self.self_inh_syn_params = {"synapse_model":"static_synapse",
            "weight" : nest.random.normal(mean=netparams.w_inh_mean,std=netparams.w_inh_std), #nS            
            "delay" : netparams.synaptic_delay}	#ms
        self.self_exc_syn_params = {"synapse_model":"static_synapse",
            "weight" : nest.random.normal(mean=netparams.w_exc_mean,std=netparams.w_exc_std), #nS
            "delay" : netparams.synaptic_delay}	#ms
        self.coupling_exc_tonic_exc_tonic = nest.Connect(self.vc_exc_tonic,self.vc_exc_tonic,netparams.conn_dict_self_neurons,self.self_exc_syn_params)
        if netparams.silence_inhibitory_neurons == 0:
            self.coupling_exc_tonic_inh_tonic = nest.Connect(self.vc_exc_tonic,self.vc_inh_tonic,netparams.conn_dict_exc_neurons,self.exc_syn_params)
            self.coupling_inh_tonic_inh_tonic = nest.Connect(self.vc_inh_tonic,self.vc_inh_tonic,netparams.conn_dict_self_neurons,self.inh_syn_params)    
            self.coupling_inh_tonic_exc_tonic = nest.Connect(self.vc_inh_tonic,self.vc_exc_tonic,netparams.conn_dict_inh_neurons,self.self_inh_syn_params)
        
        #Connect spike detectors to neuron populations
        nest.Connect(self.vc_exc_tonic,self.spike_detector_vc_exc_tonic,"one_to_one")
        self.spike_detector_vc_exc_tonic.n_events = 0	#ensure no spikes left from previous simulations
        nest.Connect(self.vc_inh_tonic,self.spike_detector_vc_inh_tonic,"one_to_one")
        self.spike_detector_vc_inh_tonic.n_events = 0	#ensure no spikes left from previous simulations
                    
        #Connect multimeters to neuron populations
        nest.Connect(self.mm_vc_exc_tonic,self.vc_exc_tonic)
        nest.Connect(self.mm_vc_inh_tonic,self.vc_inh_tonic)      	        
