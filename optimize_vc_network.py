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
    def __init__(self,inh_pop_size,w_exc,w_inh,w_selfexc,w_selfinh,sparsity_interconn,sparsity_selfconn):
    #def __init__(self,inh_pop_size,w_inh,w_selfinh,sparsity_interconn,sparsity_selfconn):    
    #def __init__(self,inh_pop_size,w_exc,w_selfexc,sparsity_interconn,sparsity_selfconn):
        self.senders = []
        self.spiketimes = []
        self.saved_spiketimes = []
        self.saved_senders = []
        self.time_window = 50		#50*0.1=5ms time window, based on time resolution of 0.1
        self.count = 0
        
        #Parameters for optimization
        self.inh_pop_size = inh_pop_size
        self.w_exc = w_exc
        self.w_inh = w_inh
        self.w_selfexc = w_selfexc
        self.w_selfinh = w_selfinh
        self.sparsity_interconn = sparsity_interconn
        self.sparsity_selfconn = sparsity_selfconn
        
        #Create populations for the visual cortex
        self.tonic_neuronparams_exc = {'C_m':nest.random.normal(mean=netparams.C_m_tonic_mean, std=netparams.C_m_tonic_std), 
                                   'g_L':10.,
                                   'E_L':-70.,
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
                                   'E_L':-70.,
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
        	
        self.exc_pop_size = netparams.vc_pop_neurons-self.inh_pop_size
        print('Pop size (exc, inh)',self.exc_pop_size,self.inh_pop_size)
        self.vc_exc_tonic = nest.Create('aeif_cond_beta_aeif_cond_beta_nestml',self.exc_pop_size,self.tonic_neuronparams_exc) 	
        self.vc_inh_tonic = nest.Create('aeif_cond_beta_aeif_cond_beta_nestml',self.inh_pop_size,self.tonic_neuronparams_inh) 
        
        #Create pulse input
        self.amplitude_times_p1 = []
        self.amplitude_times_p2 = []
        current_time = 0.0

        while current_time < netparams.sim_time:
            self.amplitude_times_p1.append(current_time)                # Start of pulse
            self.amplitude_times_p1.append(current_time + netparams.pulse_duration)  # End of pulse
            current_time += netparams.pulse_duration + netparams.interval_duration  # Move to the next cycle
        self.amplitude_times_p2 = [t + netparams.delay_bt_probes for t in self.amplitude_times_p1]
        # If the last time exceeds sim_time, truncate it
        self.amplitude_times_p1 = [t for t in self.amplitude_times_p1 if t <= netparams.sim_time]
        self.amplitude_times_p2 = [t for t in self.amplitude_times_p2 if t > netparams.sim_time/2 and t <= netparams.sim_time]
        self.amplitude_values_p1 = [netparams.I_e_tonic_mean if i % 2 == 0 else 0 for i in range(len(self.amplitude_times_p1))]
        self.amplitude_values_p2 = [netparams.I_e_tonic_mean if i % 2 == 0 else 0 for i in range(len(self.amplitude_times_p2))]

        print("Amplitude times (p1):", self.amplitude_times_p1)
        print("Amplitude times (p2):", self.amplitude_times_p2)
        print("Amplitude values (p1):", self.amplitude_values_p1)
        print("Amplitude values (p2):", self.amplitude_values_p2)

        self.step_current_p1 = nest.Create("step_current_generator")
        nest.SetStatus(self.step_current_p1, {
            "amplitude_times": self.amplitude_times_p1[1:], #skip time 0
            "amplitude_values": self.amplitude_values_p1[1:]
        })
        
        self.step_current_p2 = nest.Create("step_current_generator")
        nest.SetStatus(self.step_current_p2, {
            "amplitude_times": self.amplitude_times_p2[1:], #skip time 0
            "amplitude_values": self.amplitude_values_p2[1:]
        })
        
        #Create noise
        self.white_noise_tonic = nest.Create("noise_generator",netparams.noise_params_tonic)     
        
        #Create spike detectors (for recording spikes) 
        self.spike_detector_vc_exc_tonic = nest.Create("spike_recorder",self.exc_pop_size)
        self.spike_detector_vc_inh_tonic = nest.Create("spike_recorder",self.inh_pop_size)
                
        #Create multimeters (for recording membrane potential)
        self.mm_vc_exc_tonic = nest.Create("multimeter",netparams.mm_params)
        self.mm_vc_inh_tonic = nest.Create("multimeter",netparams.mm_params)
        
        #Connect current generator to neurons
        nest.Connect(self.step_current_p1,self.vc_exc_tonic,"all_to_all")
        nest.Connect(self.step_current_p1,self.vc_inh_tonic,"all_to_all")
        
        #if netparams.num_probes == 2:
        nest.Connect(self.step_current_p2,self.vc_exc_tonic,"all_to_all")
        nest.Connect(self.step_current_p2,self.vc_inh_tonic,"all_to_all")
	
        #Connect white noise to neurons
        nest.Connect(self.white_noise_tonic,self.vc_exc_tonic,"all_to_all") 
        nest.Connect(self.white_noise_tonic,self.vc_inh_tonic,"all_to_all") 
	
        #Connect neurons within vc
        self.inh_syn_params = {"synapse_model":"static_synapse",
            "weight" : nest.random.normal(mean=self.w_inh,std=self.w_inh*-.25), #nS            
            "delay" : netparams.synaptic_delay}	#ms
        self.exc_syn_params = {"synapse_model":"static_synapse",
            "weight" : nest.random.normal(mean=self.w_exc,std=self.w_exc*.25), #nS
            "delay" : netparams.synaptic_delay}	#ms
        self.self_inh_syn_params = {"synapse_model":"static_synapse",
            "weight" : nest.random.normal(mean=self.w_inh,std=self.w_inh*-.25), #nS            
            "delay" : netparams.synaptic_delay}	#ms
        self.self_exc_syn_params = {"synapse_model":"static_synapse",
            "weight" : nest.random.normal(mean=self.w_exc,std=self.w_exc*.25), #nS
            "delay" : netparams.synaptic_delay}	#ms
        
        self.conn_dict_custom_selfexc_vc = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_selfconn}
        self.conn_dict_custom_selfinh_vc = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_selfconn}
        self.conn_dict_custom_vc = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_interconn}
             	  
        self.coupling_exc_tonic_exc_tonic = nest.Connect(self.vc_exc_tonic,self.vc_exc_tonic,self.conn_dict_custom_selfexc_vc,self.self_exc_syn_params)
        self.coupling_exc_tonic_inh_tonic = nest.Connect(self.vc_exc_tonic,self.vc_inh_tonic,self.conn_dict_custom_vc,self.exc_syn_params)            
        self.coupling_inh_tonic_inh_tonic = nest.Connect(self.vc_inh_tonic,self.vc_inh_tonic,self.conn_dict_custom_selfinh_vc,self.inh_syn_params)    
        self.coupling_inh_tonic_exc_tonic = nest.Connect(self.vc_inh_tonic,self.vc_exc_tonic,self.conn_dict_custom_vc,self.self_inh_syn_params) 

        #Connect spike detectors to neuron populations
        nest.Connect(self.vc_exc_tonic,self.spike_detector_vc_exc_tonic,"one_to_one")
        self.spike_detector_vc_exc_tonic.n_events = 0	#ensure no spikes left from previous simulations
        nest.Connect(self.vc_inh_tonic,self.spike_detector_vc_inh_tonic,"one_to_one")
        self.spike_detector_vc_inh_tonic.n_events = 0	#ensure no spikes left from previous simulations
                    
        #Connect multimeters to neuron populations
        nest.Connect(self.mm_vc_exc_tonic,self.vc_exc_tonic)
        nest.Connect(self.mm_vc_inh_tonic,self.vc_inh_tonic)      	        
