#!/usr/bin/env python

import nest
import numpy as np
import pathlib, sys
import pylab
import math
import matplotlib.pyplot as plt
import random
import pickle, yaml
import time, datetime
import re

#Import parameters for network
file = open(r'configuration_run_nest.yaml')
args = yaml.load(file, Loader=yaml.FullLoader)
print(f'\nLoading parameters from configuration file:\n')

class neural_network():
    def __init__(self):
        self.args = args
        self.optimizing = args['optimizing']
        
        #Define test setup
        self.pulse_duration = args['pulse_duration']
        self.interval_duration = args['interval_duration']
        self.delay_bt_electrodes = args['delay_bt_electrodes']                    
        self.I_e_tonic_mean = args['current_stimulus'] #pA
        self.silence_inhibitory_neurons = args['silence_inhibitory_neurons']
        self.noise_removed = args['noise_removed']
        
        #Define population size
        self.ratio_exc_inh = 4
        self.vc_pop_neurons= 100
        self.exc_tonic_count = int(np.round(self.vc_pop_neurons * (self.ratio_exc_inh / (self.ratio_exc_inh + 1)))) # N_E = N*(r / (r+1))
        self.inh_tonic_count = int(np.round(self.vc_pop_neurons * ( 1 / (self.ratio_exc_inh + 1)))) # N_I = N*(1 / (r+1))
        #self.exc_tonic_count = 50
        #self.inh_tonic_count = 10
        
        #Define synaptic strength
        self.w_exc_multiplier = .56 #Use this to change the network balance through excitatory weights 
        self.w_exc_mean = 0.6/self.ratio_exc_inh+self.w_exc_multiplier*0.6 
        self.w_exc_std = 0.12 
        self.w_inh_mean = -0.6
        self.w_inh_std = 0.12 
        #Define synaptic connectivity
        self.sparsity_vc_ei = 0.5 
        self.sparsity_vc_ie = 0.5
        self.selfexc_vc = 0.25   
        self.selfinh_vc = 0.25     
        #Define rise and decay time constants for synapses 
        self.tau_syn_e_rise = 0.2              
        self.tau_syn_e_decay = 1.0 
        self.tau_syn_i_rise = 0.5 
        self.tau_syn_i_decay = 20.0
        print('Excitatory synaptic rise/decay',self.tau_syn_e_rise,self.tau_syn_e_decay)
        print('Inhibitory synaptic rise/decay',self.tau_syn_i_rise,self.tau_syn_i_decay)
        
        #Define simulation parameters
        if len(sys.argv) > 1:
            self.rng_seed = int(sys.argv[1]) 
        else:
            self.rng_seed = args['seed']
        #self.rng_seed = np.random.randint(10**7) if args['seed'] == 0 else args['seed'] #set seed for NEST 	
        self.time_resolution = args['delta_clock'] 		#equivalent to "delta_clock"        
        self.sim_time = args['t_steps']         #time in ms
        
        #Define neuronal parameters
        self.V_th_mean_tonic = -50.0 #mV  
        self.V_th_std_tonic = 1.0 #mV
        self.V_m_mean = -60.0 #mV 
        self.V_m_std = 10.0 #mV  
        self.C_m_tonic_mean = 200.0 #pF         
        self.C_m_tonic_std = 40.0 #pF       
        self.t_ref_mean = 1.0 #ms
        self.t_ref_std = 0.2 #ms
        
        self.synaptic_delay = 2. #args['synaptic_delay']
       
        print('Running test with mean current: ',self.I_e_tonic_mean)
        self.I_e_tonic_std = 0.25*self.I_e_tonic_mean #pA
        self.noise_std_dev_tonic = self.I_e_tonic_mean #pA
        print('Noise standard deviation ',self.noise_std_dev_tonic)

        #Set data evaluation parameters
        self.convstd_rate = args['convstd_rate']
        self.chop_edges_amount = args['chop_edges_amount']
        self.remove_mean = args['remove_mean']
        self.high_pass_filtered = args['high_pass_filtered']
        self.downsampling_convolved = args['downsampling_convolved']
        self.remove_silent = args['remove_silent']
        self.calculate_balance = args['calculate_balance']               
        self.raster_plot = args['raster_plot']
        self.rate_coded_plot = args['rate_coded_plot']
        self.membrane_potential_plot = args['membrane_potential_plot']
        self.time_window = args['smoothing_window']
        self.time_window_indiv_neurons = args['smoothing_window_indiv_neurons']

        #Set spike detector parameters 
        self.sd_params = {"withtime" : True, "withgid" : True, 'to_file' : False, 'flush_after_simulate' : False, 'flush_records' : True}
        
        #Connection parameters based on spatial parameters
        #Set decay rate - decrease for faster decay, increase for wider spread
        decay_exc_neuron=1. 
        decay_inh_neuron=1.  
        decay_self_neuron=1. 
        decay_electrode_conn=4. #Determines the probability of connection based on distance
        decay_electrode_weight=0.35 #Determines the strength of connection based on distance TESTING .35
        
        self.electrode_syn_params = {"synapse_model":"static_synapse",
            "weight" : nest.spatial_distributions.exponential(nest.spatial.distance, beta=decay_electrode_weight), #nS            
            "delay" : self.synaptic_delay}	#ms
        
        self.conn_dict_exc_neurons = {
            'rule': 'pairwise_bernoulli',
            "p": nest.spatial_distributions.exponential(nest.spatial.distance, beta=decay_exc_neuron), #e^(-x/beta)
            'allow_autapses': False
        }
        
        self.conn_dict_inh_neurons = {
            'rule': 'pairwise_bernoulli',
            "p": nest.spatial_distributions.exponential(nest.spatial.distance, beta=decay_inh_neuron), #e^(-x/beta)
            'allow_autapses': False
        }
        
        self.conn_dict_self_neurons = {
            'rule': 'pairwise_bernoulli',
            "p": nest.spatial_distributions.exponential(nest.spatial.distance, beta=decay_self_neuron), #e^(-x/beta)
            'allow_autapses': False
        }
        
        self.conn_dict_electrodes = {
            'rule': 'pairwise_bernoulli',
            "p": nest.spatial_distributions.exponential(nest.spatial.distance, beta=decay_electrode_conn), #e^(-x/beta) 
            'allow_autapses': False
        }
        
        #Connection parameters - no spatial parameters
        self.conn_dict_custom_selfexc_vc = {'rule': 'pairwise_bernoulli', 'p': self.selfexc_vc}
        self.conn_dict_custom_selfinh_vc = {'rule': 'pairwise_bernoulli', 'p': self.selfinh_vc}
        self.conn_dict_custom_vc_ei = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_vc_ei}
        self.conn_dict_custom_vc_ie = {'rule': 'pairwise_bernoulli', 'p': self.sparsity_vc_ie}
        
        #Set multimeter parameters
        self.mm_params = {'interval': 1., 'record_from': ['V_m']}

        #Set noise parameters
        self.noise_params_tonic = {"dt": self.time_resolution, "std":self.noise_std_dev_tonic}
   
    ################
    # Save results #
    ################
    if args['save_results'] and not args['optimizing']:
        if len(sys.argv) > 1:
            id_ = str(int(sys.argv[1]))+'_'+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        else:
            id_ = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        path = 'saved_simulations' + '/' + id_ 
        pathFigures = 'saved_simulations' + '/' + id_ + '/Figures'
        pathlib.Path(path).mkdir(parents=True, exist_ok=False)
        pathlib.Path(pathFigures).mkdir(parents=True, exist_ok=False)
        with open(path + '/args_' + id_ + '.yaml', 'w') as yamlfile:
            yaml.dump(args, yamlfile)
      

