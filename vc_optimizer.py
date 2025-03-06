import nest
import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer
import matplotlib.pyplot as plt
import optimize_vc_output as sim
from set_network_params import neural_network
netparams = neural_network()

# Define the search space for optimal parameters
search_space = [
    Integer(2, 15),        # Inh pop size 
    Real(0.01, 2.),     # Exc weight
    Real(-2., -0.01),   # Inh weight
    Real(0.01, 2.),     # Self-exc weight 
    Real(-1., -0.01),   # Self-inh weight
    Real(0.5, 1.),       # Conn% E-I, I-E
    Real(0.1, 0.5)        # Conn% E-E, I-I
]

def wrapped_simulation(*params):
    # Run the simulation and get the result
    score_value, plotting_output = sim.run_vc_simulation(*params)
    
    # Return only the value to minimize (diff_value) to gp_minimize
    return score_value    
    
# Run the Bayesian optimization
result = gp_minimize(
    wrapped_simulation,                 # The NEST simulation function
    dimensions=search_space,            # The search space for optimization
    n_calls=10,                         # Number of optimization iterations
    random_state=netparams.rng_seed           # Seed for reproducibility
)

# Get the optimal parameters
optimal_parameters = result.x
print(f"Optimal parameters: {optimal_parameters}")

# Plot the convergence of the optimization process
from skopt.plots import plot_convergence
plt.figure() 
plot_convergence(result)

# Check how close the optimal weights got to the target
final_difference = result.fun
print(f"Final score: {final_difference}")

diff_value, optimal_output = sim.run_vc_simulation(optimal_parameters)
min_bin_array_exc,max_bin_array_exc,min_bin_array_inh,max_bin_array_inh = optimal_output

# Plot results of the optimal solution
t = np.arange(0,len(min_bin_array_exc),1)
fig, ax = plt.subplots(2,sharex='all')
ax[0].plot(t, min_bin_array_exc)
ax[0].plot(t, max_bin_array_exc)
ax[1].plot(t, min_bin_array_inh)
ax[1].plot(t, max_bin_array_inh)		
for i in range(2):
    ax[i].set_xticks([])
    ax[i].set_xlim(0,len(min_bin_array_exc))
ax[1].set_xlabel('Time (ms)')
ax[1].set_xlim(0,len(min_bin_array_exc))
ax[0].legend(['Min_exc', 'Max_exc'],loc='upper right',fontsize='x-small') 
ax[1].legend(['Min_inh', 'Max_inh'],loc='upper right',fontsize='x-small') 
ax[0].set_title("Excitatory min and max firing neuron")
ax[1].set_title("Inhibitory min and max firing neuron")
figure = plt.gcf() # get current figure
figure.set_size_inches(8, 6)
plt.tight_layout()
#if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + 'rate_coded_output.pdf',bbox_inches="tight")

plt.show()
