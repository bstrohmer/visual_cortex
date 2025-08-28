#!/usr/bin/env python

import matplotlib.pyplot as plt
import nest
import numpy as np
import os

from pynestml.frontend.pynestml_frontend import generate_nest_target

neuron_model = "aeif_cond_beta"

files = [os.path.join("models", "neurons", neuron_model + ".nestml")]
input_path = "/opt/miniconda3/envs/als_nestml/models/neurons/aeif_cond_beta.nestml"

generate_nest_target(input_path=input_path,
                     target_path="/opt/miniconda3/envs/als_nestml/models/aeif_cond_beta_component",
                     logging_level="DEBUG",
                     module_name="nestml_aeif_cond_beta_module",
                     suffix="_aeif_cond_beta_nestml")
