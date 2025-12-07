#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: P5.ipynb
Conversion Date: 2025-12-07T12:19:36.927Z
"""

import networkx as nx
import matplotlib
import matplotlib.pyplot as plt

!pip install ndlib

import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
from ndlib.viz.mpl.DiffusionTrend import DiffusionTrend

#1 Define the network topology 
# create a random erdos-Renyi graph as an example
g = nx.erdos_renyi_graph(1000,0.1)

#  2. Define the epidermic model (eg SIR)
model = ep.SIRModel(g) #ep.SISModel(g)

# 3. configure the model parameters
cfg = mc.Configuration()
cfg.add_model_parameter('beta', 0.01) #infection rate
cfg.add_model_parameter('gamma', 0.005) #recovery rate
cfg.add_model_parameter('percentage_infected', 0.05) #inital percentage of infected nodes
model.set_initial_status(cfg)

#4. execute the simulation
iterations = model.iteration_bunch(200) #simulaed for 200 iterations
trends = model.build_trends(iterations)


%matplotlib inline
# 5. visualize the diffusion trends
viz = DiffusionTrend(model, trends)
viz.plot(filename='sir_diffusion_trend.pdf') #save the plot to a pdf file 
plt.show()

model_sis = ep.SISModel(g)
cfg_sis = mc.Configuration()
cfg_sis.add_model_parameter("beta", 0.02)
cfg_sis.add_model_parameter("lambda", 0.01)
cfg_sis.add_model_parameter("percentage_infected", 0.05)
model_sis.set_initial_status(cfg_sis)

its_sis = model_sis.iteration_bunch(200)
tr_sis = model_sis.build_trends(its_sis)

viz_sis = DiffusionTrend(model_sis, tr_sis)
viz_sis.plot(filename="sis_diffusion_trend.pdf")
plt.show()


model_ic = ep.IndependentCascadesModel(g)

cfg_ic = mc.Configuration()
cfg_ic.add_model_initial_configuration("Infected", list(range(10)))
cfg_ic.add_model_parameter("threshold", 0.5)
cfg_ic.add_model_parameter("fraction_infected", 0.05)
model_ic.set_initial_status(cfg_ic)

its_ic = model_ic.iteration_bunch(200)
tr_ic = model_ic.build_trends(its_ic)

viz_ic = DiffusionTrend(model_ic, tr_ic)
viz_ic.plot(filename="ic_diffusion_trend.pdf")
plt.show()