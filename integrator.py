#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import nengo
from nengo.utils.functions import piecewise
from nengo.utils.ensemble import tuning_curves
import scipy.interpolate
import scipy.special

# Legendre polynomial of degree n.
# The closed-form representation is
# P_n(x) = 2^n * (sum from k=0 to n) x^k * (n choose k)*([n+k+-1]/2 choose n) 
def legendre(n):
    coeffs = []
    for k in range(n+1):
        c1 = scipy.special.binom(n, k)
        c2 = scipy.special.binom((n+k-1)/2, n)
        coeffs.append(c1 * c2)
    def L(x):
        accumulator = np.zeros(x.shape)
        argument = np.ones(x.shape)
        for k in range(n+1):
            accumulator += np.multiply(argument, coeffs[k])
            argument = np.multiply(argument, x)
        return 2.0**n * accumulator
    return L


model = nengo.Network(label='Integrator')
with model:
    A = nengo.Ensemble(100, dimensions=1)
    input = nengo.Node(piecewise({0:0, 0.2:1, 1:0, 2:-2, 3:0, 4:1, 5:0}))
    tau = 0.1
    conn_recurrent = nengo.Connection(A, A, transform=[[1]], synapse=tau)
    conn_input = nengo.Connection(input, A, transform=[[tau]], synapse=tau)
    input_probe = nengo.Probe(input)
    A_probe = nengo.Probe(A, synapse=0.01)

sim = nengo.Simulator(model)
# find 1-D principal components and plot

eval_points_in = np.linspace(-2.0, 2.0, 50)
eval_points_in = np.ndarray(shape=(50,1), buffer=eval_points_in)
eval_points, activities = tuning_curves(A, sim, eval_points_in)
#plt.figure()
#plt.title("Tuning curves")
#plt.plot(eval_points, activities)

u, s, v = np.linalg.svd(activities.transpose())
# plot first 'npc' principal components
npc = 7
S = np.zeros((u.shape[0], v.shape[0]), dtype=complex)
S[:s.shape[0], :s.shape[0]] = np.diag(s)
usi = np.linalg.pinv(np.dot(u, S))
principal_components = np.real(np.dot(usi[0:npc, :], activities.transpose()))
principal_component_functions = []
for n in range(npc):
    pc = principal_components[n]
    f = scipy.interpolate.interp1d(eval_points[:,0], pc, kind='cubic')
    principal_component_functions.append(f)
#for n in range(npc):
#    pc = principal_components[n]
#    fig = plt.figure()
#    plt.title("Principal component " + str(n))
#    plt.plot(eval_points, pc)
# compute approximate decoders
sim.run(6) # cheating. we only do this so that we build the model
rng = np.random.RandomState()
base_decoders = sim.model.sig[conn_recurrent]['decoders'].value
decoders = np.dot(S[0:npc, 0:npc], np.dot(u[:,0:npc].transpose(), base_decoders.transpose()))

# simulate on the CPU with surrogate populations
dt = sim.model.dt
x_recurrent = 0.0
pstc_state = 0.0
pstc_alpha = 1.0 - np.exp(-dt / tau)
surrogate_data = []
for t in sim.trange():
    # get the output value from our piecewise input
    x_in = input.output(t)
    # encode input by multiplying each connection with its weight
    input_prefilter = x_in * tau + x_recurrent
    # low-pass filter to simulate post-synaptic current dynamics
    input_postfilter = input_prefilter * pstc_alpha + pstc_state * (1 - pstc_alpha)
    pstc_state = input_postfilter
    # for each principal component, find the corresponding output given input = input_postfilter
    pc_outputs = []
    for n in range(npc):
        f = principal_component_functions[n]
        y = f(input_postfilter)
        y += np.random.normal(0.0, 0.002)
        pc_outputs.append(y)
    # calculate decoded output as pc_outputs (dot) decoders
    pc_outputs = np.ndarray(shape=(1, 7), buffer=np.array(pc_outputs))
    decoded_output = np.real(np.dot(pc_outputs, decoders)[0,0]);
    surrogate_data.append(decoded_output)
    x_recurrent = decoded_output

# simulate with Actual Neurons(TM) and plot results
plt.figure()
plt.title("Simulation results")
plt.plot(sim.trange(), sim.data[input_probe], label="Input")
plt.plot(sim.trange(), sim.data[A_probe], label="Integrator output")
plt.plot(sim.trange(), surrogate_data, label="Surrogate simulation output")
plt.legend()
plt.show()
