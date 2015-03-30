#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import nengo
from nengo.utils.functions import piecewise
from nengo.utils.ensemble import tuning_curves
import scipy.interpolate
import pyopencl as cl
import pyopencl.array as cl_array
import time

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


time_nengo_start = time.clock()
sim.run(6) # do this so that we build the model, but also to get a benchmark time
time_nengo = time.clock() - time_nengo_start
print("Nengo runtime: " + str(time_nengo) + " seconds")

rng = np.random.RandomState()
base_decoders = sim.model.sig[conn_recurrent]['decoders'].value
decoders = np.dot(S[0:npc, 0:npc], np.dot(u[:,0:npc].transpose(), base_decoders.transpose()))

# simulate on the CPU with surrogate populations
dt = sim.model.dt
x_recurrent = 0.0
pstc_state = 0.0
pstc_alpha = 1.0 - np.exp(-dt / tau)
surrogate_data = []

time_surrogate_cpu_start = time.clock()
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

time_surrogate_cpu = time.clock() - time_surrogate_cpu_start
print("Surrogate CPU time: " + str(time_surrogate_cpu) + " seconds")
    
# simulate with Actual Neurons(TM) and plot results
plt.figure()
plt.title("Simulation results")
plt.plot(sim.trange(), sim.data[input_probe], label="Input")
plt.plot(sim.trange(), sim.data[A_probe], label="Integrator output")
plt.plot(sim.trange(), surrogate_data, label="Surrogate simulation output")

# now do it on the GPU

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
n_gpu_populations = 1

gpu_pstc_state = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.zeros(n_gpu_populations).astype(np.float32))

pc_samples = 1024
host_pc_samples = []
sample_points = np.linspace(-2.0, 2.0, pc_samples)
for n in range(7):
    samples = np.empty(pc_samples).astype(np.float32)
    for i in range(len(samples)):
        samples[i] = principal_component_functions[n](sample_points[i])
    host_pc_samples.append(samples)

host_pc_samples_all = []
for n in range(7):
    host_pc_samples_all += host_pc_samples[n].tolist()
    
gpu_pc_samples = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.array(host_pc_samples_all).astype(np.float32))

kernel_file = open("surrogate1d.cl", 'r')
kernel_str = "".join(kernel_file.readlines())
kernel = cl.Program(ctx, kernel_str).build()
kernel_file.close()

# allocate and copy decoders
host_decoders = []
for n in range(n_gpu_populations):
    for d in decoders:
        host_decoders.append(np.real(d[0]))
gpu_decoders = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.array(host_decoders).astype(np.float32))

time_surrogate_gpu_start = time.clock()
last_event = None
decoded_outputs = []
gpu_output_buffers = []
for i in range(2):
    gpu_output_buffers.append(cl_array.to_device(queue, np.zeros(n_gpu_populations).astype(np.float32)))
polarity = 0
for t in sim.trange():
    gpu_x_recurrent = gpu_output_buffers[polarity]
    gpu_decoded_output = gpu_output_buffers[1-polarity]
    kevent = kernel.surrogate1d(queue, (n_gpu_populations, 1), None,
                                np.float32(input.output(t)), gpu_x_recurrent.data,
                                np.float32(tau), np.float32(pstc_alpha), gpu_pstc_state,
                                gpu_pc_samples, gpu_decoders, gpu_decoded_output.data,
                                wait_for = last_event)
    last_event = [kevent]
    host_decoded_output = gpu_decoded_output.get()
    decoded_outputs.append(host_decoded_output)
    polarity = 1 - polarity
    
queue.finish()
time_surrogate_gpu = time.clock() - time_surrogate_gpu_start
print("Surrogate GPU time: " + str(time_surrogate_gpu) + " seconds")

# get output from GPU
surrogate_gpu_output = []
for i in range(len(sim.trange())):
    surrogate_gpu_output.append(decoded_outputs[i][0])

plt.plot(sim.trange(), surrogate_gpu_output, label="Surrogate GPU output")
plt.legend()    
plt.show()
