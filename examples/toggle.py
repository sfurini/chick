#!/usr/bin/env python
"""
Toggle switch

toggle_1
    k_b = 1e-2

toggle_2
    k_b = 1e-1
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle
import functools
print = functools.partial(print, flush=True)
np.set_printoptions(linewidth = np.inf)

import sys
#sys.path.append("/home/simone/msm_cme_runs")
#sys.path.append("/home/simone/msm_cme_runs/chick")
#sys.path.append("/home/simone/MySoftware")
#sys.path.append("/Users/simone/Desktop/MarkovModels_CME/old_tests")
sys.path.append("/home/Marilisa/chick")
sys.path.append("/home/Marilisa/chick/chick")
sys.path.append("/home/Marilisa/")

import chick

def gene_regulatory_network(K, pdf):
    b = 1e-1
    a_r = 1e0*b
    a_f = 50*a_r
    k_A_b = k_b
    k_B_b = k_b
    k_A_f = k_A_b / K**2.0
    k_B_f = k_B_b / K_sym**2.0
    K_A = (k_A_b/k_A_f)**0.5
    K_B = (k_B_b/k_B_f)**0.5
    gnr = chick.engine.Gillespie(dt = 1.0, nsteps = np.inf, total_time = 1e3)
    gnr.add_reaction(chick.engine.Reaction('P_A_f  ->  P_A_f + A',a_f))
    gnr.add_reaction(chick.engine.Reaction('P_A_B  ->  P_A_B + A',a_r))
    gnr.add_reaction(chick.engine.Reaction('P_B_f  ->  P_B_f + B',a_f))
    gnr.add_reaction(chick.engine.Reaction('P_B_A  ->  P_B_A + B',a_r))
    gnr.add_reaction(chick.engine.Reaction('A ->',b))
    gnr.add_reaction(chick.engine.Reaction('B ->',b))
    gnr.add_reaction(chick.engine.Reaction('P_A_f  + 2*B ->  P_A_B',k_A_f))
    gnr.add_reaction(chick.engine.Reaction('P_A_B -> P_A_f  + 2*B',k_A_b))
    gnr.add_reaction(chick.engine.Reaction('P_B_f  + 2*A ->  P_B_A',k_B_f))
    gnr.add_reaction(chick.engine.Reaction('P_B_A -> P_B_f  + 2*A',k_B_b))
    gnr.add_conservation('P_A_f + P_A_B = 1')
    gnr.add_conservation('P_B_f + P_B_A = 1')
    #--- plot equilibrium conditions
    #print 'k_A_f: ',k_A_f
    #print 'k_A_b: ',k_A_b
    #print 'k_B_f: ',k_B_f
    #print 'k_B_b: ',k_B_b
    A = np.linspace(0,a_f/b,1000)
    B = np.linspace(0,a_f/b,1000)
    A_Beq = (1.0/b) * ( a_f + a_r * (B/K_A)**2 ) / ( 1.0 + (B/K_A)**2  )
    B_Aeq = (1.0/b) * ( a_f + a_r * (A/K_B)**2 ) / ( 1.0 + (A/K_B)**2  )
    f = plt.figure()
    ax = f.add_subplot(1,1,1)
    ax.plot(B,A_Beq,'-b')
    ax.plot(B_Aeq,A,'-r')
    plt.xlabel('B')
    plt.ylabel('A')
    pdf.savefig()
    plt.close()
    return gnr

#--- Set parameters
prefix = 'toggle'
k_b = 1e-1
K_sym = 1e1
lags = [1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
n_run_training = 10 # number of training sets tested
learning_rate = 1e-5
batch_size = 1000
n_epochs = 500
n_layers = 4 # number of layers
n_inputs = 6 # number of inputs node (same as number of features)
N_OUTPUTS = [2,] #,5,6]
pdf = PdfPages('./{0:s}.pdf'.format(prefix))

#--- Initialize parameters
K = K_sym*np.array([1e0])
gnr = gene_regulatory_network(K, pdf)
    
#--- Initialize replica
if os.path.isfile('./{0:s}.pk'.format(prefix)):
    print('Reading previos data from: ./{0:s}.pk'.format(prefix))
    with open('./{0:s}.pk'.format(prefix),'rb') as fin:
        replica = pickle.load(fin)
        #cls_uniq = pickle.load(fin)
        #msm_uniq = pickle.load(fin)
else:
    gnr.run(stride = 0, initial_state = {'A': 0, 'B': 0, 'P_B_A': 0, 'P_A_f': 1, 'P_B_f': 1, 'P_A_B': 0})
    replica = chick.replica.Random(gnr)
replica.run(1000)
print('Number of frames: ',replica.n_frames())
print('Species: ',replica.simulation.species)
replica.show(pdf)

#--- Fit MSM - all unique samples
#cls_uniq = chick.cluster.Unique(replica.trajs)
#cls_uniq.fit_predict()
#msm_uniq = chick.sorter.MarkovStateModel(cluster = cls_uniq)
#msm_uniq.timescales_lags(lags = [1,], nits = 10, pdf = pdf)
#n_sims = int(1e3)
#dtraj = msm_uniq.simulate(n_sims)
#traj = cls_uniq.analogic(dtraj)
#f = plt.figure()
#ax = f.add_subplot(1,1,1)
#for i in range(n_inputs):
#    ax.plot(range(n_sims), traj[:,i])
#pdf.savefig()
#plt.close()

#--- Estimate VAMPnets
scores_train = np.empty((len(N_OUTPUTS),n_run_training))
scores_vali = np.empty((len(N_OUTPUTS),n_run_training))
for i_outputs, n_outputs in enumerate(N_OUTPUTS):
    #--- Define number of nodes
    nodes = []
    node_rate = (n_inputs/n_outputs)**(1.0/(n_layers+2))
    if n_inputs > n_outputs:
        nodes.append(int(max(np.floor(n_inputs/node_rate),n_outputs)))
    else:
        nodes.append(int(min(np.ceil(n_inputs/node_rate),n_outputs)))
    for i_layer in range(1,n_layers):
        if n_inputs > n_outputs:
            nodes.append(int(max(np.floor(nodes[-1]/node_rate),n_outputs)))
        else:
            nodes.append(int(min(np.ceil(nodes[-1]/node_rate),n_outputs)))
    print('n_inputs = {0:d}'.format(n_inputs))
    print('n_outputs = {0:d}'.format(n_outputs))
    print('nodes = ',nodes)
    #exit()
    #--- Go !
    N = chick.sorter.Neural(replica, nodes, n_outputs, lags, epsilon = 1e-4)
    scores_train[i_outputs,:], scores_vali[i_outputs,:] = N.fit(n_run_training = n_run_training, train_ratio = 0.9, batch_size = batch_size, n_epochs = n_epochs, learning_rate = learning_rate, pdf = pdf)

##--- Plotting scores
#f = plt.figure()
#ax = f.add_subplot(111)
#ax.errorbar(N_OUTPUTS, scores_train.mean(axis = 1), scores_train.std(axis = 1))
#ax.errorbar(N_OUTPUTS, scores_vali.mean(axis = 1), scores_vali.std(axis = 1))
#pdf.savefig()
#plt.close()
#f = plt.figure()
#ax = f.add_subplot(111)
#inds = np.argsort(scores_train, axis = 1)[:,-1]
#ax.plot(N_OUTPUTS, scores_train[range(scores_train.shape[0]),inds],'o-')
#ax.plot(N_OUTPUTS, scores_vali[range(scores_vali.shape[0]),inds],'s-')
#pdf.savefig()
#plt.close()
#print('scores_train = ',scores_train)
#print('scores_vali = ',scores_vali)

#--- Dumping
with open('./{0:s}.pk'.format(prefix),'wb') as fout:
    pickle.dump(replica,fout)
#    #pickle.dump(cls_uniq,fout)
#    #pickle.dump(msm_uniq,fout)

pdf.close()
exit(0)
