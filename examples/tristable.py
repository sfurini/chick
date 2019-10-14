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
#sys.path.append('/Users/marilisacortesi/')
#sys.path.append('/Users/marilisacortesi/chick')
#sys.path.append('/Users/marilisacortesi/Desktop/deeptime-master')
sys.path.append("/home/Marilisa/chick")
sys.path.append("/home/Marilisa/chick/chick")
sys.path.append("/home/Marilisa/")
import chick

def kAA(A):
    return A*(A-1)*(A-2)*(A-3)

def kAB(B):
    return B

def kBB(B):
    return B*(B-1)*(B-2)*(B-3)

def kBA(A):
    return A

def gene_regulatory_network(omega, pdf):
    g_00=5
    lA=8
    lR=0.2
    g_00_lA=g_00*lA
    g_00_lR=g_00*lR
    g_00_lA_lR=g_00*lA*lR
    k=0.1
    X_eq1=15
    X_eq2=50**4
    f=k*omega
    k_AA_f=f/X_eq2
    k_AB_f=f/X_eq1
    k_BB_f=f/X_eq2
    k_BA_f=f/X_eq1

    
    gnr = chick.engine.Gillespie(dt = 1.0, nsteps = np.inf, total_time = 1e3)
    gnr.add_reaction(chick.engine.Reaction('P_A_f  ->  P_A_f + A',g_00)) # 1
    gnr.add_reaction(chick.engine.Reaction('P_A_A  ->  P_A_A + A',g_00_lA)) # 2
    gnr.add_reaction(chick.engine.Reaction('P_A_B  ->  P_A_B + A',g_00_lR)) # 3
    gnr.add_reaction(chick.engine.Reaction('P_A_AB  ->  P_A_AB + A',g_00_lA_lR)) #4
    gnr.add_reaction(chick.engine.Reaction('A ->',k)) # 5
    gnr.add_reaction(chick.engine.Reaction('P_A_f  + 4*A ->  P_A_A',k_AA_f, rate_patch = kAA)) # 6
    gnr.add_reaction(chick.engine.Reaction('P_A_A -> P_A_f + 4*A',k)) # 7
    gnr.add_reaction(chick.engine.Reaction('P_A_f  + B ->  P_A_B',k_AB_f, rate_patch = kAB)) # 8
    gnr.add_reaction(chick.engine.Reaction('P_A_B -> P_A_f + B',k)) # 9
    gnr.add_reaction(chick.engine.Reaction('P_A_A  + B ->  P_A_AB',k_AB_f, rate_patch = kAB)) # 10
    gnr.add_reaction(chick.engine.Reaction('P_A_AB -> P_A_A + B',k)) # 11
    gnr.add_reaction(chick.engine.Reaction('P_A_B  + 4*A ->  P_A_AB',k_AA_f, rate_patch = kAA)) # 12
    gnr.add_reaction(chick.engine.Reaction('P_A_AB -> P_A_B + 4*A',k)) # 13
    gnr.add_conservation('P_A_f + P_A_A + P_A_B + P_A_AB = 1') # 14
    gnr.add_reaction(chick.engine.Reaction('P_B_f  ->  P_B_f + B',g_00)) # 15
    gnr.add_reaction(chick.engine.Reaction('P_B_A  ->  P_B_A + B',g_00_lR)) # 16
    gnr.add_reaction(chick.engine.Reaction('P_B_B  ->  P_B_B + B',g_00_lA)) # 17
    gnr.add_reaction(chick.engine.Reaction('P_B_AB  ->  P_B_AB + B',g_00_lA_lR)) # 18
    gnr.add_reaction(chick.engine.Reaction('B ->',k)) # 19
    gnr.add_reaction(chick.engine.Reaction('P_B_f  + 4*B ->  P_B_B',k_BB_f, rate_patch = kBB)) # 20
    gnr.add_reaction(chick.engine.Reaction('P_B_B -> P_B_f + 4*B',k)) # 21
    gnr.add_reaction(chick.engine.Reaction('P_B_f  + A ->  P_B_A',k_BA_f, rate_patch = kBA)) # 22
    gnr.add_reaction(chick.engine.Reaction('P_B_A -> P_B_f + A',k)) # 23
    gnr.add_reaction(chick.engine.Reaction('P_B_B  + A ->  P_B_AB',k_BA_f, rate_patch = kBA)) # 24
    gnr.add_reaction(chick.engine.Reaction('P_B_AB -> P_B_B + A',k)) # 25
    gnr.add_reaction(chick.engine.Reaction('P_B_A  + 4*B ->  P_B_AB',k_BB_f, rate_patch = kBB)) # 26
    gnr.add_reaction(chick.engine.Reaction('P_B_AB -> P_B_A + 4*B',k)) # 27
    gnr.add_conservation('P_B_f + P_B_A + P_B_B + P_B_AB = 1') # 28


    #--- plot equilibrium conditions
    #print 'k_A_f: ',k_A_f
    #print 'k_A_b: ',k_A_b
    #print 'k_B_f: ',k_B_f
    #print 'k_B_b: ',k_B_b
    #A = np.linspace(0,g_00/k,1000)
    #B = np.linspace(0,g_00/k,1000)
    #A_Beq = (1.0/b) * ( a_f + a_r * (B/K_A)**2 ) / ( 1.0 + (B/K_A)**2  )
    #B_Aeq = (1.0/b) * ( a_f + a_r * (A/K_B)**2 ) / ( 1.0 + (A/K_B)**2  )
    #f = plt.figure()
    #ax = f.add_subplot(1,1,1)
    #ax.plot(B,A_Beq,'-b')
    #ax.plot(B_Aeq,A,'-r')
    #plt.xlabel('B')
    #plt.ylabel('A')
    #pdf.savefig()
    #plt.close()
    return gnr

#--- Set parameters
prefix = 'tristable'
#k_b = 1e-1
#K_sym = 1e1
lags = [1,2] #,3,4,5] #5,10,25,50,75,100]
n_run_training = 10000 # number of training sets tested
learning_rate = 1e-5
batch_size = 1000
n_epochs = 500
n_layers = 4 # number of layers
n_inputs = 10 # number of inputs node (same as number of features)
N_OUTPUTS = [3,] #5,6]
pdf = PdfPages('./{0:s}.pdf'.format(prefix))
#--- Initialize parameters
omega = 100
gnr = gene_regulatory_network(omega, pdf)
    
#--- Initialize replica
if os.path.isfile('./{0:s}.pk'.format(prefix)):
    print('Reading previos data from: ./{0:s}.pk'.format(prefix))
    with open('./{0:s}.pk'.format(prefix),'rb') as fin:
        replica = pickle.load(fin)
        #cls_uniq = pickle.load(fin)
        #msm_uniq = pickle.load(fin)
else:
    gnr.run(stride = 0, initial_state = {'A': 0, 'B': 0, 'P_A_f': 1, 'P_A_A':0, 'P_A_B':0,'P_A_AB':0,'P_B_f': 1, 'P_B_A': 0, 'P_B_B':0, 'P_B_AB':0})
    replica = chick.replica.Random(gnr)
replica.run(10000)
print ('Number of frames: ',replica.n_frames())
print('Species: ',replica.simulation.species)
#replica.show(pdf)
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

    #--- Go !
    N = chick.sorter.Neural(replica, nodes, n_outputs, lags, epsilon = 1e-4)
    print(pdf)	
    scores_train[i_outputs,:], scores_vali[i_outputs,:] = N.fit(n_run_training = n_run_training, train_ratio = 0.9, batch_size = batch_size, n_epochs = n_epochs, learning_rate = learning_rate, pdf = pdf)

#--- Plotting scores
f = plt.figure()
ax = f.add_subplot(111)
ax.errorbar(N_OUTPUTS, scores_train.mean(axis = 1), scores_train.std(axis = 1))
ax.errorbar(N_OUTPUTS, scores_vali.mean(axis = 1), scores_vali.std(axis = 1))
pdf.savefig()
plt.close()
f = plt.figure()
ax = f.add_subplot(111)
inds = np.argsort(scores_train, axis = 1)[:,-1]
ax.plot(N_OUTPUTS, scores_train[range(scores_train.shape[0]),inds],'o-')
ax.plot(N_OUTPUTS, scores_vali[range(scores_vali.shape[0]),inds],'s-')
pdf.savefig()
plt.close()
print('scores_train = ',scores_train)
print('scores_vali = ',scores_vali)

#--- Dumping
with open('./{0:s}.pk'.format(prefix),'wb') as fout:
    pickle.dump(replica,fout)
#    #pickle.dump(cls_uniq,fout)
#    #pickle.dump(msm_uniq,fout)
pdf.close()
exit(0)
