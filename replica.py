#!/usr/bin/env python

import sys
import pickle
import numpy as np
import scipy as sp
import numpy.ma as ma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages
from math import *
from collections import *
from subprocess import call
from scipy import constants
from scipy import interpolate
from scipy import io
from scipy.sparse import *
from scipy.sparse import linalg
from scipy.optimize import minimize

sys.path.append("/home/simone/MySoftware")
sys.path.append("/Users/simone/Desktop")

class Random(object):
    """
    Class for multiple simulations

    Attributes
    ----------
    simulation : class
        Class used to run simulations
        It is necessary that this class has the method self.run that run a simulation and creates simulation.X
            where simulation.X is a np.ndarray with shape <number of samples> X <number of features>
    trajs : list
        List of trajectories
        Each trajectory is a np.ndarray with shape: <number of samples> x <number of features>
    """
    def __init__(self, simulation = None, trajs = []):
        self.simulation = simulation # class that is used for running one simulation
        self.trajs = [] # analogical trajectories
        for traj in trajs:
            self.add_trajectory(traj)
    def n_trajs(self):
        """Return the number of trajectories"""
        return len(self.trajs)
    def n_dims(self):
        """Return the number of features"""
        if not self.trajs:
            return np.nan
        list_n_dims = [self.n_dims_traj(traj) for traj in self.trajs]
        if all(list_n_dims[0] == other for other in list_n_dims):
            return list_n_dims[0]
        raise ValueError('ERROR: inconsistent dimensions')
    def n_dims_traj(self, traj):
        """Return the dimensionality of a trajectory"""
        if len(np.shape(traj)) == 1:
            return 1
        else:
            return np.shape(traj)[1]
    def n_frames(self):
        """Return the total number of frames"""
        n = 0
        for traj in self.trajs:
            n += np.shape(traj)[0]
        return n
    def n_frames_longest(self):
        """Return the number of frames of the longest trajectory"""
        return max([traj.shape[0] for traj in self.trajs])
    def reset_trajs(self):
        """Remove all the trajectories"""
        self.trajs = []
    def resample_trajs(self, stride):
        """Change the sampling period of all the trajectories"""
        trajs_resampled = []
        for traj in self.trajs:
            if (traj.shape[0]/stride) > 2:
                trajs_resampled.append(traj[::stride,:])
        self.trajs = trajs_resampled
    def add_trajectory(self, traj, verbose = False):
        ind_samples = np.ones(np.shape(traj)[0]).astype(bool)
        if np.any(ind_samples):
            if verbose:
                print("Adding trajectory with ",np.shape(traj[ind_samples,:])," (samples, dimensions) to the set")
            self.trajs.append(traj[ind_samples,:])
    def read(self, file_name):
        """
        Input
        -----
        file_name : str
            The name of the pickle file
        """
        with open(file_name,'rb') as fin:
            self.trajs.extend(pickle.load(fin))
    def possible_values(self, i_dim):
        """
        Return all the possible values sampled along dimension i_dim
        
        Parameter
        ---------
        i_dim   int
            The index of the dimension analyzed
        """
        print('Retrieving all possible values for dimension {0:d}'.format(i_dim))
        values = set()
        for traj in self.trajs:
            if traj[:,i_dim].dtype != 'int64':
                raise ValueError('ERROR: possible_values only work with integer trajectories')
            values |= set([x for x in traj[:,i_dim]])
        values = np.array(list(values))
        values.sort()
        return values
    def select_frames(self, i_dim, value):
        """
        Return a new Random object that includes only the frames where dimension i_dim correspond to value
        
        Parameter
        ---------
        i_dim   int
            The index of the dimension analyzed
        value   int
            The value selected for dimension i_dim

        Return
        ------
        Random
        """
        new_trajs = []
        for traj in self.trajs:
            if traj[:,i_dim].dtype != 'int64':
                raise ValueError('ERROR: select_frames only work with integer trajectories')
            ind_selected = (traj[:,i_dim] == value)
            new_trajs.append(traj[ind_selected,:])
        return Random(trajs = new_trajs)
    def average_sync(self, pdf, indexes_step, one_plot = True):
        """
        indexes_step    list of integer
            Time points for syncronization
        """
        if len(indexes_step) != self.n_trajs():
            raise ValueError('ERROR: wrong number of indexes_step in average_sync')
        average = np.zeros((self.n_frames_longest(),self.n_dims()))
        n_samples = np.zeros(self.n_frames_longest())
        for i_traj, traj in enumerate(self.trajs):
            if indexes_step[i_traj] >= 0:
                n_steps = traj.shape[0] - indexes_step[i_traj]
                average[:n_steps,:] += traj[indexes_step[i_traj]:,:]
                n_samples[:n_steps] += 1
        average = average / n_samples.reshape(n_samples.shape[0],1)
        last_step_above_half_n_samples = np.where(n_samples > 0.5*np.max(n_samples))[0][-1]
        average = (average - np.nanmean(average, axis = 0))
        f = plt.figure()
        if one_plot:
            ax = f.add_subplot(1,1,1)
        for i_dim in range(self.n_dims()):
            if not one_plot:
                ax = f.add_subplot(self.n_dims(),1,i_dim+1)
            if i_dim == 0:
                plt.title('Averages')
            ax.plot(average[:,i_dim])
            plt.xlim([0, last_step_above_half_n_samples])
        pdf.savefig()
        plt.close()
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.plot(n_samples)
        plt.xlabel('step')
        plt.ylabel('number of samples')
        pdf.savefig()
        plt.close()
        return np.mean(np.nansum(np.abs(average[:last_step_above_half_n_samples,:]),axis = 0))
    def show(self, pdf, one_plot = True, list_index_active = None):
        """
        Plot trajectories
        list_index_active: None
            plot a bar at -10 for the indexes with values > 0
        """
        for i_traj, traj in enumerate(self.trajs):
            f = plt.figure()
            if one_plot:
                ax = f.add_subplot(1,1,1)
            for i_dim in range(self.n_dims()):
                if not one_plot:
                    ax = f.add_subplot(self.n_dims(),1,i_dim+1)
                if i_dim == 0:
                    plt.title('Trajectory '+str(i_traj))
                ax.plot(traj[:,i_dim])
            if list_index_active is not None:
                indexes = np.where(list_index_active[i_traj] > 0)[0]
                ax.plot(indexes, -10*np.ones(indexes.shape), 's', color = 'red')
            pdf.savefig()
            plt.close(f)
        if (self.n_dims() == 2):
            f = plt.figure()
            ax = f.add_subplot(111)
            for i_traj, traj in enumerate(self.trajs):
                traj = self.trajs[i_traj]
                ax.plot(traj[:,0],traj[:,1],'.b')
            plt.xlabel('Reaction Coordinate 0')
            plt.ylabel('Reaction Coordiante 1')
            pdf.savefig()
            plt.close(f)
    def dump(self, file_name):
        with open(file_name,'wb') as fin:
            pickle.dump(self, fin)
    def run(self, n_simulations):
        """
        Parameters
        ----------
        n_simulations : int
            Number of simulations to run
        """
        for i in range(n_simulations):
            print('Running simulation {0:d}/{1:d}'.format(i+1,n_simulations))
            self.simulation.run()
            self.add_trajectory(self.simulation.X)

class FromInitialStates(Random):
    def run(self, initial_states):
        """
        Parameters
        ----------
        initial_states: np.ndarray
            <number of simulations> x <number of dimensions>
        """
        for i in range(initial_states.shape[0]):
            print('Running simulation {0:d}/{1:d}'.format(i,initial_states.shape[0]))
            self.simulation.run(initial_state = initial_states[i,:])
            self.add_trajectory(self.simulation.X)

class ExploreSorter(Random):
    def __init__(self, simulation, sorter):
        super(ExploreSorter, self).__init__(simulation)
        self.sorter = sorter


class ExploreMarkovModel(Random):
    """
    Class for replica simulations directed to improve the accuracy of the estimated Markov Model

    Attributes
    ----------
    cluster: object of type Cluster
        It is used to clusterize the trajectories
    featurize: object of type tica (N.B. andrebbe cambiato in un oggetto generale per fare riduzione della dimensionalita')
        It is used to reduce the number of dimensions in the trajectories
    cluster_micro: object of type Cluster
        It is used to clusterize the trajectories in the reduced space
    markov_model: object of type MarkovModel
        It is used to estimate the Markov Model
    convergence_data: dict
        It is used to store data about the convergence of the Markov Model estimation
    """
    def __init__(self, simulation, cluster, featurize, cluster_micro, markov_model):
        super(ExploreMarkovModel, self).__init__(simulation)
        self.cluster = cluster
        self.featurize = featurize
        self.cluster_micro = cluster_micro
        self.markov_model = markov_model
        self.convergence_data = {}
    def run(self, runs_per_round = 0, max_rounds = 0, nsteps_per_simulation = None
            , numbins_trajs = 100, lag_featurize = 1, n_clusters = 10
            , lags_mm = [1,10,20,50,100], lag_mm = 10, lag_compare = 100, n_timescales = 5
            , pdf = None):
        """
        Parameters
        ----------
        runs_per_step: int
            Number of simulations for each round
        max_steps: int
            Maximum number of rounds
        numbins_trajs: int
            The number of bins used to discretize the original trajectories
            The same number of bins is used for all the dimensions
        lag_featurize: int
            The lag used for dimensionality reduction
        n_clusters: int
            The number of clusters used to discretize the trajectories in the reduced space
        lags_mm: list
            value: int
            The lags used to estimate the lga dependency for the timescales of the markov model
        lag_mm: int
            The lag used to estimate the markov model
        lag_compare: int
            The lag used to test if the timescales are constant
        n_timescales: int
            The number of timescales used to check convergence
        pdf: PDF file
        """
        #--- Initialize parameters
        if (lag_compare not in lags_mm):
            print('ERROR: check parameters\n\tlag_compare needs to be included in lags_mm')
            exit()
        if nsteps_per_simulation is None:
            nsteps_per_simulation = self.simulation.default_nsteps
        #--- Run the first random set of simulations
        if self.n_trajs() == 0:
            super(ExploreMarkovModel, self).run(runs_per_round) # In order to run this step, it is necessary that one stocastic simulation was run before
        #--- Define the clustering strategy for the simulations
        cluster = self.cluster(numbins = numbins_trajs, trajs = self.trajs)
        initial_state = None
        for i_round in range(max_rounds):

            #--- Run a new set of simulations
            if initial_state is None:
                if i_round > 0:
                    print('Starting round {0:d} of random simulations'.format(i_round))
                    super(ExploreMarkovModel, self).run(runs_per_round)
            else:
                print('Starting round {0:d}/{1:d} of simulations from :'.format(i_round,max_rounds),initial_state)
                for i in range(runs_per_round):
                    print('Running simulation {0:d}/{1:d}'.format(i,runs_per_round))
                    self.simulation.run(initial_state, nsteps = nsteps_per_simulation)
                    self.add_trajectory(self.simulation.X)

            #--- Clusterize the trajectory
            cluster.run()
            #cluster.show(pdf, plot_trajs = True, plot_maps = False)

            #--- Map to low-dimensional space
            featurizer = self.featurize(lag = lag_featurize, var_cutoff = 0.90, commute_map = True, kinetic_map = False, reversible = False)
            #featurizer = self.featurize(lag = lag_featurize, dim = 2, commute_map = True, kinetic_map = False)
            trajs_micro = featurizer.fit_transform(self.trajs)
            print('Cumulative variance explained in the reduced space: ',featurizer.cumvar)
            cluster_micro = self.cluster_micro(trajs = trajs_micro)
            print('Number of dimensions: {0:d}'.format(cluster_micro.n_dims()))
            print('Clustering trajectories in reduced space using {0:d} microstates'.format(n_clusters))
            cluster_micro.fit(n_clusters = n_clusters)
            cluster_micro.run()
            #cluster_micro.show(pdf, plot_trajs = True, plot_maps = True)
            print('Number of samples per microstate: ',cluster_micro.n_samples())
            if np.any(cluster_micro.n_samples() == 0):
                print('WARNING: missing microclusters')
                initial_state = None
                continue
            mapping = cluster.get_mapping(cluster_micro)

            #--- Define markov model
            model = self.markov_model(cluster_micro, self.simulation.dt)
            if lag_compare > lag_mm:
                reference_lags = [lag_mm, lag_compare]
            else:
                reference_lags = [lag_compare, lag_mm]
            lag_dependency = model.timescales_lags(pdf, lags = lags_mm, reference_lags = reference_lags, n_timescales = n_timescales, reversible = False)
            if 'lag_dependency' not in self.convergence_data:
                self.convergence_data['lag_dependency'] = [lag_dependency]
            else:
                self.convergence_data['lag_dependency'].append(lag_dependency)
            model.estimate(lag = lag_mm, reversible = False)
            longest_timescale = model.timescales(1)
            if 'longest_timescale' not in self.convergence_data:
                self.convergence_data['longest_timescale'] = [longest_timescale]
            else:
                self.convergence_data['longest_timescale'].append(longest_timescale)
            #index_connected = np.append(model.M.connected_sets[0],-1)
            #cluster_micro.dtrajs = []
            #print 'model.M.count_matrix_active'
            #print model.M.count_matrix_active
            #print 'model.M.count_matrix_full'
            #print model.M.count_matrix_full
            #print cluster_micro.bin_centers
            #for dtraj in  model.M.dtrajs_active:
            #    cluster_micro.dtrajs.append(index_connected[dtraj])
            #cluster_micro.show(pdf, plot_trajs = True, plot_maps = True)

            #--- Test converge

            #--- Generate new starting state
            inactive_list = list(set(range(n_clusters)) - set(model.M.active_set))
            if inactive_list:
                print('Microstates excluded from the markov model: ',inactive_list)
                inactive_nsamples = cluster_micro.n_samples(list_bins = inactive_list)
                print('Number of samples in the most populated microstate: ',np.max(inactive_nsamples))
                #print 'Selecting most populated microstate among the ones excluded for next round'
                #micro_out = inactive_list[np.argmax(inactive_nsamples)] # choose the one with more points (it's closer to the connected ones ?)
                print('Selecting less populated microstate among the ones excluded for next round')
                micro_out = inactive_list[np.argmin(inactive_nsamples)] # choose the one with less points (it's closer to the barrier ?)
                #micro_out = inactive_list[np.random.randint(len(inactive_list))] # choose a random one
                print('Restarting from microstate: ',micro_out)
                initial_state = {}
                for i in range(self.n_dims()):
                    prob = mapping[i,:,micro_out].flatten()
                    #print 'Probability microstates: ',prob
                    initial_state[self.simulation.species[i]] = np.random.choice(np.round(cluster.bin_centers[:,i]).astype(int), p = prob)
            else:
                print('Selecting less populated microstate for next round')
                initial_state = {}
                micro_down = model.microstates_down_sampled(1, pdf = pdf) # this are the microstates with minimal number of samples in the reduced space, but then we need to map back to the real space
                #min_sample_bins = cluster.get_minima()
                for i in range(self.n_dims()):
                    prob = mapping[i,:,micro_down].flatten()
                    #prob = 1.0/prob
                    #prob[np.logical_not(np.isfinite(prob))] = 0.0
                    #prob /= np.sum(prob)
                    #print 'Probability microstates: ',prob
                    #initial_state[self.simulation.species[i]] = int(np.round(min_sample_bins[i]))
                    initial_state[self.simulation.species[i]] = np.random.choice(np.round(cluster.bin_centers[:,i]).astype(int), p = prob)
            initial_state = self.simulation.force_conservation_laws(initial_state)
        #self.show(pdf)
        f = plt.figure()
        ax = f.add_subplot(211)
        ax.plot(self.convergence_data['lag_dependency'])
        plt.ylabel('Percent change diff. lags')
        ax = f.add_subplot(212)
        ax.plot(self.convergence_data['longest_timescale'])
        plt.xlabel('N. iteration')
        plt.ylabel('Longest timescale')
        pdf.savefig()
        plt.close()
        print('Dependency of longest timescale on lagtime: ')
        for value in self.convergence_data['lag_dependency']:
            print('\t',value)
        print('Longest timescale over iterations: ')
        for value in self.convergence_data['longest_timescale']:
            print('\t',value[0])
        print('Total number of samples: ',cluster_micro.n_frames())
        return model, cluster_micro, cluster, mapping
