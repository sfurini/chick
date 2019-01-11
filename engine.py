#!/usr/bin/env python

import sys
import inspect
import datetime
import pickle
import itertools
import numpy as np
import scipy as sp
import numpy.ma as ma
import matplotlib
import cmath

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages
from math import *
from collections import *
from subprocess import call
from scipy import constants
from scipy import special
from scipy import interpolate
from scipy import io
from scipy.sparse import *
from scipy.sparse import linalg
from scipy.optimize import minimize
from scipy.optimize import newton_krylov
from scipy.optimize import anderson
from scipy.optimize import curve_fit
from scipy.optimize.nonlin import NoConvergence
from scipy.integrate import odeint

colors = ["blue", "green", "red", "cyan", "magenta", "yellow", "black"]

class Simulation(object):
    """
    General object for numerical simulations

    Attributes
    ----------
    X : ndarray
        <Number of steps> X <Number of dimensions>
        Trajectory
    V : ndarray
        <Number of steps> X <Number of dimensions>
        Velocity
    dt : float
        Time between steps 
    default_nsteps : int
        Default number of steps used in run if not explicitly defined otherwise
    default_total_time float
        Default length of the simulation, if not explicitly defined when calling self.run
    """

    def __init__(self, dt=1.0, nsteps=0, total_time=0):
        self.X = None  # trajectory = <number_of_steps> X <number_of_dimensions>
        self.V = None  # velocity = <number_of_steps> X <number_of_dimensions>
        self.dt = dt  # timestep
        self.default_nsteps = nsteps  # default number of steps for self.run
        self.default_total_time = total_time  # default lenght of the simulation for self.run
    def get_nsteps(self):
        """Return the number of steps"""
        return np.shape(self.X)[0]
    def get_ndims(self):
        """Return the number of dimensions"""
        if len(np.shape(self.X)) == 1:
            return 1
        else:
            return np.shape(self.X)[1]
    def get_distance_from_indexes(self, ind1, ind2):
        return np.linalg.norm(self.X[ind1, :] - self.X[ind2, :])
    def get_trajectories(self, n_sims=0):
        for i_traj in range(n_sims):
            self.run()
            yield self.X
    def get_mean_square_distance(self, lags, pdf=None, n_sims=0, box_length=0):
        """
        box_length  float   If different from zero, the trajectories are unwrapped
                            to remove periodic boundary conditions
        Return value:
            Average of the mean square displacemente for the last (higher) value of lags
            along the different dimensions
        """
        if isinstance(lags, list):
            lags = np.array(lags)
        elif isinstance(lags, int):
            lags = np.array([lags])
        msd = np.zeros((len(lags), self.get_ndims()))
        n_samples_msd = np.zeros(len(lags))
        for trajectory in self.get_trajectories(n_sims):
            if box_length:
                trajectory = np.unwrap((constants.pi * 2.0 / box_length) * trajectory, axis=0) / (
                        constants.pi * 2.0 / box_length)
            max_lag_trajectory = self.get_nsteps() - 1
            max_lag = min((lags[-1], max_lag_trajectory))
            for ind_lag, lag in enumerate(lags):
                if lag > max_lag:
                    break
                trajectory_lag = np.roll(trajectory, lag, axis=0)
                move = trajectory[lag:, :] - trajectory_lag[lag:, :]
                move_square = np.square(move)
                mean_move_square = np.sum(move_square, axis=0)
                msd[ind_lag, :] += mean_move_square
                n_samples_msd[ind_lag] += np.shape(move)[0]
        msd /= np.reshape(n_samples_msd, (len(lags), 1))
        if pdf is not None:
            f = plt.figure()
            ax = f.add_subplot(111)

            def linear_msd(x, d):
                return 2.0 * x * d

            for idim in range(np.shape(msd)[1]):
                popt, pcov = curve_fit(linear_msd, self.dt * lags, msd[:, idim], 1)
                ax.plot(self.dt * lags, msd[:, idim], '*', color=colors[idim % len(colors)], label=None)
                ax.plot(self.dt * np.arange(0.0, np.max(lags)),
                        linear_msd(self.dt * np.arange(0.0, np.max(lags)), popt), '-', color=colors[idim % len(colors)],
                        label=str(popt[0]))
                print('Diffusion coefficient along dimension {0:d} = {1:f}'.format(idim, popt[0]))
            plt.xlabel('Time')
            plt.ylabel('MSD')
            plt.xlim((0.0, self.dt * np.max(lags)))
            plt.legend()
            pdf.savefig()
        return np.mean(msd[-1])
    def unwrap(self, box_length=1.0):
        # Multiplication and division by 2.0*pi is needed because np.unwrap works in radians
        self.X = np.unwrap((constants.pi * 2.0 / box_length) * self.X, axis=0) / (constants.pi * 2.0 / box_length)
    def cast_to_box_and_split(self, box_size):
        """
        box_size    float
        Recast all the samples to a cubic box of size box_size
        and split the trajectory if crossing boundary condutions
        """
        return cast_to_box_and_split(self.X, box_size)
    def cast_to_box(self, box_size):
        """
        box_size    float
        Recast all the samples to a cubic box of size box_size
        """
        self.X = cast_to_box(self.X, box_size)
    def show(self, pdf):
        f = plt.figure()
        if self.V is not None:
            ax1 = f.add_subplot(221)
        else:
            ax1 = f.add_subplot(211)
        if self.get_ndims() == 1:
            ax1.plot(self.dt * np.arange(self.get_nsteps()), self.X, '-')
        else:
            for i in range(self.get_ndims()):
                ax1.plot(self.dt * np.arange(self.get_nsteps()), self.X[:, i], '-')
        plt.xlabel('Time')
        plt.ylabel('Reaction Coordinate')
        if self.V is not None:
            ax2 = f.add_subplot(223)
        else:
            ax2 = f.add_subplot(212)
        if self.get_ndims() == 1:
            h = np.histogram(self.X, 100, normed=True)
            ax2.plot(h[1][:-1], h[0])
        else:
            for i in range(self.get_ndims()):
                h = np.histogram(self.X[:, i], 100, normed=True)
                ax2.plot(h[1][:-1], h[0])
        plt.xlabel('Reaction Coordinate')
        plt.ylabel('Frequency')
        if self.V is not None:
            ax3 = f.add_subplot(222)
            if self.get_ndims() == 1:
                ax3.plot(self.dt * np.arange(self.get_nsteps()), self.V, '-')
            else:
                for i in range(self.get_ndims()):
                    ax3.plot(self.dt * np.arange(self.get_nsteps()), self.V[:, i], '-')
            plt.xlabel('Time')
            plt.ylabel('Velocity')
            ax4 = f.add_subplot(224)
            if (self.get_ndims() == 1):
                h = np.histogram(self.V, 100, normed=True)
                ax4.plot(h[1][:-1], h[0])
            else:
                for i in range(self.get_ndims()):
                    h = np.histogram(self.V[:, i], 100, normed=True)
                    ax4.plot(h[1][:-1], h[0])
            # Vthr = np.exp(-np.square(h[1][:-1])/(2.0*self.sqrt_kTOnMass*self.sqrt_kTOnMass)) \
            #		/ (sqrt(2*constants.pi)*self.sqrt_kTOnMass)
            # ax4.plot(h[1][:-1],Vthr,':r')
            plt.xlabel('Velocity [nm/fs]')
            plt.ylabel('Frequency')
        pdf.savefig()
        if (self.get_ndims() == 2):
            f = plt.figure()
            ax = f.add_subplot(111)
            ax.plot(self.X[:, 0], self.X[:, 1], '.')
            plt.xlabel('Reaction Coordinate 0')
            plt.ylabel('Reaction Coordiante 1')
            pdf.savefig()
        elif (self.get_ndims() == 3):
            f = plt.figure()
            ax = f.gca(projection='3d')
            ax.plot(self.X[:, 0], self.X[:, 1], self.X[:, 2], '.')
            pdf.savefig()

class Gillespie(Simulation):
    """
    
    Attributes
    ----------
    reactions	list of objects Reaction
    rates	list of floats
    species	list of string
        Name of the molecular species
    A   np.ndarray
        <number_of_reactions> X <number_of_species>
    	A[i,j] = Change in the number of molecules j when the reaction i occurs
    P   np.ndarray
        <number_of_reactions> X <number_of_species>
    	P[i,j] = Number of molecules j needed for the reaction i
    state np.array
        <number_of_species>
        The current state of the system
    rates np.array
        <number_of_reactions>
    volume_index    int
        Index of the reaction V -> V + V, if it exist (-1 otherwise)
    index_reactions_volume_biased : list
        Indexes of the reaction with rates modified by volume
    times   np.array
        Time steps of the simulated trajectory
        In this array the time steps are not saved with a constant period
    states  np.ndarrya
        <number_of_time_steps> X <number_of_molecular_species>
    	Evolution of the state in a simulated trajectory
        In this array the time steps are not saved with a constant period (the same of self.times)
        The attributes self.X provides the samples with the constant sampling self.dt
    rate_patches    dictionary
        key     index of the reaction
        value   function used for adjusting the rate
    conservation_laws: list
        values: dict
            key: specie + 'total'
            element: number of element of that specie in the conservation law
            element 'total': the value of this conservation law
        Example: [{'A':1,'B':2,'total':1}]
            it means that 1*A + 2*B = 1
    """
    def __init__(self, dt=1.0, nsteps=0, total_time=0):
        """
        Parameters
        ----------
        dt  float
            Time step for sampling the trajectory
        nsteps  int
            Default number of steps, if not explicitly defined when calling self.run
        total_time float
            Default length of the simulation, if not explicitly defined when calling self.run

        Example:
            gnr = cookies.simulation.Gillespie(dt = 10.0, nsteps = 1e5)
            gnr.add_reaction(cookies.simulation.Reaction('p_a_f  ->  p_a_f + a',g1))
            gnr.add_reaction(cookies.simulation.Reaction('p_a_b  ->  p_a_b + a',g0))
            gnr.add_reaction(cookies.simulation.Reaction('p_b_f  ->  p_b_f + b',g1))
            gnr.add_reaction(cookies.simulation.Reaction('p_b_a  ->  p_b_a + b',g0))
            gnr.add_reaction(cookies.simulation.Reaction('a ->',k))
            gnr.add_reaction(cookies.simulation.Reaction('b ->',k))
            gnr.add_reaction(cookies.simulation.Reaction('p_a_f  + 2*b ->  p_a_b',h))
            gnr.add_reaction(cookies.simulation.Reaction('p_a_b -> p_a_f  + 2*b',f))
            gnr.add_reaction(cookies.simulation.Reaction('p_b_f  + 2*a ->  p_b_a',h))
            gnr.add_reaction(cookies.simulation.Reaction('p_b_a -> p_b_f  + 2*a',f))
        """
        super(Gillespie, self).__init__(dt, nsteps, total_time)
        self.reactions = []
        self.species = []
        self.A = np.empty((self.n_reactions(), self.n_species()))
        self.P = np.empty((self.n_reactions(), self.n_species()))
        self.volume_index = -1
        self.index_reactions_volume_biased = []
        self.rates = np.empty(self.n_reactions())
        self.state = np.zeros(self.n_species(), dtype=np.int64)
        self.times = None
        self.states = None
        self.rate_patches = {}
        self.conservation_laws = []
    def n_reactions(self):
        return len(self.reactions)
    def n_species(self):
        return len(self.species)
    def add_reaction(self, reaction):
        self.reactions.append(reaction)
        self.update_species()
        self.update_state_change_matrix()
        self.update_stochiometry_matrix()
        if reaction.is_cell_growth():
            if self.volume_index > -1:
                raise ValueError('ERROR: double definition of cell growth reaction')
            self.volume_index = self.n_reactions() - 1
        if reaction.is_volume_biased():
            self.index_reactions_volume_biased.append(self.n_reactions() - 1)
        self.define_rates()
        self.state = np.zeros(self.n_species(), dtype=np.int64)  # initialize state to zero...
        if self.volume_index > -1:
            self.state[self.volume_index] = int(0.5 * self.reactions[
                self.volume_index].volume)  # ... apart from volume that is initialized to half of the maximum value
        if reaction.has_rate_patch():
            self.rate_patches[self.n_reactions() - 1] = reaction.rate_patch
        self.times = None
        self.states = None
    def add_conservation(self, conservation):
        fake_reaction = Reaction(conservation.replace('=', '->'), 1.0)
        fake_reaction.reagents['total'] = int(conservation.split('=')[1])
        self.conservation_laws.append(fake_reaction.reagents)
    def update_species(self):
        for reaction in self.reactions:  # cicle over all the reactions
            for specie in reaction.species:  # cicle over all the species involved in this reaction
                if specie not in self.species:
                    self.species.append(specie)  # add the specie to the set
    def update_state_change_matrix(self):
        self.A = np.zeros((self.n_reactions(), self.n_species())).astype(int)
        for i_reaction, reaction in enumerate(self.reactions):  # cicle over all the reactions
            for specie, change in reaction.state_change.items():  # cicle over all the specie:change couples for this reaction
                i_specie = self.species.index(specie)  # Find the index of the specie
                self.A[i_reaction, i_specie] = change
    def update_stochiometry_matrix(self):
        self.P = np.zeros((self.n_reactions(), self.n_species())).astype(int)
        for i_reaction, reaction in enumerate(self.reactions):  # cicle over all the reactions
            for specie, num in reaction.reagents.items():  # cicle over all the specie:change couples for this reaction
                i_specie = self.species.index(specie)  # Find the index of the specie
                self.P[i_reaction, i_specie] = num
    def define_rates(self):
        self.rates = np.ones(self.n_reactions())
        for i, reaction in enumerate(self.reactions):
            self.rates[i] = reaction.rate
    def update_rates(self):
        for i in self.index_reactions_volume_biased:
            self.rates[i] = self.reactions[i].rate
            volume_ratio = 1.0 * self.state[self.volume_index] / self.reactions[
                self.volume_index].volume  # (current volume) / (maximum volume)
            if self.reactions[i].is_volume_proportional():
                self.rates[i] *= volume_ratio
            else:
                self.rates[i] /= volume_ratio
    def convert_dict_state_to_array(self, state_dict):
        state = np.empty(len(state_dict))
        for key, value in state_dict.items():
            i_specie = self.species.index(key)  # Find the index of the specie
            state[i_specie] = value
        return state
    def convert_state_array_to_dict(self, state=None, cast=True):
        if state is None:
            state = self.state
        dict_state = {}
        for i_specie, specie in enumerate(self.species):
            if cast:
                dict_state[specie] = int(round(state[i_specie]))
            else:
                dict_state[specie] = state[i_specie]
        return dict_state
    def check_conservation_laws(self, state=None):
        if state is None:
            state = self.state
        for conservation_law in self.conservation_laws:
            number_molecules = 0
            for i_specie, specie in enumerate(self.species):
                if isinstance(state, dict):
                    number_molecules += conservation_law.get(specie, 0) * state[specie]
                else:
                    number_molecules += conservation_law.get(specie, 0) * state[i_specie]
            if number_molecules != conservation_law['total']:
                return False
        return True
    def force_conservation_laws(self, state):
        """
        Return a state that obeys to the conservation laws
        Only the last specie involved in the conservation law is changed
        Thus negative values might appear !!

        Parameters
        ----------
        state: dict
        """
        for conservation_law in self.conservation_laws:
            number_molecules = 0
            species = conservation_law.keys()
            species.remove('total')
            for specie in species[:-1]:
                number_molecules += state[specie]
            state[species[-1]] = conservation_law['total'] - number_molecules
        return state
    def get_propensity(self, state=None):
        """
        Example:
            Species :
                M          =          2
                P          =          3
            Transition matrix :
                                  ->  M               M ->             M -> M + P             P ->
                M                  1                   -1                  0                   0
                P                  0                   0                   1                   -1
                rate            1.000000            2.000000            3.000000            4.0000
                state
                    [2
                     3]
                P
                    [[0 0]
                     [1 0]
                     [1 0]
                     [0 1]
                    ]
            number_of_each_reagent
                [[1 1]	First reaction -> M: No reagent
                 [2 1]  Second reaction M -> : M is the reagent
                 [2 1]  Third reaction M -> M + P:  M is the reagent
                 [1 3]] Forth reaction P ->: P is the reagent
            number_of_reagents
                [1  First reaction -> M: No reagent
                 2  Second reaction M -> : M is the reagent, so 2 molecules
                 2  Third reaction M -> M + P:  M is the reagent, so 2 molecules
                 3] Forth reaction P ->: P is the reagent so 3 molecules
            number_of_reagents * self.rates
                [1	This is rate_1(1) * 1 (that's why we need a 1 in number_of_reagents for a constant rate reaction)
                 4      This is rate_2(2) * M(2)
                 6      This is rate_3(3) * M(2)
                 12]    This is rate_4(4) * P(3)

        """
        if state is None:
            state = self.state
        number_of_each_reagents = np.power(state, self.P)
        # print 'state: ',state
        # print 'self.P: ',self.P
        # print 'number_of_each_reagents(before correction): ',number_of_each_reagents
        inds_row, inds_col = np.where(self.P > 1)
        # This correction is needed for reactions that use more than one molecule of the same kind
        for ind_row, ind_col in zip(inds_row, inds_col):
            # self.P[ind_row,ind_col] = Number of molecules needed for the reaction
            # state[ind_col] = Number of molecules that are present
            # e.g. A + 3*B -> C
            # with number_of_each_reagents = np.power(state,self.P) the contributio of B is B**3
            # here we correct to binomial_coefficient(B,3)
            number_of_each_reagents[ind_row, ind_col] = special.binom(state[ind_col], self.P[ind_row, ind_col])
        # print 'number_of_each_reagents(after correction): ',number_of_each_reagents
        number_of_reagents = np.prod(number_of_each_reagents, axis=1)
        # print 'number_of_reagents: ',number_of_reagents
        propensities = number_of_reagents * self.rates
        for i_reaction, rate_patch in self.rate_patches.items():
            input_values = [state[self.species.index(specie)] for specie in inspect.getargspec(rate_patch).args]
            propensities[i_reaction] *= rate_patch(*input_values)
        # print 'propensities: ',propensities
        return propensities
    def get_rates(self, state=None, t=None):
        if state is None:
            state = self.state
        return np.dot(np.transpose(self.A), self.get_propensity(state))
    def get_rate_matrix(self, states):
        print('Computing rate matrix')
        #print(self.species)
        #dstates = states[self.species[0]]
        #for i_specie in range(1,len(self.species)):
        #    specie = self.species[i_specie]
        #    print(dstates)
        #    print(states[specie])
        #    print(np.array(np.meshgrid(dstates,states[specie])).T.reshape(-1,2))
        #    exit()
        indep_species = deepcopy(self.species)
        for conservation_law in self.conservation_laws:
            for specie in conservation_law.keys():
                if specie != 'total':
                    indep_species.remove(specie)
                    break
            else:
                print('ERROR')
        enum_states = np.array(np.meshgrid(*[states[specie] for specie in indep_species])).T.reshape(-1,len(indep_species))
        print('Number of enumerated states: ',enum_states.shape)
        enum_states_full = np.nan*np.ones((enum_states.shape[0],len(self.species)))
        for i_specie, specie in enumerate(self.species):
            if specie in indep_species:
                i_indep_species = indep_species.index(specie)
                enum_states_full[:,i_specie] = enum_states[:,i_indep_species]
            else:
                for conservation_law in self.conservation_laws:
                    if specie in conservation_law.keys():
                        enum_states_full[:,i_specie] = conservation_law['total']
                        for specie_j in conservation_law.keys():
                            if (specie_j != 'total') and (specie_j != specie):
                                j_indep_species = indep_species.index(specie_j)
                                enum_states_full[:,i_specie] -= enum_states[:,j_indep_species]
        print('Enumerated states: ',enum_states_full)
        K = np.zeros((enum_states_full.shape[0],enum_states_full.shape[0]))
        for i in range(enum_states_full.shape[0]):
            state_i = enum_states_full[i,:]
            print('Searching connection for state {0:d}'.format(i))
            for j in range(enum_states_full.shape[0]):
                state_j = enum_states_full[j,:]
                diff = (state_j - state_i).astype(int)
                for i_reaction in range(self.A.shape[0]):
                    if np.all(self.A[i_reaction,:] == diff):
                        number_of_each_reagents = np.power(state_i, self.P[i_reaction,:])
                        #print('\tadding component {0:d} {1:d}'.format(i,j))
                        for ind_col in np.where(self.P[i_reaction,:] > 1):
                            number_of_each_reagents[ind_col] = special.binom(state_i[ind_col], self.P[i_reaction, ind_col])
                        number_of_reagents = np.prod(number_of_each_reagents)
                        #print('species ',self.species)
                        #print('state_i ',state_i)
                        #print('state_j ',state_j)
                        #print('P ',self.P[i_reaction,:])
                        #print('A ',self.A[i_reaction,:])
                        #print('number_of_each_reagents ',number_of_each_reagents)
                        #print('number_of_reagents ',number_of_reagents)
                        K[j,i] += number_of_reagents * self.rates[i_reaction]
        for i in range(enum_states_full.shape[0]):
            K[i,i] = -np.sum(K[:,i])
        print('Rate matrix: ',K)
        T = sp.linalg.expm(K)
        print('Transition matrix: ',T)
        print(np.sum(T,axis = 0))
        eigv = np.linalg.eigvals(T)
        inds = np.argsort(np.abs(eigv))[-1::-1]
        print('eigvs = ',eigv[inds])
        output = ''
        for i in inds:
            eig = eigv[i]
            output += '\tEigenvalue = {0:f}\n'.format(eig)
            output += '\tTimescale = {0:f}\n'.format(-1.0/np.log(np.abs(eig)))
            if eig.imag != 0:
                r, phi = cmath.polar(eig)
                output += '\t\tperiod = {0:f}\n'.format(2*np.pi/np.abs(phi))
        print(output)
        exit()
    def get_independent_rates(self, state=None):
        inds, species = self.get_independent_species()
        full_state = {}
        for i, i_specie in enumerate(inds):
            full_state[self.species[i_specie]] = state[i]
        for specie in self.species:
            if specie not in full_state:
                full_state[specie] = 0
        # print 'full_state 0: ',full_state
        full_state = self.force_conservation_laws(full_state)
        # print 'full_state 1: ',full_state
        state = self.convert_dict_state_to_array(full_state)
        # print 'self.species: ',self.species
        # print 'state: ',state
        # print 'inds: ',inds
        return self.get_rates(state)[inds]

    def get_independent_species(self):
        independent_species = deepcopy(self.species)
        for conservation_law in self.conservation_laws:
            for specie in conservation_law.keys():
                if specie in independent_species:
                    independent_species.remove(specie)
                    break
            else:
                raise ValueError('ERROR: too many conservation laws')
        i_independent_species = [i_specie for i_specie, specie in enumerate(self.species) if
                                 specie in independent_species]
        return i_independent_species, independent_species

    def get_independent_state_array(self, state_dict):
        state = []
        inds, species = self.get_independent_species()
        # print 'indep. species: ',species
        for specie in species:
            state.append(state_dict[specie])
        return state
    def find_equilibrium(self, initial_state=None):
        """
        Parameters
        ----------
        initial_state: dict
        """
        initial_state = self.get_independent_state_array(initial_state)
        print('initial_state: ', initial_state)
        try:
            # root = newton_krylov(self.get_independent_rates, initial_state, method='lgmres', verbose=1)
            root = anderson(self.get_independent_rates, initial_state, verbose=1)
        except NoConvergence as e:
            root = e.args[0]
        inds, species = self.get_independent_species()
        root_dict = {}
        for i, ind in enumerate(inds):
            root_dict[species[i]] = int(round(root[i]))
        return root_dict
    def is_incomplete(self):
        # Check if there are biased reaction and the volume is not defined --> error
        if sum([reaction.is_volume_biased() for reaction in self.reactions]) and self.volume_index == -1:
            return True
        return False
    def get_random_state(self):
        if self.states is not None:
            nsteps_states = np.shape(self.states)[0]
            nsteps_X = self.get_nsteps()
            if nsteps_X > nsteps_states:
                half_nsteps_X = int(0.5 * nsteps_X)
                ind_random_frame = np.random.randint(half_nsteps_X, nsteps_X)
                state = self.X[ind_random_frame, :]
            else:
                half_nsteps_X = int(0.5 * nsteps_states)
                ind_random_frame = np.random.randint(half_nsteps_X, nsteps_states)
                state = self.states[ind_random_frame, :]
            return state
        else:
            print('ERROR: it is not possible to start a simulation from a random state if no previous simulation was run')
            exit()
    def run(self, initial_state=None, total_time=None, nsteps=None, stride=np.inf):
        """
        Simulate trajectory with the Gillespie algorithm

        Parameters
        ----------
        initial_state : dictionary
            key: name of the molecular species
            value: initial condition
            If None choose a random state from the last simulation otherwise
                if self.states is None (no previous simulation was run)
                it generates and error
        total_time : float
            Stop after this time interval
        nsteps : int
            Stop after this number of reactions
        stride : float
            Save time and states with temporal frequence > stride
        """
        if self.is_incomplete():
            raise ValueError('ERROR: wrong definition of the model')
        if initial_state is None:
            self.state = self.get_random_state()
        elif type(initial_state) == dict:
            self.state = self.convert_dict_state_to_array(initial_state)
        elif type(initial_state) == np.ndarray:
            self.state = initial_state
        else:
            raise ValueError('ERROR: wrong definition of initial state')
        if nsteps is None:
            nsteps = self.default_nsteps
        if total_time is None:
            total_time = self.default_total_time
        i_step = 0
        time = 0.0
        times = [time]
        states = [list(self.state)]
        last_state = list(self.state)
        states_uniform_sampling = [list(self.state)]
        time_last_saved_uniform_sampling = time
        print("Running stochastic simulation with the Gillespie's algorithm starting from: ",self.convert_state_array_to_dict(self.state))
        while (time < total_time) and (i_step < nsteps):
            # print 'i_step = {0:d} time = {1:f}'.format(i_step,time)
            if (time - times[-1]) > stride:
                times.append(time)
                states.append(list(self.state))
            n_periods = int((time - time_last_saved_uniform_sampling) / self.dt)
            # print 'n_periods = {0:d} dt = {1:f}'.format(n_periods, self.dt)
            for i_period in range(n_periods):
                states_uniform_sampling.append(last_state)
                time_last_saved_uniform_sampling += self.dt
            # Check cell division
            if self.volume_index > -1:  # cell growth reaction was included in the model
                if (last_state[self.volume_index] >= self.reactions[self.volume_index].volume):  # volume above threshold
                    for i in range(self.n_species()):
                        if self.state[i] == 0:
                            self.state[i] = 0
                        else:
                            self.state[i] = np.random.binomial(self.state[i], 0.5)
            r1, r2 = np.random.random_sample(2)
            self.update_rates()
            a = self.get_propensity()
            a_cum = np.cumsum(a)
            if a_cum[-1] == 0:  # stop the simulation: the system will never go away from this state
                break
            dt = -(1.0 / a_cum[-1]) * np.log(r1)  # time for the next transition
            for i_reaction in range(len(self.reactions)):  # Cicle over all the possible reactions
                if a_cum[i_reaction] > r2 * a_cum[-1]:  # Do this transition
                    # print "r1 = ",r1," r2 = ",r2
                    # print "a = ",a
                    # print "np.cumsum(a) = ",a_cum
                    # print "dt = ",dt
                    # print "time = ",dt
                    # print "state = ",self.state
                    # print "i_reaction = ",i_reaction
                    # print "reaction = ",self.reactions[i_reaction].reaction
                    self.state += self.A[i_reaction, :]  # Do reaction i_reaction
                    last_state = list(self.state)
                    time += dt
                    i_step += 1
                    break  # stop the cycle: reaction done
        self.times = np.array(times)
        self.states = np.array(states)
        self.X = np.array(states_uniform_sampling)
        print('Total time = ', time)
        print('Total steps = ', i_step)
        print('Total number of samples = ', self.X.shape[0])
    def get_mean(self, specie, time_start=0):
        i_time = int(time_start / self.dt)
        i_specie = self.species.index(specie)  # Find the index of the specie
        return np.mean(self.X[i_time:, i_specie])
    def get_variance(self, specie, time_start=0):
        i_time = int(time_start / self.dt)
        i_specie = self.species.index(specie)  # Find the index of the specie
        return np.var(self.X[i_time:, i_specie])
    def get_standard_deviation(self, specie, time_start=0):
        return np.sqrt(self.get_variance(specie, time_start))
    def get_trajectory_specie(self, specie):
        """
        Return the simulated trajectory of the specie
        """
        i_specie = self.species.index(specie)
        return self.X[:, i_specie]
    def get_statistics(self, time_start=0):
        """
        Return
        ------
        dictionary
            keys:   name of the species
            values: dictionary
                keys:   mean / std
                values: average / standard deviation
        """
        stats = {}
        for i_specie, specie in enumerate(self.species):
            stats[specie] = {'mean': self.get_mean(specie, time_start),
                             'std': self.get_standard_deviation(specie, time_start)}
        return stats
    def print_statistics(self, time_start=0):
        for specie, stat in self.get_statistics(time_start).items():
            print('{0:s}\t{1:8.3f}\t{2:8.3f}'.format(specie, stat['mean'], stat['std']))
    def show(self, pdf_stream, means={}, stds={}, one_by_one=True, species=None):
        """
        Parameters
        ----------
        means   dictionary
            key:    specie
            value:  reference mean
        std   dictionary
            key:    specie
            value:  reference standard deviation
        one_by_one: bool
            If True make one plot for each state variable
        species: None/list
            List of string with the species to show
        """
        if (self.states is None) and (self.X is None):
            print('ERROR: first run a simulation then plot it !')
            raise ValueError
        if species is None:
            ind_species = range(self.n_species())
        else:
            ind_species = []
            for specie in species:
                ind_species.append(self.species.index(specie))
        if one_by_one:
            for i_specie in ind_species:
                specie = self.species[i_specie]
                f = plt.figure()
                axis = f.add_subplot(111)
                axis.plot(self.times, self.states[:, i_specie])
                if self.X is not None:
                    axis.plot(self.dt * np.arange(self.get_nsteps()), self.X[:, i_specie], '-')
                if specie in means.keys():
                    axis.plot([0.0, self.dt * self.get_nsteps()], [means[specie], means[specie]], '-k')
                if specie in stds.keys():
                    axis.plot([0.0, self.dt * self.get_nsteps()],
                              [means[specie] - stds[specie], means[specie] - stds[specie]], ':k')
                    axis.plot([0.0, self.dt * self.get_nsteps()],
                              [means[specie] + stds[specie], means[specie] + stds[specie]], ':k')
                plt.xlabel('Time [s]')
                plt.ylabel(specie + ' [#]')
                pdf_stream.savefig()
                plt.close()
        f = plt.figure()
        axis = f.add_subplot(111)
        for i_specie in ind_species:
            specie = self.species[i_specie]
            if self.X is not None:
                axis.plot(self.dt * np.arange(self.get_nsteps()), self.X[:, i_specie], '-',
                          label=self.species[i_specie])
        #plt.ylim([0,80])
        #plt.xlim([0,5000])
        #plt.grid()
        plt.legend()
        pdf_stream.savefig()
        plt.close()
    def __str__(self):
        output = "CELL MODEL\n"
        output += "Number of molecular species = {0:d}\n".format(len(self.species))
        output += "Species : \n"
        for i_specie, specie in enumerate(self.species):
            output += "\t{0:10s} = {1:10d}\n".format(specie, self.state[i_specie])
        output += "Transition matrix : \n"
        output += "{0:<10s}".format('')
        for i_reaction, reaction in enumerate(self.reactions):
            output += "{0:^20s}".format(reaction.reaction)
        output += "\n"
        for i_specie, specie in enumerate(self.species):
            output += "{0:<10s}".format(specie)
            for i_reaction, reaction in enumerate(self.reactions):
                output += "{0:^20d}".format(self.A[i_reaction, i_specie])
            output += "\n"
        output += "{0:<10s}".format('rate')
        for i_reaction, reaction in enumerate(self.reactions):
            output += "{0:^20f}".format(self.rates[i_reaction])
        for i_reaction, reaction in enumerate(self.reactions):
            output += "\n"
            output += "REACTION {0:d}\n".format(i_reaction)
            output += reaction.__str__()
        for conservation_law in self.conservation_laws:
            output += '\n' + 'CONSERVATION LAW: ' + str(conservation_law)
        return output

    def run_deterministic(self, initial_state=None, total_time=None, nsteps=None, stride=np.inf):
        if self.is_incomplete():
            raise ValueError('ERROR: wrong definition of the model')
        if initial_state is None:
            self.state = self.get_random_state()
        elif type(initial_state) == dict:
            self.state = self.convert_dict_state_to_array(initial_state)
        elif type(initial_state) == np.ndarray:
            self.state = initial_state
        else:
            raise ValueError('ERROR: wrong definition of initial state')
        if nsteps is None:
            nsteps = self.default_nsteps
        if total_time is None:
            total_time = self.default_total_time
        print('Running numerical integration from: ', self.state)
        self.times = np.arange(0, total_time, self.dt)
        self.X = odeint(self.get_rates, self.state, self.times)  # , hmax = 1e-1*self.dt)
        self.states = self.X

class Trajectory(Simulation):
    """
    Generate deterministic trajectories

    Attributes
    ----------
    species	list of string
        Name of the species
        If None species are named 'A', 'B', 'C', etc
    rules   list
        elements: functions to be used to generate the trajectories for each specie
            Functions should take two arguments: time, state at previos time
            and return one float value: the value of the specie at the requested time
            
            Example:
                def rule_1(t, x):
                    if t < 10.0:
                        return 0.0
                    else:
                        return 1.0
    """
    def __init__(self, dt=1.0, nsteps=0, rules=[], species=[]):
        """
        Parameters
        ----------
        nsteps int
            Default number of steps
        rules list of functions
            The rules used to generate trajectories
        species list of strings
            The name of the species
        """
        super(Trajectory, self).__init__(dt, nsteps)
        self.rules = rules
        self.species = species
        if len(self.species) == 0:
            for i_rule, rule in enumerate(self.rules):
                self.species.append(chr(65 + i_rule))
        if len(self.species) != len(self.rules):
            raise ValueError('ERROR: rules and species must have the same length')
    def n_species(self):
        return len(self.species)
    def run(self, nsteps=None):
        """
        Parameters
        ----------
        nsteps int
            Number of steps to generate
        """
        if nsteps is None:
            nsteps = self.default_nsteps
        self.X = np.empty((int(nsteps), self.n_species()))
        for i_specie in range(self.n_species()):
            self.X[0, i_specie] = 0
            for i_t in np.arange(1, nsteps).astype(int):
                self.X[i_t, i_specie] = self.rules[i_specie](self.dt * i_t, self.X[i_t - 1, i_specie])

class Reaction(object):
    """
    Class for chemical reactions

    Attributes
    ----------
    reaction : string
    	Textual description of the reaction
    rate : float
    species : set
    	Names of the molecular species involved in the reaction
    state_change : dictionary
    	Changes when the reaction happens
    		key: name of the molecular specie
    		value: change in the number of molecules for this specie
    reagents : dictionary
    	Necessary reagents for the reaction
    		key: name of the molecular specie
    		value: how many molecule of this specie are needed
    	Example:
    		-> M: reagents = {}
    		A + 2*B -> C: reagents = {'A':+1,'B':+2}
    		M -> M + P: reagents = {'M':+1}
    			This kind of reactions (number of reagents that don't change)
    			is the reason why self.reagents is needed
    volume : str / int
        str --> Coefficient representing rate dependency upon cellular volume:
            none	No dependency [DEFAULT]
            V	Directly proportional to volume
            1/V	Inversely proportional to volume
        int --> Maximum possible value for the volume
            This is used only for reaction V + V -> V, representing cell grow
            In case of rection V + V -> V, the parameter volume is compulsory
    rate_patch  function
        Function used to correct the rate with respect to the standard CME rates
        If None no patch function is applied 
        If different from None the attribute self.rate is ignored
        Example: auto-repression of transcription 
            def cooperative_repression(P):
                return k**n / (k**n + P**n)
            Reaction('M -> M + P',rate_patch = cooperative_repression)
    """

    def __init__(self, reaction, rate=1.0, volume='none', rate_patch=None):
        """
        Examples:
            Reaction('-> A', 1.0)	Constant rate synthesis
            Reaction('A ->', 1.0)	Degradation
            Reaction('A + B -> C', 1.0)
            Reaction('A + 2*B -> 3*C + 4*D', 1.0)
        """
        self.reaction = reaction
        self.rate = rate
        self.species = set()
        self.state_change = {}
        self.reagents = {}
        index = reaction.find('->')
        if index == -1:
            raise ValueError("ERROR: symbol '->' missing in reaction definition")
        reagents = reaction[0:index].strip()
        for num, specie in self.split_species_stochiometry(reagents):
            self.species.add(specie)
            self.state_change[specie] = -num
            self.reagents[specie] = num
        products = reaction[index + 2:].strip()
        for num, specie in self.split_species_stochiometry(products):
            self.species.add(specie)
            try:
                self.state_change[specie] += num
                if self.state_change[specie] == 0:
                    del self.state_change[specie]
            except:
                self.state_change[specie] = num
        if isinstance(volume, str):
            if volume not in ['none', 'V', '1/V']:
                raise ValueError('ERROR: wrong value for volume')
        elif self.is_cell_growth():
            if not isinstance(volume, int):
                raise ValueError('ERROR: use parameter volume to define max volume for reaction V -> V + V')
        self.volume = volume
        self.rate_patch = rate_patch
    def is_volume_biased(self):
        return self.is_volume_proportional() or self.is_volume_inverse_proportional()
    def is_volume_proportional(self):
        return self.volume == 'V'
    def is_volume_inverse_proportional(self):
        return self.volume == '1/V'
    def is_cell_growth(self):
        return (self.species == set('V')) and (self.reagents == {'V': 1}) and (self.state_change == {'V': 1})
    def has_rate_patch(self):
        return (self.rate_patch is not None)
    def split_species_stochiometry(self, species):
        """
        """
        list_terms = []
        for i_term, term in enumerate(species.split('+')):
            if len(term) > 0:
                if '*' in term:
                    num_molecule = int(term.split('*')[0])
                    specie = term.split('*')[1].strip()
                else:
                    num_molecule = 1
                    specie = term.strip()
                list_terms.append((num_molecule, specie))
        return list_terms
    def __str__(self):
        output = "CHEMICAL REACTION: " + self.reaction + "\n"
        output += "rate = {0:8.3f}\n".format(self.rate)
        if self.has_rate_patch():
            output += 'rate_patch = \n'
            output += inspect.getsource(self.rate_patch)
        output += "state change ="
        for specie, num in self.state_change.items():
            output += "\n"
            output += "\t{0:s}:{1:d}".format(specie, num)
        return output

if __name__ == '__main__':
    print('-----------------')
    print('Testing Gillespie')
    print('-----------------')
    pdf = PdfPages('./test_simulation.pdf')

    # --- Transcription + Translation
    k1 = 100.0
    k2 = 10.0
    k3 = 1.0
    k4 = 0.1
    gnr = Gillespie(dt = 1.0, nsteps = 1e4)
    gnr.add_reaction(Reaction('  ->  M',k1))
    gnr.add_reaction(Reaction('M ->',k2))
    gnr.add_reaction(Reaction('M -> M + P',k3))
    gnr.add_reaction(Reaction('P ->',k4))
    print(gnr)
    gnr.run(initial_state = {'M':0,'P':0}, stride = 0)
    gnr.show(pdf, means = {'M':k1/k2,'P':k1*k3/(k2*k4)}, stds = {'M':np.sqrt(k1/k2),'P':np.sqrt( k1*k3*( 1 + (k3/k2) / (1+(k4/k2)) )/(k2*k4))})
    gnr.run()
    gnr.show(pdf, means = {'M':k1/k2,'P':k1*k3/(k2*k4)}, stds = {'M':np.sqrt(k1/k2),'P':np.sqrt( k1*k3*( 1 + (k3/k2) / (1+(k4/k2)) )/(k2*k4))})
    gnr.run_deterministic(initial_state = {'M':0,'P':0}, total_time = 40.0)
    print(gnr.convert_state_array_to_dict(gnr.X[-1,:]))
    gnr.show(pdf, means = {'M':k1/k2,'P':k1*k3/(k2*k4)})

    ##--- Regulated Transcription + Translation
    # k5 = 20.0
    # n = 1.0
    # gnr = Gillespie(dt = 1.0, nsteps = 1e5)
    # def regulation(P):
    #    return k1 / (1.0 + (P/k5)**n)
    # gnr.add_reaction(Reaction('  ->  M',rate_patch = regulation))
    # gnr.add_reaction(Reaction('M ->',k2))
    # gnr.add_reaction(Reaction('M -> M + P',k3))
    # gnr.add_reaction(Reaction('P ->',k4))
    # print gnr
    # peq = 0.5*k5*(np.sqrt(1 + 4*k1*k3/(k2*k4*k5))-1)
    # meq = k4*peq/k3
    # gnr.run(initial_state = {'M':2,'P':3}, stride = 0)
    # gnr.show(pdf, means = {'M':meq,'P':peq})
    # gnr.run()
    # gnr.show(pdf, means = {'M':meq,'P':peq})

    # gnr.add_reaction(Reaction('V ->  V + V',1e-2, volume = 50))
    # gnr.add_reaction(Reaction('  ->  M',100.0)) #,volume = 'V'))
    # gnr.add_reaction(Reaction('M ->',10.0))
    # gnr.add_reaction(Reaction('M -> M + P',1.0,rate_patch = regulation)) #,volume = '1/V'))
    # gnr.add_reaction(Reaction('P ->',0.1))
    pdf.close()
