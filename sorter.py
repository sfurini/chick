#!/usr/bin/env python

import cmath
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import vampnet
from vampnet import data_generator
from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Input, BatchNormalization, concatenate
from keras import optimizers
from keras.utils import plot_model
import tensorflow as tf
from keras.backend import clear_session

numerical_precision = 1e-10

import sys
sys.path.append("/home/simone")
sys.path.append("/home/simone/MySoftware")
sys.path.append("/Users/simone/Desktop")

import chick

class Neural(object):
    def __init__(self, replica, nodes, n_outputs, lags, epsilon = 1e-10):
        """
        trajs  list
            Each element is an np.ndarray with shape <number_samples> x <number_dimensions>
        nodes list
            Elements: number of nodes in each layer
        n_outputs   int
            Number of output nodes
        lag list
            Lag time
        epsilon float
            Eigenvalues below this threshold are considered zero
        """
        self.lags = lags
        self.trajs = replica.trajs
        self.traj_ord = np.empty(shape = (0, self.n_dims()))
        self.traj_ord_lag = np.empty(shape = (0, self.n_dims()))
        for traj in self.trajs:
            if len(traj.shape) == 1:
                traj = np.expand_dims(traj, 1)
            self.traj_ord = np.vstack((self.traj_ord, traj[:-self.lags[0],:])).astype('float32')
            self.traj_ord_lag = np.vstack((self.traj_ord_lag, traj[self.lags[0]:,:])).astype('float32')
        #--- Initialize the network
        self.vamp = vampnet.VampnetTools(epsilon = epsilon)
        self.losses = [
            #self.vamp._loss_VAMP_sym,
            #self.vamp.loss_VAMP,
            self.vamp.loss_VAMP2,
        ]
        self.n_outputs = n_outputs
        #--- Define the neural network
        self.nodes = nodes
        self.model = self.define_model()
        #plot_model(self.model, to_file='model.png', show_shapes = True)
    def define_model(self):
        Data_X = Input(shape = (self.n_dims(),))
        Data_Y = Input(shape = (self.n_dims(),))
        bn_layer = BatchNormalization() # alternative: bn_layer = Activation('linear')
        dense_layers = [Dense(node, activation = 'relu',) for node in self.nodes]
        lx_branch = bn_layer(Data_X)
        rx_branch = bn_layer(Data_Y)
        for i, layer in enumerate(dense_layers):
            lx_branch = dense_layers[i](lx_branch)
            rx_branch = dense_layers[i](rx_branch)
        softmax = Dense(self.n_outputs, activation='softmax')
        lx_branch = softmax(lx_branch)
        rx_branch = softmax(rx_branch)
        merged = concatenate([lx_branch, rx_branch])
        return Model(inputs = [Data_X, Data_Y], outputs = merged)
    def n_dims_traj(self, traj):
        """Return the dimensionality of a trajectory"""
        if len(np.shape(traj)) == 1:
            return 1
        else:
            return np.shape(traj)[1]
    def n_dims(self):
        """Return the dimensionality of the list of trajs"""
        list_n_dims = [self.n_dims_traj(traj) for traj in self.trajs]
        if all(list_n_dims[0] == other for other in list_n_dims):
            return list_n_dims[0]
        raise ValueError('ERROR: inconsistent dimensions')
    def shuffle_trajs(self, train_ratio):
        """
        """
        n_samples = self.traj_ord.shape[0]
        indexes = np.arange(n_samples)
        np.random.shuffle(indexes)
        traj = self.traj_ord[indexes] # shuffled trajectory
        traj_lag = self.traj_ord_lag[indexes] # shuffled lagged trajectory
        length_train = int(np.floor(n_samples * train_ratio)) # number of samples used for training
        length_vali = n_samples - length_train # numero of samples used for validation
        traj_data_train = traj[:length_train] # training trajectory
        traj_data_train_lag = traj_lag[:length_train] # lagged training trajectory
        traj_data_valid = traj[length_train:] # validation trajecotry
        traj_data_valid_lag = traj_lag[length_train:] # lagged validation trajectory
        X_now_train = traj_data_train.astype('float32') # training trajecotry
        X_lag_train  = traj_data_train_lag.astype('float32') # lagged training trajectory
        X_now_vali = traj_data_valid.astype('float32') # validation trajectory
        X_lag_vali = traj_data_valid_lag.astype('float32') # lagged validation trajectory
        return X_now_train, X_lag_train, X_now_vali, X_lag_vali
    def train(self, X_now_train, X_lag_train, X_now_vali, X_lag_vali, batch_size, n_epochs, learning_rate, pdf):
        """
        X_now/lag_train/vali    np.ndarray
            <number of samples> x <number of dimensions>
        """
        Y_train = np.zeros((X_now_train.shape[0],2*self.n_outputs)).astype('float32') # dummy variable ?
        Y_vali = np.zeros((X_now_vali.shape[0],2*self.n_outputs)).astype('float32') # dummy variable ?
        valid_metric = np.zeros((len(self.losses), n_epochs))
        train_metric = np.zeros((len(self.losses), n_epochs))
        for l_index, loss in enumerate(self.losses):
            #adam = optimizers.adam(lr = learning_rate/10)
            #self.model.compile(optimizer = adam, loss = loss, metrics = [self.vamp.metric_VAMP2])
            self.model.compile(optimizer = 'adam', loss = loss, metrics = [self.vamp.metric_VAMP2])
            hist = self.model.fit([X_now_train, X_lag_train], Y_train, batch_size = batch_size, epochs = n_epochs, verbose = 1,
                    validation_data=([X_now_vali, X_lag_vali], Y_vali))
            valid_metric[l_index] = np.array(hist.history['val_metric_VAMP2'])
            train_metric[l_index] = np.array(hist.history['metric_VAMP2'])
            score_train = self.model.evaluate([X_now_train, X_lag_train], Y_train, batch_size = batch_size)
            score_vali = self.model.evaluate([X_now_vali, X_lag_vali], Y_vali, batch_size = batch_size)
            print('score_train = ',score_train)
            print('score_vali = ',score_vali)
        valid_metric = np.reshape(valid_metric, (-1))
        train_metric = np.reshape(train_metric, (-1))
        if pdf is not None:
            f = plt.figure()
            plt.plot(train_metric, label = 'Training')
            plt.legend()
            plt.plot(valid_metric, label = 'Validation')
            plt.legend()
            pdf.savefig()
            plt.close()
        return score_train[1], score_vali[1]
    def get_msm(self, n_samples_show = int(1e3), pdf = None):
        """
        """
        np_polar = np.vectorize(cmath.polar)
        taus = np.empty((self.n_outputs-1,len(self.lags)))
        periods = np.empty((self.n_outputs-1,len(self.lags)))
        #--- Trasform the input trajectories using the network
        states_prob = self.model.predict([self.traj_ord, self.traj_ord_lag])[:,:self.n_outputs]
        coor_pred = np.argmax(states_prob, axis = 1)
        indexes = [np.where(coor_pred == np.multiply(np.ones_like(coor_pred), n)) for n in range(self.n_outputs)]
        states_num = [len(i[0]) for i in indexes]
        states_order = np.argsort(states_num).astype('int')[::-1]
        pred_ord = states_prob[:,states_order]
        if pdf is not None:
            f = plt.figure()
            ax = f.add_subplot(211)
            for i in range(self.n_dims()):
                ax.plot(np.arange(n_samples_show),self.traj_ord[:n_samples_show,i],'-')
            ax = f.add_subplot(212)
            for i in range(self.n_outputs):
                ax.plot(np.arange(n_samples_show),states_prob[:n_samples_show,i],':')
                print('Total probability for state {0:d} = {1:f}'.format(i, np.sum(states_prob[:,i])))
            ax.plot(np.arange(n_samples_show),coor_pred[:n_samples_show],'-')
            pdf.savefig()
            plt.close()
        K = self.vamp.estimate_koopman_op(pred_ord, self.lags[0])
        #K = self.vamp.estimate_koopman_constrained(pred_ord, self.lags[0])
        #eigv = np.linalg.eigvals(K)
        eigv, eigvec = np.linalg.eig(K)
        inds = np.argsort(np.abs(eigv))[-1::-1]
        r, phi = np_polar(eigv[inds[1:]])
        taus[:,0] =  -self.lags[0]/np.log(r)
        periods[:,0] = 2*np.pi*self.lags[0]/np.abs(phi)
        print('lag = ',self.lags[0])
        print('K = ',K)
        print('eigvs = ',eigv[inds])
        print('eigvec[:,0] = ',eigvec[:,inds[0]])
        output = ''
        for i in inds:
            eig = eigv[i]
            output += '\tEigenvalue = {0:f}\n'.format(eig)
            output += '\tTimescale = {0:f}\n'.format(-self.lags[0]/np.log(np.abs(eig)))
            if eig.imag != 0:
                r, phi = cmath.polar(eig)
                output += '\t\tperiod = {0:f}\n'.format(2*np.pi*self.lags[0]/np.abs(phi))
        print(output)
        for i_lag in range(1,len(self.lags)):
            lag = self.lags[i_lag]
            K = self.vamp.estimate_koopman_op(pred_ord, lag)
            #K = self.vamp.estimate_koopman_constrained(pred_ord, lag)
            #eigv = np.linalg.eigvals(K)
            eigv, eigvec = np.linalg.eig(K)
            inds = np.argsort(np.abs(eigv))[-1::-1]
            print('lag = ',self.lags[i_lag])
            print('K = ',K)
            print('eigvs = ',eigv[inds])
            print('eigvec[:,0] = ',eigvec[:,inds[0]])
            r, phi = np_polar(eigv[inds[1:]])
            taus[:,i_lag] =  -lag/np.log(r)
            periods[:,i_lag] = 2*np.pi*lag/np.abs(phi)
        print('Timescales as functions of lags: ',taus)
        print('Periods as functions of lags: ',periods)
        if pdf is not None:
            f = plt.figure()
            ax1 = f.add_subplot(211)
            for i in range(self.n_outputs-1):
                ax1.plot(self.lags,taus[i,:],'-')
            plt.ylabel('Timescale')
            ax2 = f.add_subplot(212)
            for i in range(self.n_outputs-1):
                ax2.plot(self.lags,periods[i,:],'-')
            plt.xlabel('Lag')
            plt.ylabel('Period')
            pdf.savefig()
            plt.close()
        ##--- Fit an MSM based on probability coordinates
        #cls = chick.cluster.Kmeans([states_prob,], n_clusters = 100)
        #cls.fit_predict()
        #cls.show(pdf, plot_trajs = True)
        #msm = chick.sorter.MarkovStateModel(cluster = cls)
        #msm.timescales_lags(lags = self.lags, nits = 2, pdf = pdf)
        #--- Fit an MSM based on discretized coordinates
        #cls = chick.cluster.Unique([coor_pred,])
        #cls.fit_predict()
        #cls.show(pdf, plot_trajs = True)
        #msm = chick.sorter.MarkovStateModel(cluster = cls)
        #msm.timescales_lags(lags = self.lags, pdf = pdf)
    def fit(self, n_run_training = 1, train_ratio = 0.9, batch_size = 2048, n_epochs = 100, learning_rate = 1e-4, n_repeats = 10, pdf = None):
        """
        Train the neural network a number of times n_run_training

        Parameters
        ----------
        n_run_training  int
            Number of iterations with different training set
        train_ratio float
            Which trajectory points percentage is used as training
        batch_size  int
            Batch size for stochastic gradient descend
        n_epochs    int
            Number of iterations in training
        n_repeats   int
            When score is 0.0, do the training again for n_repeats time
        """
        scores_train = np.empty(n_run_training)
        scores_vali = np.empty(n_run_training)
        for i_run in range(n_run_training):
            for i_repeat in range(n_repeats):
                self.model = self.define_model()
                X_now_train, X_lag_train, X_now_vali, X_lag_vali = self.shuffle_trajs(train_ratio)
                score_train, score_vali = self.train(X_now_train, X_lag_train, X_now_vali, X_lag_vali, batch_size, n_epochs, learning_rate, pdf)
                if np.abs(score_train) > numerical_precision:
                    break # that's good, it worked so stop iterating
                    print('WARNING: training failed, training another time')
                else:
                    print('WARNING: training failed')
            #--- Calculate the MSM for the neural network just trained
            self.get_msm(pdf = pdf)
            #--- Calculate state statistics for the neural network just trained 
            self.get_statistics(pdf = pdf)
            #--- Save scores for this training set
            scores_train[i_run] = score_train
            scores_vali[i_run] = score_vali
        return scores_train, scores_vali
    def get_statistics(self, pdf = None):
        """
        """
        #--- Transform the input trajectory using the network
        # states_prob: np.ndarray
        #   <number of samples> x <number of output states>
        #   value: probability that the sample belongs to the output state
        states_prob = self.model.predict([self.traj_ord, self.traj_ord_lag])[:, :self.n_outputs]
        #--- Order the output states based on their population
        coor_pred = np.argmax(states_prob, axis = 1)
        indexes = [np.where(coor_pred == np.multiply(np.ones_like(coor_pred), n)) for n in range(self.n_outputs)]
        states_num = [len(i[0]) for i in indexes]
        states_order = np.argsort(states_num).astype('int')[::-1]
        pred_ord = states_prob[:,states_order]
        #--- Collect statistics
        averages_w = np.dot(np.transpose(states_prob),self.traj_ord) / self.traj_ord.shape[0]
        averages = np.empty((self.n_outputs, self.traj_ord.shape[1]))
        stds = np.empty((self.n_outputs, self.traj_ord.shape[1]))
        if pdf is not None:
            for i_feature in range(self.traj_ord.shape[1]): # cycle over all the features
                f = plt.figure()
                ax1 = f.add_subplot(2,1,1)
                ax2 = f.add_subplot(2,1,2)
                for i_state in range(self.n_outputs): # cycle over all the output states
                    h,e = np.histogram(self.traj_ord[:,i_feature], weights = states_prob[:,i_state], bins = 100)
                    h = h / np.sum(h)
                    b = 0.5*(e[:-1]+e[1:])
                    ax1.plot(b,h,label = 'State {0:d}'.format(i_state))
                    h,e = np.histogram(self.traj_ord[coor_pred == i_state,i_feature], bins = 100)
                    print('Number of elements state {0:d}'.format(np.sum(coor_pred == i_state)))
                    averages[i_state, i_feature] = np.mean(self.traj_ord[coor_pred == i_state,i_feature])
                    stds[i_state, i_feature] = np.std(self.traj_ord[coor_pred == i_state,i_feature])
                    h = h / np.sum(h)
                    b = 0.5*(e[:-1]+e[1:])
                    ax2.plot(b,h,label = 'State {0:d}'.format(i_state))
                plt.xlabel('Feature {0:d}'.format(i_feature))
                plt.ylabel('Probability')
                pdf.savefig()
                plt.close()
            f = plt.figure()
            ax1 = f.add_subplot(2,1,1)
            for i_feature in range(self.traj_ord.shape[1]): # cycle over all the features
                ax1.plot(range(self.traj_ord.shape[0]), self.traj_ord[:,i_feature])
            ax2 = f.add_subplot(2,1,2)
            for i_state in range(self.n_outputs): # cycle over all the output states
                ax2.plot(range(self.traj_ord.shape[0]), states_prob[:,i_state])
            pdf.savefig()
            plt.close()
            f = plt.figure()
            ax = f.add_subplot(1,1,1)
            cax = ax.matshow(averages_w/np.max(averages_w, axis=0), cmap=plt.get_cmap('bwr'), interpolation='nearest')
            cbar = f.colorbar(cax, orientation='horizontal')
            pdf.savefig()
            plt.close()
            f = plt.figure()
            ax = f.add_subplot(1,1,1)
            cax = ax.matshow(averages/np.max(averages, axis=0), cmap=plt.get_cmap('bwr'), interpolation='nearest')
            cbar = f.colorbar(cax, orientation='horizontal')
            pdf.savefig()
            plt.close()
            f = plt.figure()
            ax = f.add_subplot(1,1,1)
            cax = ax.matshow((averages - np.min(self.traj_ord, axis=0))/(np.max(self.traj_ord, axis=0) - np.min(self.traj_ord, axis=0)), cmap=plt.get_cmap('bwr'), interpolation='nearest')
            cbar = f.colorbar(cax, orientation='horizontal')
            pdf.savefig()
            plt.close()
            print('averages_w = \n',averages_w)
            print('averages = \n',averages)
            print('stds = \n',stds)
            print('MIN = ',np.min(self.traj_ord, axis=0))
            print('MAX = ',np.max(self.traj_ord, axis=0))
        return averages_w

class MarkovStateModel(object):
    """
    cluster: class Cluster
    dtrajs: list of discrete trajecotries
    count_matrix: np.ndarray
        <n microstates> X <n microstates>
    transition_matrix: np.ndarray
        <n microstates> X <n microstates>
    """
    def __init__(self, cluster):
        self.cluster = cluster
        self.cluster.remove_empty_states()
        self.count_matrix = np.zeros((self.cluster.n_clusters(), self.cluster.n_clusters()))
        self.transition_matrix = np.zeros((self.cluster.n_clusters(), self.cluster.n_clusters()))
        self.verbose = 0
    def estimate(self, lag = 1, count_mode = 'sample'):
        while True:
            print('Estimating transition matrix with lag {0:d} and mode {1:s}'.format(lag, count_mode))
            print('Number of microstates {0:d}'.format(self.cluster.n_clusters()))
            self.count_matrix = np.zeros((self.cluster.n_clusters(), self.cluster.n_clusters()))
            self.transition_matrix = np.zeros((self.cluster.n_clusters(), self.cluster.n_clusters()))
            if count_mode == 'sliding':
                dt = 1
            elif count_mode == 'sample':
                dt = lag
            else:
                raise NotImplemented()
            for dtraj in self.cluster.dtrajs:
                ind_0 = 0
                while ind_0 < (len(dtraj) - lag):
                    inds_r = dtraj[ind_0:-lag]
                    inds_c = dtraj[ind_0+lag:]
                    for ind_r, ind_c in zip(inds_r, inds_c):
                        self.count_matrix[ind_r,ind_c] += 1
                    ind_0 += dt
            not_sampled = np.where(np.sum(self.count_matrix, axis = 0) == 0)[0]
            if len(not_sampled):
                print('Need to remove clusters ',not_sampled,' doing just the first')
                print('Removing samples of cluster {0:d}'.format(not_sampled[0]))
                self.cluster.remove_cluster(not_sampled[0])
            else:
                break # stop the cycle when the count matrix is connected
        self.transition_matrix = self.count_matrix / np.sum(self.count_matrix, axis = 0).reshape(1,self.cluster.n_clusters())
        if self.verbose > -1:
            print('dt = ',dt)
            print('count_matrix = ',self.count_matrix)
            print('np.sum(count_matrix, axis = 0) = ',np.sum(self.count_matrix, axis = 0))
            print('np.sum(np.sum(count_matrix, axis = 0) == 0) = ',np.sum(np.sum(self.count_matrix, axis = 0) == 0))
            print('transition_matrix = ',self.transition_matrix)
            print('np.sum(transition_matrix, axis = 0) = ',np.sum(self.transition_matrix, axis = 0))
    def eigenvalues(self):
        try:
            eigv = np.linalg.eigvals(self.transition_matrix)
        except:
            print('ERROR: something went wrong in eigenvales')
            print('\ttransition_matrix = ',self.transition_matrix)
            print('\tnp.sum(transition_matrix, axis = 0) = ',np.sum(self.transition_matrix, axis = 0))
            return np.nan*np.ones(self.cluster.n_clusters())
        inds = np.argsort(np.abs(eigv))[-1::-1]
        return eigv[inds]
    def eigenvectors(self):
        try:
            eigval, eigvec = np.linalg.eig(self.transition_matrix)
        except:
            print('ERROR: something went wrong in eigenvectors')
            print('\ttransition_matrix = ',self.transition_matrix)
            print('\tnp.sum(transition_matrix, axis = 0) = ',np.sum(self.transition_matrix, axis = 0))
            return np.nan*np.ones(self.cluster.n_clusters())
        inds = np.argsort(np.abs(eigval))[-1::-1]
        print('Eigenvectors: ',eigvec)
        print('Eigenvalues: ',eigval)
        print('Eigenvectors sorted: ',eigvec[:,inds])
        return eigvec[:,inds]
    def equilibrium(self):
        eigvec = self.eigenvectors()
        return eigvec[:,0] / np.sum(eigvec[:,0])
    def timescales(self, lag = 1, nits = None, pdf = None):
        if nits is None:
            nits = self.transition_matrix.shape[0]
        np_polar = np.vectorize(cmath.polar)
        eigv = self.eigenvalues()[1:nits+1]
        r, phi = np_polar(eigv)
        times =  -lag/np.log(r)
        periods = 2*np.pi*lag/np.abs(phi)
        print('Timescale: ',times)
        print('Period: ',periods)
        print('Equilibrium distribution: ',self.equilibrium())
        self.eigenvectors()
        if pdf is not None:
            f = plt.figure()
            ax = f.add_subplot(1,1,1)
            ax.plot(range(nits), times, 'ob')
            plt.xlabel('Index eigenvector')
            plt.ylabel('Timescale [#lag]')
            pdf.savefig()
            plt.close()
            f = plt.figure()
            ax = f.add_subplot(1,1,1)
            ax.plot(range(nits), periods, 'ob')
            plt.xlabel('Index eigenvector')
            plt.ylabel('Period [#lag]')
            pdf.savefig()
            plt.close()
        return times, periods
    def timescales_lags(self, lags, nits = None, pdf = None):
        """
        nits: int
            Number of implied timescales
        """
        if nits is None:
            nits = self.cluster.n_clusters()-1
        times = np.zeros((nits, len(lags)))
        periods = np.zeros((nits, len(lags)))
        for i_lag, lag in enumerate(lags):
            self.estimate(lag)
            times[:,i_lag], periods[:,i_lag] = self.timescales(lag, nits)
        if pdf is not None:
            f = plt.figure()
            ax = f.add_subplot(1,1,1)
            for its in range(nits):
                ax.plot(lags, times[its,:], 'o-')
            plt.xlabel('Lag')
            plt.ylabel('Timescale [#lag]')
            pdf.savefig()
            plt.close()
            f = plt.figure()
            ax = f.add_subplot(1,1,1)
            for its in range(nits):
                ax.plot(lags, periods[its,:], 'o-')
            plt.xlabel('Lag')
            plt.ylabel('Period [#lag]')
            pdf.savefig()
            plt.close()
            for i_lag, lag in enumerate(lags):
                f = plt.figure()
                ax = f.add_subplot(1,1,1)
                ax.plot(range(nits),times[:,i_lag], 'o-')
                plt.xlabel('Index')
                plt.ylabel('Timescale [#lag]')
                plt.title('Lag {0:d}'.format(int(lag)))
                pdf.savefig()
                plt.close()
                f = plt.figure()
                ax = f.add_subplot(1,1,1)
                ax.plot(range(nits),periods[:,i_lag], 'o-')
                plt.xlabel('Index')
                plt.ylabel('Period [#lag]')
                plt.title('Lag {0:d}'.format(int(lag)))
                pdf.savefig()
                plt.close()
        print('Timescales as a function of lag-times = ',times)
        print('Periods as a function of lag-times = ',periods)
        return times, periods
    def simulate(self, n_steps, initial_state = None):
        n_steps = int(n_steps)
        if initial_state is None:
            initial_state = np.random.randint(self.cluster.n_clusters())
        states = np.empty(n_steps).astype(int)
        states[0] = initial_state
        for i in range(1,n_steps):
            try:
                states[i] = np.random.choice(range(self.cluster.n_clusters()), p = self.transition_matrix[:,states[i-1]])
            except:
                print('ERROR: something went wrong in simulate')
                print('\tcount_matrix = ',self.count_matrix[:,states[i-1]])
                print('\ttransition_matrix = ',self.transition_matrix[:,states[i-1]])
                exit()
        return states
    def __str__(self):
        output = 'Markov Model\n'
        taus, periods = self.timescales()
        for tau in taus:
            output += '\tTimescale[#lag] = {0:f}\n'.format(tau)
        for i, eig in enumerate(self.eigenvalues()):
            output += '\tEigenvalue = {0:f}\n'.format(eig)
            if eig.imag != 0:
                r, phi = cmath.polar(eig)
                output += '\t\tperiod[#lag] = {0:f}\n'.format(2*np.pi/np.abs(phi))
        output += '\tTransition matrix: '+str(self.transition_matrix)+'\n'
        return output[:-1]
