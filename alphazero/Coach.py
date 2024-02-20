from pyximport import install as pyxinstall
from numpy import get_include
pyxinstall(setup_args={'include_dirs': get_include()})

from alphazero.SelfPlayAgent import SelfPlayAgent
from alphazero.utils import get_iter_file, dotdict, get_game_results, default_temp_scaling, default_const_args
from alphazero.Arena import Arena
from alphazero.GenericPlayers import RawMCTSPlayer, NNPlayer, MCTSPlayer
from alphazero.pytorch_classification.utils import Bar, AverageMeter

from torch import multiprocessing as mp
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
from torch_geometric.data import Data, Batch
from tensorboardX import SummaryWriter
from glob import glob
from queue import Empty
from time import time, sleep
from math import ceil, floor, sqrt
from enum import Enum

import numpy as np
import torch
import pickle
import os
import itertools
import ctypes

class ModeOfGameGen(Enum):
    # Where n = numPlayers; 
    
    # For all i1, ..., in there will be an equal number of games (i1,...,in)
    CROSS_PRODUCT  = 0
    # Each worker will uniformly at random get assigned an i1, ..., in  and they will compute only those games
    ONE_PER_WORKER = 1
    # A blend of the 2 above
    ROUND_ROBIN_EVERY_TIME = 2

DEFAULT_ARGS = dotdict({
    # NECESSARY TO DEFINE ARGS
        'run_name': 'boardgame',

    # Performance related args (as in will need to be set on a per computer basis)
        'cuda': torch.cuda.is_available(),
        'workers': mp.cpu_count(),
        # The size of the batches used for batching MCTS during self play. 
        #  Equivalent to the number of games that should be played at the same time in each worker.
        'process_batch_size': 256,
        'train_batch_size': 1024,
        # Same as process_batch_size but for arena
        'arena_batch_size': 64,
        'train_steps_per_iteration': 64,
        # should preferably be a multiple of process_batch_size and workers
        'gamesPerIteration': 256 * mp.cpu_count(),
        # Iterations where games are played randomly, 0 for none
        'numWarmupIters': 1, 

    # Kinda housekeeping args
        #Automatically set
        'startIter': 0,
        'numIters': 1000,
        '_num_players': None,  # Doesn't have to be changed, set automatically by the env.
        # The number of self play data generation iterations to skip. 
        #   This assumes that training data already exists for those iterations can be 
        #   used for training. For example, useful when training is interrupted because 
        #   data doesn't have to be generated from scratch because it's saved on disk.
        'skipSelfPlayIters': None,
        'selfPlayModelIter': None,
        'load_model': True,
        'checkpoint': 'checkpoint',
        'data': 'data',

    # Training Related Args
        'train_sample_ratio': 1,
        'averageTrainSteps': False,
        # Calculates the average number of samples in the training window
        #   if averageTrainSteps set to True, otherwise uses the latest
        #   number of samples, and does train_sample_ratio * avg_num_steps 
        #   or last_num_train_steps // train_batch_size
        #   training steps.
        'autoTrainSteps': True,   
        'train_on_past_data': False,
            'past_data_chunk_size': 25,
            'past_data_run_name': 'boardgame',
        # The number of past iterations to load self play training data from. 
        #   Starts at min and increments once every trainHistoryIncrementIters 
        #   iterations until it reaches max
        'minTrainHistoryWindow': 4,
        'maxTrainHistoryWindow': 20,
        'trainHistoryIncrementIters': 2,
        # Weather to use population based training to train hyperparameters
        'withPopulation' : False,
            # Useful if loading a pretrained model; makes sure network when loaded as getInitialArgs describes 
            'forceArgs' : False,
            # See the above enum - defines what games each worker will compute in self play
            'modeOfAssigningWork' : ModeOfGameGen.ONE_PER_WORKER,
            # The number of individuals in the population
            'populationSize' : 1,
            # A function that given an id of a memeber of the population (from 0 to populationSize - 1)
            #  that returns the trainable arguments to start that instance on
            'getInitialArgs' : default_const_args,
            # How often a roundRobin is done to optimize hyperparameters
            'roundRobinFreq' : 5,
            # If true then the round robin will take place by replacing a self play session
            #  every roundRobinFreq number of steps
            'roundRobinAsSelfPlay' : True,
            # Number of games played between each net in the round robin
            #  note populationSize^2 * (num below) games will be played in total
            'roundRobinGames' : 6,
            # The %age of the nets that will be killed
            #  They will be from the bottom of the round robin
            'percentageKilled' : 0.2,
            # Which net to use for compare to baseline and elo
            'bestNet' : 0,
            # Defines the max deviation from original values that can happen when a new models is created
            'deviation' : 0.2,
            # Whether to use symetries when generating games to train off of
        # if mctsCanonicalStates then must be True
        'symmetricSamples': True,

        'scheduler': torch.optim.lr_scheduler.MultiStepLR,
        'scheduler_args': dotdict({
            'milestones': [75, 125],
            'gamma': 0.1

            # 'min_lr': 1e-4,
            # 'patience': 3,
            # 'cooldown': 1,
            # 'verbose': False
        }),

        'lr': 1e-2,
        'optimizer': torch.optim.SGD,
        'optimizer_args': dotdict({
            'momentum': 0.9,
            'weight_decay': 1e-4
        }),
        'value_loss_weight': 1.5,

    # Monte Carlo Tree Seach Args
        'min_discount': 1,
        'fpu_reduction': 0.2,
        # This will let the MCTS search know that you are using canonical states (i.e all
        #   states are from the perspective of one player). It will then interpret the value
        #   vector returned from the neural network differently. It will go from :
        #     (ℙ(Player 1 wins), ℙ(Player 2 wins),ℙ(Draw)) to
        #     (ℙ(Player about to play wins), ℙ(Other player wins),ℙ(Draw))
        # Note this shouldn't affect the output of win_state, however should change the output
        #  of symmetry (i.e. when you call game.win_state() you should still get [0,1,0] if Player
        #  2 wins but when you call symmetry you should get [1,0,0] if player 2 wins and it's player
        #  2's turn
        'mctsCanonicalStates': False,
        # Num MCTS sims to use normally
        'numMCTSSims': 100,
        # or how many to use when doing fast sim
        'numFastSims': 20,
        # or how many to use when doing a warmup sim
        'numWarmupSims': 5,
        'probFastSim': 0.75,
        # None if none is wanted otherwise the # of simulations before the mcts is reset
        'mctsResetThreshold': None,
        # The initial temperate 
        'startTemp': 1,
        'temp_scaling_fn': default_temp_scaling,
        'root_policy_temp': 1.1,
        'root_noise_frac': 0.1,
        'add_root_noise': True,
        'add_root_temp': True,
        'cpuct': 1.25,
        # The root node is forced to play :
        #   forcedPlayoutsMultiplier * (pi(c) * n)**0.5 playouts
        #   per child where pi(c) is the probability acording to the policy and n is 
        #   the number of sims run at the root node
        #   set to 0 if none are wanted
        # Unsure if working properly
        'forcedPlayoutsMultiplier' : 0,

    # Neural Network Args
        'policy_softmax_temperature': 1.4,
        'value_softmax_temperature': 1.4,
        'nnet_type': 'resnet',  # 'resnet', 'fc' or 'graphnet'
        'value_dense_layers': [1024, 512],
        'policy_dense_layers': [1024, 512],
        # In Resnet and Graphnet
            # In Resnet   : The number of features per tile in game
            # In Graphnet : The number of hidden features in the graph
            'num_channels': 32,
            # In Resnet   : The number of times a Resnet is applied
            # In Graphnet : The number of message passing happens
            #   (so defining max radius each square can "see")
            'depth': 4,
            # In Resnet   : The number of features for the head of value fc
            'value_head_channels': 16,
            # In Resnet   : The number of features for the head of policy fc
            'policy_head_channels': 16,
            # In Graphnet : Both must be same, number of features at the head 
            #   of both fc layers

        # Only FC
            'input_fc_layers': [1024] * 4,  # only for fc networks
    
        # Wierd Error with low workers/ itters on graph

        # Only Graphnet
            #normally 2x num_channels
            'middle_layers' : [2*32],
            #if true it will be assumed that the environment has a constant 
            #  function which gives the edges of the graph
            'constant_edges' : False,

    # Comparing to Baseline Args
    'compareWithBaseline': True,
        'baselineCompareFreq': 1,
        'baselineTester': RawMCTSPlayer,
        #'arenaCompareBaseline': 128,
        'arenaCompare': 128,
        'arenaTemp': 0.25,
        'arenaMCTS': True,
        'arenaBatched': True,

    # Comparing with Past Args
    'compareWithPast': True,
        'pastCompareFreq': 1,
        'model_gating': True,
        'max_gating_iters': None,
        'min_next_model_winrate': 0.52,
        'use_draws_for_winrate': True,

    # Elo Calculation Args
    'calculateElo': True,
        'calculateEloFreq':1,
        'eloMCTS': 15,
        'eloGames':10,
        'eloMatches':10,
        'eloUniform': False,

    # Not working args
        'num_stacked_observations': 1  # TODO: built-in stacked observations (arg does nothing right now)
})


def get_args(args=None, **kwargs):
    new_args = DEFAULT_ARGS
    print()
    if args:
        new_args.update(args)
    for key, value in kwargs.items():
        setattr(new_args, key, value)

    if new_args.mctsCanonicalStates:
        assert new_args.symmetricSamples, "Counting who has won with cannonical state representation of board requires symetries to get win_state into correct form"

    if new_args.modeOfAssigningWork == ModeOfGameGen.ROUND_ROBIN_EVERY_TIME:
        assert new_args.roundRobinFreq == 1, "When using ROUND_ROBIN_EVERY_TIME as mode of assiging work the frequency of round robins must be one per round"

    if not new_args.compareWithPast and new_args.model_gating:
        print("Be aware you are not comaring to past but are gating so the model that is used for self play will only be changed when you restart the program")
    if new_args.compareWithPast and not new_args.model_gating:
        print("You are comparing to the past and not gating so the current version will be used always even if it is suboptimal")

    return new_args

class TrainState(Enum):
    STANDBY = 0
    INIT = 1
    INIT_AGENTS = 2
    SELF_PLAY = 3
    SAVE_SAMPLES = 4
    PROCESS_RESULTS = 5
    KILL_AGENTS = 6
    TRAIN = 7
    COMPARE_BASELINE = 8
    COMPARE_PAST = 9
    ROUND_ROBIN = 10


def _set_state(state: TrainState):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            self.state = state
            ret = func(self, *args, **kwargs)
            self.state = TrainState.STANDBY
            return ret
        return wrapper
    return decorator


class Coach:
    @_set_state(TrainState.INIT)
    def __init__(self, game_cls, nnet, args):
        np.random.seed()
        self.game_cls = game_cls
        self.numNets = args.populationSize if args.withPopulation else 1
        self.train_nets = np.full(self.numNets, None)
        self.self_play_nets = np.full(self.numNets, None)
        
        self.elo_play_net = nnet.__class__(game_cls, args)
        self.elo_play_net_2 = nnet.__class__(game_cls, args)
        self.args = args
        self.args._num_players = self.game_cls.num_players() + self.game_cls.has_draw()
        # If this is the first itter or no population is there
        #  then automatically set it to 0
        self.args.bestNet  = self.args.bestNet if (args.withPopulation and self.args.startIter != 0) else 0

        for i in range(0, self.numNets):
            argsi = args.copy()
            #print(args.getInitialArgs(i))
            if args.withPopulation:
                argsi.update(args.getInitialArgs(i))

            self.train_nets[i]      = nnet.__class__(game_cls, argsi)
            self.self_play_nets[i]  = nnet.__class__(game_cls, argsi)

        
        train_iter = self.args.startIter
        self.trainableArgs = set() if not(args.withPopulation) else set(self.args.getInitialArgs(0).keys())
        self.argsNotToCheck = {'startIter', 'process_batch_size'}
        self.argsUsedInTraining = {'scheduler', 'scheduler_args', 'optimizer',
            'optimizer_args', 'lr','nnet_type','num_channels','depth','value_head_channels',
            'policy_head_channels','input_fc_layers','value_dense_layers',
            'policy_dense_layers','train_batch_size', 'autoTrainSteps',
            'train_steps_per_iteration','train_on_past_data','minTrainHistoryWindow',
            'maxTrainHistoryWindow'}

        if self.args.load_model:
            networks = sorted(glob(self.args.checkpoint + '/' + self.args.run_name + '/*'))
            self.args.startIter = len(networks)//self.numNets
            if self.args.startIter == 0:
                for net in range(0, self.numNets):
                    self._save_model(self.train_nets[net], 0, net)
                self.args.startIter = 1

            train_iter = self.args.startIter - 1
            for net in range(0, self.numNets): 
                self._load_model(self.train_nets[net], train_iter, net)
                if self.args.withPopulation and self.args.forceArgs:
                    self.train_nets[net].args.update(args.getInitialArgs(i))
            self.args.bestNet = self.train_nets[0].args.bestNet
            del networks

        self.self_play_iter = np.full(self.numNets, 0)

        for net in range(0, self.numNets):
            if self.train_nets[i].args.selfPlayModelIter == 0:
                self.self_play_iter[net] = 0
            else:
                self.self_play_iter[net] = self.train_nets[i].args.selfPlayModelIter or train_iter

        if self.args.model_gating:
            for net in range(0, self.numNets):
                self._load_model(self.self_play_nets[net], self.self_play_iter[net],net)
        

        self.gating_counter = np.zeros(self.numNets)
        self.warmup = False
        self.loss_pis = np.zeros(self.numNets)
        self.loss_vs = np.zeros(self.numNets)
        self.sample_time = 0
        self.iter_time = 0
        self.eta = 0
        self.arena = None
        self.model_iter = self.args.startIter
        self.agents = []
        self.input_tensors = []
        self.input_tensors2 = []
        self.input_queues = []
        self.policy_tensors = []
        self.value_tensors = []
        self.batch_ready = []
        self.games_for_agent = []
        self.stop_train = mp.Event()
        self.pause_train = mp.Event()
        self.stop_agents = mp.Event()
        for net in range(0, self.numNets):
            self.train_nets[net].stop_train = self.stop_train
            self.train_nets[net].pause_train = self.pause_train
        self.ready_queue = mp.Queue()
        self.file_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.completed = mp.Value('i', 0)
        self.games_played = mp.Value('i', 0)
        if self.args.run_name != '':
            self.writer = SummaryWriter(log_dir='runs/' + self.args.run_name)
        else:
            self.writer = SummaryWriter()
        # self.args.expertValueWeight.current = self.args.expertValueWeight.start

    # Returns smaller dot dictionary of all attributes that can be learned and their vals in current dict
    def get_trainable_attributes(self, args : dotdict) -> dotdict:
        return dotdict({key: args[key] for key in self.trainableArgs})

    def _load_model(self, model, iteration, i):
        modelArgs = model.load_checkpoint(
            folder=os.path.join(self.args.checkpoint, self.args.run_name),
            filename=get_iter_file(iteration, i)
        )
        
        # Check that it agrees not on trainable args if training more than one
        if self.numNets != 1:
            for key in modelArgs:
                val = self.args[key]
                assert ((self.args[key] == val) or (key in self.trainableArgs) or (key in self.argsNotToCheck)), "One of the models differs from default arguments not on a training arguement - {} is {} in the model but {} in the defaults".format(key, val, self.args.get(key))

    def _save_model(self, model, iteration, i):
        model.save_checkpoint(
            folder=os.path.join(self.args.checkpoint, self.args.run_name),
            filename=get_iter_file(iteration, i)
        )

    def learn(self):
        print('Because of batching, it can take a long time before any games finish.')

        try:
            print("-----args-------")
            for i,j in self.args.items():
                print("\"{}\" : {},".format(i,j))
            while self.model_iter <= self.args.numIters:
                print(f'------ITER {self.model_iter}------')
                reset = None
                dat = [None]*5
                if ((not self.args.skipSelfPlayIters\
                        or self.model_iter > self.args.skipSelfPlayIters)\
                    and not (self.args.train_on_past_data and self.model_iter == self.args.startIter)):
                    if self.model_iter <= self.args.numWarmupIters:
                        print('Warmup: random policy and value')
                        self.warmup = True
                    # elif self.self_play_iter == 0:
                    #     self.warmup = True
                    elif self.warmup:
                        self.warmup = False

                    if self.warmup:
                        print('Warmup: random policy and value')

                    
                    
                    if self.args.withPopulation and\
                       self.args.roundRobinAsSelfPlay and\
                       ((self.model_iter - 1) %self.args.roundRobinFreq == 0):
                       reset = (self.args.modeOfAssigningWork, 
                                self.args.startTemp,
                                self.args.gamesPerIteration)
                       if self.args.roundRobinFreq != 1:
                           self.args.modeOfAssigningWork = ModeOfGameGen.CROSS_PRODUCT
                           self.args.startTemp = self.args.arenaTemp
                           self.args.gamesPerIteration = self.args.roundRobinGames * (self.numNets**self.game_cls.num_players())

                    selfPlay = None
                    if reset != None and self.args.compareWithPast and (self.model_iter - 1) % self.args.pastCompareFreq == 0:
                        selfPlay = self.args.arenaCompare

                    for i in range(self.args.workers):
                        self.games_for_agent.append(self.gamesFor(i, self.args.workers, self.args.modeOfAssigningWork, selfPlay))
                        print(self.games_for_agent[i])
                    

                    if selfPlay != None:
                        self.args.gamesPerIteration = sum([num*len(listOfGames) for num,listOfGames in self.games_for_agent])
                    self.generateSelfPlayAgents(exact = (reset != None))
    
                    self.processSelfPlayBatches(self.model_iter)
                    if self.stop_train.is_set():
                        break
                    sleep(2)
                    self.saveIterationSamples(self.model_iter)
                    if self.stop_train.is_set():
                        break
                    dat = self.processGameResults(self.model_iter)
                    
                    if reset != None:
                        self.args.modeOfAssigningWork, self.args.startTemp, self.args.gamesPerIteration = reset

                    if self.stop_train.is_set():
                        break
                    self.killSelfPlayAgents()
                    if self.stop_train.is_set():
                        break

                    if self.args.withPopulation and ((self.model_iter - 1) %self.args.roundRobinFreq == 0):
                        self.roundRobin(self.model_iter-1, dat[0], dat[1], dat[2])

                self.train(self.model_iter)
                if self.stop_train.is_set():
                    break
                
                if self.args.compareWithBaseline and (self.model_iter - 1) % self.args.baselineCompareFreq == 0:
                    net = self.args.bestNet
                    self.compareToBaseline(self.model_iter, net)
                    if self.stop_train.is_set():
                        break

                if self.args.compareWithPast and (self.model_iter - 1) % self.args.pastCompareFreq == 0:
                    for net in range(0, self.numNets):
                        if reset!=None:
                            self.compareToPast(self.model_iter-1, net, True, dat[4][net], dat[5][net])
                        else:
                            self.compareToPast(self.model_iter, net)
                    if self.stop_train.is_set():
                        break

                if self.args.calculateElo and (self.model_iter - 1) % self.args.calculateEloFreq == 0:
                    net = self.args.bestNet
                    self.calculateElo(net)

                for net in range(0, self.numNets):
                    self.writer.add_scalar(str(net)+'win_rate/self_play_model', self.self_play_iter[net], self.model_iter)
                self.model_iter += 1
                print()

        except KeyboardInterrupt:
            pass

        print()
        self.writer.close()
        if self.agents:
            self.killSelfPlayAgents()

    # Tries to evenly divide up all the different combinations of self play games 
    #  between all the workers
    def gamesFor(self, i : int, numWorkers : int, modeOfAssigningWork, numSelfPlay = None):

        numPlayers = self.game_cls.num_players()
        if modeOfAssigningWork == ModeOfGameGen.CROSS_PRODUCT:
            numPerPair = self.args.roundRobinGames
            lists = list(itertools.product(list(range(0, self.numNets)), repeat = numPlayers))
     
            step = len(lists)//numWorkers
            rem  = len(lists)%numWorkers
        
            retList = lists[floor(step*i) : floor(step*(i+1))] + ([] if (rem == 0) else [lists[-1-floor(rem/numWorkers * i)]])
            listOfSelfPlays = []

            selfStep = self.numNets/numWorkers
            for net in range(floor(selfStep*i), floor(selfStep*(i+1))):
                if numSelfPlay != None:
                    pairsToAdd = floor(numSelfPlay/numPerPair/numPlayers)
                    for position in range(numPlayers):
                        beforeMe         = [net+self.numNets]*position
                        afterMe          = [net+self.numNets]*(numPlayers - 1 - position)
                        thisGame         = tuple(beforeMe + [net] + afterMe)
                        allTheseGames    = pairsToAdd*[thisGame]
                        listOfSelfPlays += allTheseGames

            return (numPerPair, listOfSelfPlays + retList)

        elif modeOfAssigningWork == ModeOfGameGen.ONE_PER_WORKER:
            nets = np.array(range(self.numNets))

            ret = [np.random.choice(nets) for _ in range(self.game_cls.num_players())]
            return (self.args.gamesPerIteration, [tuple(ret)])

        elif self.args.modeOfAssigningWork == ModeOfGameGen.ROUND_ROBIN_EVERY_TIME:
            proportionWorkersRobin = (self.args.roundRobinGames*(self.numNets**numPlayers))/self.args.gamesPerIteration
            workersRobin   = ceil(proportionWorkersRobin*numWorkers)
            workersRegular = numWorkers-workersRobin

            if i < workersRobin:
                ret = self.gamesFor(i, workersRobin, ModeOfGameGen.CROSS_PRODUCT)
            else:
                ret = self.gamesFor(i, workersRegular, ModeOfGameGen.ONE_PER_WORKER)

            return ret

        else:
            raise ValueError("modeOfAssigningWork must be set to an element of ModeOfGameGen (or the mode you have picked is not implemented)")
 
    @_set_state(TrainState.INIT_AGENTS)
    def generateSelfPlayAgents(self, exact = False):
        self.stop_agents = mp.Event()
        self.ready_queue = mp.Queue()
        for i in range(self.args.workers):
            if self.args.nnet_type != "graphnet":
                self.input_tensors.append(torch.zeros(
                    [self.args.process_batch_size, *self.game_cls.observation_size()]
                ))
                self.input_tensors[i].share_memory_()
                if self.args.cuda:
                    self.input_tensors[i].pin_memory()
            else:
                obs_size = self.game_cls.observation_size()
                self.input_tensors.append(torch.zeros(
                    [self.args.process_batch_size, obs_size[1], obs_size[0]]
                ))
                self.input_tensors[i].share_memory_()
                if self.args.cuda:
                    self.input_tensors[i].pin_memory()


                self.input_tensors2.append(torch.zeros(
                    [self.args.process_batch_size, 2, obs_size[2]], dtype=int
                ))
                self.input_tensors2[i].share_memory_()
                if self.args.cuda:
                    self.input_tensors2[i].pin_memory()
            
            self.input_queues.append(mp.Queue())

            self.policy_tensors.append(torch.zeros(
                [self.args.process_batch_size, self.game_cls.action_size()]
            ))
            self.policy_tensors[i].share_memory_()

            self.value_tensors.append(torch.zeros(
                [self.args.process_batch_size, self.game_cls.num_players() + self.game_cls.has_draw()]
            ))
            self.value_tensors[i].share_memory_()
            self.batch_ready.append(mp.Event())

            if self.args.cuda:
                self.policy_tensors[i].pin_memory()
                self.value_tensors[i].pin_memory()

            #print(self.gamesFor(i))
            self.agents.append(
                SelfPlayAgent(i, self.games_for_agent[i], self.game_cls, self.ready_queue, self.batch_ready[i],
                              self.input_tensors[i], self.policy_tensors[i], self.value_tensors[i], self.file_queue,
                              self.result_queue, self.completed, self.games_played, self.stop_agents, self.pause_train,
                              self.args, self.input_tensors2[i] if self.args.nnet_type == "graphnet" else None, _is_arena=False,  _is_warmup=self.warmup, _exact_game_count=exact)
            )
            self.agents[i].daemon = True

            self.agents[i].start()
            #assert 1 ==0 


    @_set_state(TrainState.SELF_PLAY)
    def processSelfPlayBatches(self, iteration):
        sample_time = AverageMeter()
        bar = Bar('Generating Samples', max=self.args.gamesPerIteration)
        end = time()
        nnets      = self.self_play_nets if self.args.model_gating else self.train_nets
        othernnets = self.train_nets     if self.args.model_gating else self.self_play_nets
        
        nnets      = np.concatenate((nnets, othernnets))
        n = 0

        while self.completed.value != self.args.workers:
            if self.stop_train.is_set() and not self.stop_agents.is_set():
                self.stop_agents.set()

            try:
                id, netsNumsList = self.ready_queue.get(timeout=1)
                #indexToNet = self.games_for_agent[id][1]
                #if id == 0:
                #    print(netsNumsList)
                #print(id)
                #'print(self.input_queues)
                cumulative = 0
                if self.args.nnet_type != "graphnet":
                    input_tensor = self.input_tensors[id]#self.input_queues[id].get()
                else :
                    input_x = self.input_tensors[id]
                    input_edge = self.input_tensors2[id]
                #print(input_tensor)
                for net, number in netsNumsList:
                    if number == 0:
                        continue;
                    if self.args.nnet_type != "graphnet":
                        to_process = input_tensor[cumulative:cumulative+number]
                    else:
                        to_process = Data(torch.cat([*input_x[cumulative:cumulative+number]]), 
                                        torch.cat([*input_edge[cumulative:cumulative+number]],-1))

                    policy, value = nnets[net].process(to_process, batch_size=number)
                    self.policy_tensors[id][cumulative: cumulative + number].copy_(policy)
                    self.value_tensors[id][cumulative: cumulative + number].copy_(value)
                    cumulative += number
                
                self.batch_ready[id].set()
            except Empty:
                pass

            size = self.games_played.value
            if size > n:
                sample_time.update((time() - end) / (size - n), size - n)
                n = size
                end = time()
            bar.suffix = f'({size}/{self.args.gamesPerIteration}) Sample Time: {sample_time.avg:.3f}s | Total: {bar.elapsed_td} | ETA: {bar.eta_td:}'
            bar.goto(size)
            self.sample_time = sample_time.avg
            self.iter_time = bar.elapsed_td
            self.eta = bar.eta_td

        if not self.stop_agents.is_set(): self.stop_agents.set()
        bar.update()
        bar.finish()
        self.writer.add_scalar('loss/sample_time', sample_time.avg, iteration)
        print()

    @_set_state(TrainState.SAVE_SAMPLES)
    def saveIterationSamples(self, iteration):
        num_samples = self.file_queue.qsize()
        print(f'Saving {num_samples} samples')
        if self.args.nnet_type != "graphnet":
            data_tensor = torch.zeros([num_samples, *self.game_cls.observation_size()])
        else:
            obs_size = self.game_cls.observation_size()
            x_tensor    = torch.zeros([num_samples, obs_size[1], obs_size[0]])
            edge_tensor = torch.zeros([num_samples, 2, obs_size[2]])
        
        policy_tensor = torch.zeros([num_samples, self.game_cls.action_size()])
        value_tensor = torch.zeros([num_samples, self.game_cls.num_players() + self.game_cls.has_draw()])
        for i in range(num_samples):
            #print(i)
            data, policy, value = self.file_queue.get() 
            
            if self.args.nnet_type != "graphnet":
                data_tensor[i] = torch.from_numpy(data)
            else:
                x_tensor[i]    = data.x
                if not self.args.constant_edges:
                    edge_tensor[i] = data.edge_index
            policy_tensor[i] = torch.from_numpy(policy)
            value_tensor[i] = torch.from_numpy(value)

        folder = os.path.join(self.args.data, self.args.run_name)
        filename = os.path.join(folder, get_iter_file(iteration).replace('.pkl', ''))
        if not os.path.exists(folder): os.makedirs(folder)

        if self.args.nnet_type != "graphnet":
            torch.save(data_tensor, filename + '-data.pkl', pickle_protocol=pickle.HIGHEST_PROTOCOL)
            del data_tensor
        else:
            torch.save(x_tensor, filename + '-xdata.pkl', pickle_protocol=pickle.HIGHEST_PROTOCOL)
            if not self.args.constant_edges:
                torch.save(edge_tensor, filename + '-edgedata.pkl', pickle_protocol=pickle.HIGHEST_PROTOCOL)
            del x_tensor
            del edge_tensor


        torch.save(policy_tensor, filename + '-policy.pkl', pickle_protocol=pickle.HIGHEST_PROTOCOL)
        torch.save(value_tensor, filename + '-value.pkl', pickle_protocol=pickle.HIGHEST_PROTOCOL)
        del policy_tensor
        del value_tensor

    @_set_state(TrainState.PROCESS_RESULTS)
    def processGameResults(self, iteration):
        wins, draws, numAvgGameLength, self_wins, self_draws = get_game_results(self.numNets, self.result_queue, self.game_cls)

        numWins        = np.sum(wins, axis = tuple(range(len(wins.shape) - 1)))
        numDraws       = np.sum(draws)
        numNormalGames = np.sum(numWins) + numDraws
        numPlayers     = self.game_cls.num_players()

        for i in range(numPlayers):
            self.writer.add_scalar(f'win_rate/player{i}', (
                    numWins[i] + (numDraws/numPlayers if self.args.use_draws_for_winrate else 0))/ 
                    numNormalGames, iteration)
        self.writer.add_scalar('win_rate/draws', numDraws / numNormalGames, iteration)
        self.writer.add_scalar('win_rate/avg_game_length', numAvgGameLength, iteration)
        

        totalWins    = np.zeros(self.numNets)
        totalDraws   = np.zeros(self.numNets)
        totalGamesBy = np.zeros(self.numNets)

        for win, numWins1 in np.ndenumerate(wins):
            if numWins1 == 0:
                continue;
            whoWon  = win[-1]
            totalWins[win[whoWon]] += numWins1
            for p in win[:-1]:
                totalGamesBy[p] += numWins1
        
        for draw, numDraws in np.ndenumerate(draws):
            if numDraws == 0:
                continue;
            for p in (draw):
                totalDraws[p] += numDraws

        totalSelfWins  = np.zeros((self.numNets,2))
        totalSelfDraws =  np.zeros(self.numNets)

        for net in range(self.numNets):
            for pos in range(numPlayers):
                for whoWon in range(numPlayers):
                    if whoWon == pos:
                        totalSelfWins[net][0] += self_wins[net][pos][whoWon]
                    else:
                        totalSelfWins[net][1] += self_wins[net][pos][whoWon]
                totalSelfDraws[net] += self_draws[net][pos]


        #print(totalWins)
        #print(totalDraws)

        #print(totalSelfWins)
        #print(totalSelfDraws)

        return totalWins, totalDraws, totalGamesBy, numAvgGameLength, totalSelfWins, totalSelfDraws

    @_set_state(TrainState.KILL_AGENTS)
    def killSelfPlayAgents(self):
        # clear queues to prevent deadlocking
        for _ in range(self.ready_queue.qsize()):
            try:
                self.ready_queue.get_nowait()
            except Empty:
                break
        for _ in range(self.file_queue.qsize()):
            try:
                self.file_queue.get_nowait()
            except Empty:
                break
        for _ in range(self.result_queue.qsize()):
            try:
                self.result_queue.get_nowait()
            except Empty:
                break

        for agent in self.agents:
            agent.join()
            del self.input_tensors[0]
            del self.policy_tensors[0]
            del self.value_tensors[0]
            del self.batch_ready[0]

        self.agents = []
        self.input_tensors = []
        #self.input_queues = []
        self.policy_tensors = []
        self.value_tensors = []
        self.batch_ready = []
        self.games_for_agent = []
        self.ready_queue = mp.Queue()
        self.file_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.completed = mp.Value('i', 0)
        self.games_played = mp.Value('i', 0)

    @_set_state(TrainState.TRAIN)
    def train(self, iteration):
        num_train_steps = 0
        sample_counter = 0
        
        def add_tensor_dataset(train_iter, tensor_dataset_list, run_name=self.args.run_name):
            filename = os.path.join(
                os.path.join(self.args.data, run_name), get_iter_file(train_iter).replace('.pkl', '')
            )
            
            try:
                if self.args.nnet_type != "graphnet":
                    data_tensor = torch.load(filename + '-data.pkl')
                else:
                    x_tensor    = torch.load(filename + '-xdata.pkl')
                    if not self.args.constant_edges:
                        edge_tensor = torch.load(filename + '-edgedata.pkl').to(int)
                    else:
                        edge_tensor = self.game_cls.get_edges().expand(x_tensor.size(0), 2, self.game_cls.observation_size()[2])
                policy_tensor = torch.load(filename + '-policy.pkl')
                value_tensor = torch.load(filename + '-value.pkl')
            except FileNotFoundError as e:
                print('Warning: could not find tensor data. ' + str(e))
                return
            if self.args.nnet_type != "graphnet":
                tensor_dataset_list.append(
                    TensorDataset(data_tensor, policy_tensor, value_tensor)
                )
            else:
                tensor_dataset_list.append(
                    TensorDataset(x_tensor, edge_tensor, policy_tensor, value_tensor)
                )
            nonlocal num_train_steps
            if self.args.averageTrainSteps:
                nonlocal sample_counter
                num_train_steps += policy_tensor.size(0)
                sample_counter += 1
            else:
                num_train_steps = policy_tensor.size(0)

        def train_data(tensor_dataset_list, train_on_all=False):
            dataset = ConcatDataset(tensor_dataset_list)
            dataloader = DataLoader(dataset, batch_size=self.args.train_batch_size, shuffle=True,
                                    num_workers=self.args.workers, pin_memory=True)
            if self.args.averageTrainSteps:
                nonlocal num_train_steps
                num_train_steps //= sample_counter
            train_steps = len(dataset) // self.args.train_batch_size \
               if train_on_all else (num_train_steps // self.args.train_batch_size
                   if self.args.autoTrainSteps else self.args.train_steps_per_iteration)
            result = np.zeros([2,self.numNets])
            
            if self.argsUsedInTraining.intersection(self.trainableArgs) == set():
                result[0][0], result[1][0] = self.train_nets[0].train(dataloader, train_steps)
                self._save_model(self.train_nets[0], iteration, 0)
                for toTrain in range(1, self.numNets):
                    print("Training Net | Using other data as no args used in training are trainable - so model can be coppied")
                    result[0][toTrain], result[1][toTrain] = result[0][0], result[1][0]
                    tempArgs = self.train_nets[toTrain].args.copy()
                    self._load_model(self.train_nets[toTrain], iteration, 0)
                    self.train_nets[toTrain].args = tempArgs
                    print()
            else:
                for toTrain in range(0, self.numNets):
                    dataset = ConcatDataset(tensor_dataset_list)
                    dataloader = DataLoader(dataset, batch_size=self.args.train_batch_size, shuffle=True,
                                    num_workers=self.args.workers, pin_memory=True)
            
                    result[0][toTrain], result[1][toTrain] = self.train_nets[toTrain].train(dataloader, train_steps)
    

            del dataloader
            del dataset

            return result

        if self.args.train_on_past_data and iteration == self.args.startIter:
            next_start_iter = 1
            total_iters = len(
                glob(os.path.join(os.path.join(self.args.data, self.args.past_data_run_name), '*.pkl'))
            ) // 3
            num_chunks = ceil(total_iters / self.args.past_data_chunk_size)
            print(f'Training on past data from run "{self.args.past_data_run_name}" in {num_chunks} chunks of '
                  f'{self.args.past_data_chunk_size} iterations ({total_iters} iterations in total).')

            for _ in range(num_chunks):
                datasets = []
                i = next_start_iter
                for i in range(next_start_iter, min(
                    next_start_iter + self.args.past_data_chunk_size, total_iters + 1
                )):
                    add_tensor_dataset(i, datasets, run_name=self.args.past_data_run_name)
                next_start_iter = i + 1

                self.loss_pis, self.loss_vs = train_data(datasets, train_on_all=True)
                del datasets
        else:
            datasets = []

            # current_history_size = self.args.numItersForTrainExamplesHistory
            current_history_size = min(
                max(
                    self.args.minTrainHistoryWindow,
                    (iteration + self.args.minTrainHistoryWindow) // self.args.trainHistoryIncrementIters
                ),
                self.args.maxTrainHistoryWindow
            )

            [add_tensor_dataset(i, datasets) for i in range(max(1, iteration - current_history_size), iteration + 1)]
            self.loss_pis, self.loss_vs = train_data(datasets)

        net = self.args.bestNet
        self.writer.add_scalar('loss/policy', self.loss_pis[net], iteration)  # TODO: policy loss not showing up in tensorboard
        self.writer.add_scalar('loss/value', self.loss_vs[net], iteration)
        self.writer.add_scalar('loss/total', self.loss_pis[net] + self.loss_vs[net], iteration)

        for net in range(self.numNets):
            self._save_model(self.train_nets[net], iteration, net)

    def calculateElo(self, player):
        if not os.path.exists("elo/"+self.args.run_name):
            os.makedirs("elo/"+self.args.run_name)
        networks = sorted(glob(self.args.checkpoint + '/' + self.args.run_name + '/*'))
        if self.model_iter == 1:
            np.savetxt('elo/'+self.args.run_name+'/' + 'ELOS.csv', [[0]], delimiter=",")

        elos = np.loadtxt('elo/'+self.args.run_name+'/' + 'ELOS.csv', delimiter=',')
        elos = [elos]
        elos = np.array(elos).flatten()
        # print(elos)
        # current_elo = elos[len(elos)-1]
        current_elo = 0

        sf_args = self.args.copy()
        sf_args.numMCTSSims = self.args.eloMCTS
        cls = MCTSPlayer if self.args.arenaMCTS else NNPlayer
        nplayer = cls(self.train_nets[player], self.game_cls, sf_args)
        running_score = 0
        running_expectation = 0

        #Sample from harmonic distribution because integral calculus of 
        #normal distribution is nasty and i dont like the error function
        #Not used, but testing it out. Currently it is just a uniform sampling
        def harmonic(n):
            a = 0
            for i in range(0, n):
                a += 1/(1+i)
            return a

        num_games = self.args.eloMatches
        harmonic_coef = 1/harmonic(len(elos))
        probs = harmonic_coef / (1+np.arange(0, len(elos)))

        #opponents = np.random.choice(np.flip(np.arange(0, len(elos))), p=probs, size=num_games)
        
        opponents = np.random.choice(np.arange(0, len(elos)), size=num_games)
        if self.args.eloUniform:
            opponents = np.arange(max(0, len(elos)-num_games), len(elos))
        print(f"Pitting against the following iters:{opponents}")
        for i in opponents:
            print(f'PITTING AGAINST ITERATION {i} FOR ELO CALCULATION ')
            opponent_elo = elos[i]
            self._load_model(self.elo_play_net, i, player)
            pplayer = cls(self.elo_play_net, self.game_cls, sf_args)
            players = [nplayer] + [pplayer] * (self.game_cls.num_players() - 1)
            self.arena = Arena(players, self.game_cls, use_batched_mcts=False, args=sf_args)
            wins, draws, winrates = self.arena.play_games(self.args.eloGames, verbose=False)

            expected_score = 1/(1+10**( (opponent_elo-current_elo)/400 ))
            actual_score = (wins[0] + 0.5*draws)#/(wins[0]+wins[1]+draws)
            running_expectation += 10*expected_score
            running_score += actual_score
            #current_elo = current_elo + 32*(actual_score - 10*expected_score)

        current_elo = current_elo + 32*(running_score - running_expectation)
        current_elo = max(current_elo, 0)
        elos = np.append(elos, current_elo)
        np.savetxt('elo/'+self.args.run_name+'/' + 'ELOS.csv', [elos], delimiter=",")
        print(f'Self play ELO : {current_elo}')
        self.writer.add_scalar('elo/self_play_elo', current_elo, self.model_iter)

    def randomPreviousGames(self, ITER):
        elos = np.loadtxt('elo/'+self.args.run_name+'/ELOS.csv', delimiter=',')
        elos = [elos]
        elos = np.array(elos).flatten()
        for i in range(1, len(elos)):
            #print(i, elos[i])
            self.writer.add_scalar('elo/self_play_elo_3', elos[i], i)
    
    def sweepCPUCT(self, num):
        params = np.linspace(0.25, 5, num)

        self._load_model(self.elo_play_net, self.model_iter)
        cls = MCTSPlayer if self.args.arenaMCTS else NNPlayer

        bestIndex = -1
        highestRate = 0
        for i in range(num):
            print(f"Testing CPUCT: {params[i]}")
            new_args = self.args.copy()
            new_args.cpuct = params[i]
            nplayer = cls(self.elo_play_net, self.game_cls, new_args)
            pplayer = cls(self.elo_play_net, self.game_cls, self.args)
            players = [nplayer] + [pplayer] * (self.game_cls.num_players() - 1)
            self.arena = Arena(players, self.game_cls, use_batched_mcts=False, args=self.args)
            wins, draws, wrs = self.arena.play_games(10, verbose=False)
            if wrs[0] > highestRate and wrs[0] > 0.52:
                highestRate = wrs[0]
                bestIndex = i

        if bestIndex != -1:
            print(f"Optimimum CPUCT: {params[bestIndex]}")
            self.args.cpuct = params[bestIndex]
        self.writer.add_scalar("hyperparmeters/CPUCT", self.args.cpuct, self.model_iter)

    #Testing code--- Not working for the moment
    def tuneHyperparams(self, num):
        print()
        print(f"Tuning hyperparmeters with population size of {num}")
        if not os.path.exists("hyperparams/"+self.args.run_name):
            os.makedirs("hyperparams/"+self.args.run_name)
        if self.model_iter == 1:
            np.savetxt("hyperparams/"+self.args.run_name+"/params.csv", [[self.args.cpuct]], delimiter=",")

        recent = np.loadtxt('hyperparams/'+self.args.run_name+'/params.csv', delimiter=',')
        recent = [recent]
        recent = np.array(recent).flatten()
        # Loaded previous hyperparameters
        print(f"Loading most recent CPUCT: {recent}")
        new_args = self.args.copy()
        new_args.cpuct = recent[0]
        params = [new_args.copy() for i in range(num)]
        WINRATES = [0] * num
        RANGE = 0.35
        #Mutate some params
        params[0].numMCTSSims = 15
        for i in range(1, len(params)):
            # params[i].fpu_reduction = np.clip(params[i].fpu_reduction + params[i].fpu_reduction * np.random.uniform(-RANGE, RANGE), 0, 1)
            params[i]["cpuct"] = np.clip(params[i].cpuct + np.random.uniform(-RANGE, RANGE), 0.25, 5)
            # params[i].root_policy_temp = params[i].root_policy_temp + params[i].root_policy_temp * np.random.uniform(-RANGE, RANGE)
            # params[i].root_noise_frac = params[i].root_noise_frac + params[i].root_noise_frac * np.random.uniform(-RANGE, RANGE)
            params[i].numMCTSSims = 15
            #print(params[i].fpu_reduction, params[i].cpuct, params[i].root_policy_temp, params[i].root_noise_frac)
        #Round robin
        for i in range(len(params)):
            print(params[i].cpuct)
        self._load_model(self.elo_play_net, self.model_iter, player)
        cls = MCTSPlayer if self.args.arenaMCTS else NNPlayer
        
        for p in range(len(params)):
            for o in range(len(params)):
                if p != o:
                    print(f"Pitting {p} against {o} with CPUCT: {params[p].cpuct} and {params[o].cpuct}. {((p)*num+(o))/(num*num) * 100}% Complete")
                    nplayer = cls(self.elo_play_net, self.game_cls, params[p])
                    pplayer = cls(self.elo_play_net, self.game_cls, params[o])
                    players = [nplayer] + [pplayer] * (self.game_cls.num_players() - 1)
                    self.arena = Arena(players, self.game_cls, use_batched_mcts=False, args=self.args)
                    wins, draws, wrs = self.arena.play_games(6, verbose=False)
                    WINRATES[p] += wrs[0]

        best = np.argmax(WINRATES)
        recent[0] = params[best].cpuct
        print("Optimimum Found:")
        print(f"CPUCT: {params[best].cpuct}")
        self.args = params[best].copy()
        np.savetxt("hyperparams/"+self.args.run_name+"/params.csv", [recent], delimiter=",")
        # self.writer.add_scalar("hyperparmeters/FPU", params[best].fpu_reduction, self.model_iter)
        self.writer.add_scalar("hyperparmeters/CPUCT", params[best].cpuct, self.model_iter)
        # self.writer.add_scalar("hyperparmeters/ROOT_POLICY_TEMP", params[best].root_policy_temp, self.model_iter)
        # self.writer.add_scalar("hyperparmeters/ROOT_NOISE_FRAC", params[best].root_noise_frac, self.model_iter)
    
    @_set_state(TrainState.ROUND_ROBIN)
    def roundRobin(self, iteration, wins, draws, gamesBy):
        print('PERFORMING ROUND ROBIN ANALYSIS')
        print()

        if (wins is None) or (draws is None) or (gamesBy is None):
            wins    = np.zeros(self.numNets)
            draws   = np.zeros(self.numNets)
            gamesBy = np.zeros(self.numNets)
            
            #cls = MCTSPlayer if self.args.arenaMCTS else NNPlayer
            #allPlayers = [cls(net, self.game_cls, self.args) for net in self.train_nets]
            numPlayers = self.game_cls.num_players()
            #toProcess  = list(itertools.product(list(range(0, self.numNets)), repeat = numPlayers))
            numPer     = self.args.roundRobinGames
            
            temp = self.args.gamesPerIteration, self.args.process_batch_size
            
            self.args.gamesPerIteration = self.args.roundRobinGames * (self.numNets**self.game_cls.num_players())
            self.args.process_batch_size = 48

            for i in range(self.args.workers):
                self.games_for_agent.append(self.gamesFor(i, self.args.workers, ModeOfGameGen.CROSS_PRODUCT))
                print(self.games_for_agent[i])

            self.generateSelfPlayAgents(exact=True)
            self.processSelfPlayBatches(self.model_iter)

            dat = self.processGameResults(self.model_iter)
            self.killSelfPlayAgents()

            self.args.gamesPerIteration, self.args.process_batch_size = temp
            wins, draws, gamesBy = dat[0], dat[1], dat[2]

        totalWins    = wins
        totalGamesBy = gamesBy
        numPlayers   = self.game_cls.num_players()

        if self.args.use_draws_for_winrate:
            totalWins += draws/numPlayers
            totalGamesBy += draws

        totalWinsProportion =  totalWins/ totalGamesBy

        ranking = np.flip(np.argsort(totalWinsProportion))

        self.args.bestNet = ranking[0]

        numReplace = round(self.args.percentageKilled * self.numNets)
        
        for replacerIndex in range(0, numReplace):
            replacingIndex = self.numNets - replacerIndex - 1

            replacerNet  = ranking[replacerIndex]
            replacingNet = ranking[replacingIndex]

            print(f'REPLACING {replacingNet} WITH {replacerNet}')
            argsToUpdate = self.get_trainable_attributes(self.train_nets[replacerNet].args)
            for key in argsToUpdate:
                argsToUpdate[key] = argsToUpdate[key] * np.random.uniform(1-self.args.deviation, 1+self.args.deviation)

            
            self._load_model(self.train_nets[replacingNet], iteration, replacerNet)
            self.train_nets[replacingNet].args.update(argsToUpdate)


            if self.args.model_gating:
                self.self_play_iter[replacingNet] = self.self_play_iter[replacerNet]
                self._load_model(self.self_play_nets[replacingNet], self.self_play_iter[replacingNet], replacerNet)
                self.self_play_nets[replacingNet].args.update(argsToUpdate)

        print('NEW ARGUEMENTS ARE : ')
        for net in range(self.numNets):
            self.train_nets[net].args.bestNet = self.args.bestNet
            print(f"{net} : {self.get_trainable_attributes(self.train_nets[net].args)}")

        print(f'NEW BEST NET IS {self.args.bestNet}')
        
    @_set_state(TrainState.COMPARE_PAST)
    def compareToPast(self, model_iter, player, usePreLoad = False,preLoadedWins = None, preLoadedDraws = None):
        print(f'PITTING P{player} AGAINST ITERATION {self.self_play_iter[player]}')
        if not usePreLoad:
            self._load_model(self.self_play_nets[player], self.self_play_iter[player], player)

            # if self.args.arenaBatched:
            #     if not self.args.arenaMCTS:
            #         self.args.arenaMCTS = True
            #         print('WARNING: Batched arena comparison is enabled which uses MCTS, but arena MCTS is set to False.'
            #                           ' Ignoring this, and continuing with batched MCTS in arena.')

            #     nplayer = self.train_net.process
            #     pplayer = self.self_play_net.process
            # else:
            #     cls = MCTSPlayer if self.args.arenaMCTS else NNPlayer
            #     nplayer = cls(self.game_cls, self.args, self.train_net)
            #     pplayer = cls(self.game_cls, self.args, self.self_play_net)
            
            cls = MCTSPlayer if self.args.arenaMCTS else NNPlayer
            nplayer = cls(self.train_nets[player], self.game_cls, self.args)
            pplayer = cls(self.self_play_nets[player], self.game_cls, self.args)

            players = [nplayer] + [pplayer] * (self.game_cls.num_players() - 1)
            self.arena = Arena(players, self.game_cls, use_batched_mcts=self.args.arenaBatched, args=self.args)
            wins, draws, winrates = self.arena.play_games(self.args.arenaCompare)
            if self.stop_train.is_set(): return
            winrate = winrates[0]
        else:
            wins    = preLoadedWins[1], preLoadedWins[0]
            draws   = preLoadedDraws
            winrate = wins[0]/(wins[1] + wins[0])

        print(f'NEW/PAST WINS FOR {player} : {wins[0]} / {sum(wins[1:])} ; DRAWS : {draws}\n')
        print(f'NEW MODEL WINRATE {player} : {round(winrate, 3)}')
        if player == self.args.bestNet:
            self.writer.add_scalar('win_rate/past', winrate, model_iter)

        ### Model gating ###
        if (
            self.args.model_gating
            and winrate < self.args.min_next_model_winrate
            and (self.args.max_gating_iters is None
                 or self.gating_counter < self.args.max_gating_iters)
        ):
            self.gating_counter[player] += 1
        elif self.args.model_gating:
            print("No Gating")
            self.self_play_iter[player] = model_iter
            self._load_model(self.self_play_nets[player], self.self_play_iter[player], player)
            self.gating_counter[player] = 0

        if self.args.model_gating:
            print(f'Using model version {self.self_play_iter[player]} for P{player} self play.')

    @_set_state(TrainState.COMPARE_BASELINE)
    def compareToBaseline(self, iteration, player):
        test_player = self.args.baselineTester(self.game_cls, self.args)
        can_process = test_player.supports_process() and self.args.arenaBatched

        print()

        nnplayer = (MCTSPlayer if self.args.arenaMCTS else NNPlayer)(self.train_nets[player], self.game_cls, self.args)

        print(f'PITTING AGAINST BASELINE: ' + self.args.baselineTester.__name__)

        players = [nnplayer] + [test_player] * (self.game_cls.num_players() - 1)
        self.arena = Arena(players, self.game_cls, use_batched_mcts=self.args.arenaBatched, args=self.args)
        wins, draws, winrates = self.arena.play_games(self.args.arenaCompare)
        if self.stop_train.is_set(): return
        winrate = winrates[0]

        print(f'NEW/BASELINE WINS FOR : {wins[0]} / {sum(wins[1:])} ; DRAWS : {draws}\n')
        print(f'NEW MODEL WINRATE FOR : {round(winrate, 3)}')
        self.writer.add_scalar('win_rate/baseline', winrate, iteration)
