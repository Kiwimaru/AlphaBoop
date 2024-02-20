# cython: language_level=3

import torch.multiprocessing as mp
import numpy as np
import torch
import traceback
import itertools
import time

from alphazero.MCTS import MCTS
from torch_geometric.data import Data



class SelfPlayAgent(mp.Process):
    def __init__(self, id, whoVwhoComputing, game_cls, ready_queue, batch_ready, batch_tensor, policy_tensor,
                 value_tensor, output_queue, result_queue, complete_count, games_played,
                 stop_event: mp.Event, pause_event: mp.Event(), argss, *args, _is_arena=False, _is_warmup=False, _exact_game_count=False):
        super().__init__()
        #print(_is_arena)
        self.id = id
        self.numPer = whoVwhoComputing[0]
        self.listToCompute = whoVwhoComputing[1]
        self.numStarted = 0
        self.whichPlayerNext = 0

        self.game_cls = game_cls
        self.numPlayers = game_cls.num_players()
        self.ready_queue = ready_queue
        self.batch_ready = batch_ready
        self.batch_tensor = batch_tensor
        if args:
            self.batch_tensor2 = args[0]
        #self.batch_queue = batch_queue
        
        #if _is_arena:
        self.batch_size = policy_tensor.shape[0]
        #else:
        #    self.batch_size = self.batch_tensor.shape[0]
        
        self.policy_tensor = policy_tensor
        self.value_tensor = value_tensor
        self.output_queue = output_queue
        self.result_queue = result_queue
        self.games = []
        self.histories = []
        self.temps = []
        self.next_reset = []
        self.mcts = []
        self.netsGoing = [0 for _ in range(0, len(self.listToCompute))]
        self.games_played = games_played
        self.complete_count = complete_count
        self.stop_event = stop_event
        self.pause_event = pause_event
        self.args = argss
        #print(self.args)

        self._is_arena         = _is_arena
        self._is_warmup        = _is_warmup
        self._exact_game_count = _exact_game_count
        
        if _is_arena:
            self.player_to_index = list(range(game_cls.num_players()))
            np.random.shuffle(self.player_to_index)
            self.batch_indices = None
        if _is_warmup:
            action_size = game_cls.action_size()
            self._WARMUP_POLICY = torch.full((action_size,), 1 / action_size).to(policy_tensor.device)
            value_size = game_cls.num_players() + game_cls.has_draw()
            self._WARMUP_VALUE = torch.full((value_size,), 1 / value_size).to(policy_tensor.device)
        self.fast = False

        for _ in range(self.batch_size):
            self.addNextToCompute()
        #print(len(self.netsGoing[0]))

    # Each net must have it's own MCTS
    #  extra machinery is to ensure that each net has exactly 1 MCTS
    def _get_mcts(self, _next_game_type = None):
        ret = [MCTS(self.args) for _ in range(self.numPlayers)]
        players = []
        p = 0
        if _next_game_type != None and len(_next_game_type) == self.numPlayers:
            for net in range(self.numPlayers):
                if _next_game_type[net] in _next_game_type[:net]:
                    players.append(_next_game_type.index(_next_game_type[net]))
                else:
                    players.append(p)
                    p += 1
        else:
            players = list(range(self.numPlayers))
            p = self.numPlayers
            
        return (players, ret[:p])

    def _mcts(self, index: int) -> MCTS:
        playersToNet, mcts = self.mcts[index]
        if self._is_arena:
            return mcts[self.games[index].player]
        else:
            return mcts[playersToNet[self.whichPlayerNext]]

    def _check_pause(self):
        while self.pause_event.is_set():
            time.sleep(.1)

    def addNextToCompute(self):
        netToAdd = (self.numStarted//self.numPer)
        if netToAdd>= len(self.listToCompute):
            if self._exact_game_count:
                return
            self.netsGoing += [0 for _ in range(len(self.listToCompute))]
            self.listToCompute *= 2

        self.netsGoing[netToAdd] +=1
        self.games.append(self.game_cls())
        self.histories.append([])
        self.temps.append(self.args.startTemp)
        self.next_reset.append(0)
        if not(self._is_arena):
            self.mcts.append(self._get_mcts(self.listToCompute[netToAdd]))
        else:
            self.mcts.append(self._get_mcts())

        self.numStarted +=1

    def getNetIndex(self, num):
        cumulative = 0
        for i in range(0,len(self.netsGoing)):
            cumulative += self.netsGoing[i]
            if cumulative > num:
                return i
        return len(self.netsGoing)

    # Returns an list containing (net, number) pairs indicating the order and number
    #   of games in a row which can be batched - should not be the case that 
    #   (net, a) follows (net, b) for any net, a or b
    def getNetNums(self):
        ret = []
        for i in range(len(self.listToCompute)):
            game     = self.listToCompute[i]
            numGoing = self.netsGoing[i]
            
            if numGoing == 0:
                continue;

            if len(ret) > 0 and game[self.whichPlayerNext] == ret[-1][0]:
                ret[-1] = (ret[-1][0], ret[-1][1] + numGoing)
            else:
                ret.append((game[self.whichPlayerNext], numGoing))
        return ret

    def removeFromComputing(self, index):
        i = self.getNetIndex(index)
        self.netsGoing[i] -= 1
        
        del (self.games[index])
        del (self.histories[index])
        del (self.mcts[index])
        del (self.temps[index])
        del (self.next_reset[index])

    def run(self):
        try:
            np.random.seed()
            while not self.stop_event.is_set() and self.games_played.value < self.args.gamesPerIteration and not(self._exact_game_count and len(self.games) == 0):
                self._check_pause()
                self.fast = np.random.random_sample() < self.args.probFastSim
                sims = self.args.numFastSims if self.fast else self.args.numMCTSSims \
                    if not self._is_warmup else self.args.numWarmupSims
                for _ in range(sims):
                    if self.stop_event.is_set(): break
                    self.generateBatch()
                    if self.stop_event.is_set(): break
                    self.processBatch()

                #print("readytodorealmove")

                if self.stop_event.is_set(): break
                self.playMoves()
                self.whichPlayerNext = (self.whichPlayerNext + 1)%self.numPlayers

            with self.complete_count.get_lock():
                self.complete_count.value += 1
            if not self._is_arena:
                self.output_queue.close()
                self.output_queue.join_thread()
        except Exception:
            print(traceback.format_exc())

    def generateBatch(self):
        if self._is_arena:
            batch_tensor = [[] for _ in range(self.game_cls.num_players())]
            self.batch_indices = [[] for _ in range(self.game_cls.num_players())]

        for i in range(len(self.games)):
            self._check_pause()
            state = self._mcts(i).find_leaf(self.games[i])
            if self._is_warmup:
                self.policy_tensor[i].copy_(self._WARMUP_POLICY)
                self.value_tensor[i].copy_(self._WARMUP_VALUE)
                continue
            if self.args.nnet_type != "graphnet":
                data = torch.from_numpy(state.observation())
            else:
                data = state.observation()

            if self._is_arena:
                if self.args.nnet_type != "graphnet":
                    data = data.view(-1, *state.observation_size())
                player = self.player_to_index[self.games[i].player]
                batch_tensor[player].append(data)
                self.batch_indices[player].append(i)
            else:
                if self.args.nnet_type != "graphnet":
                    self.batch_tensor[i].copy_(data)
                else:
                    self.batch_tensor[i].copy_(data.x)
                    self.batch_tensor2[i].copy_(data.edge_index)
                    #print(data)
                    #print(self.batch_tensor[i])
                    #data.share_memory_()
                    #self.batch_tensor[i].x = (data.x)
                    #self.batch_tensor[i].edge_index = (data.edge_index)

                #batch_tensor.append(data)
                #self.batch_tensor[i].copy_(data)

        if self._is_arena:
            for player in range(self.game_cls.num_players()):
                player = self.player_to_index[player]
                data = batch_tensor[player]
                if self.args.nnet_type != "graphnet" and data != []:
                    data = torch.cat(data)
                if data != []:
                    batch_tensor[player] = data
            self.output_queue.put(batch_tensor)
            self.batch_indices = list(itertools.chain.from_iterable(self.batch_indices))
        #elif self.args.nnet_type == "graphnet":
            #print(self.batch_queue)
            #self.batch_queue.put(self.batch_tensor)
        #else:
        #    if self.args.nnet_type != "graphnet":
        #        self.batch_tensor.copy_(torch.cat(batch_tensor))#self.batch_queue.put(torch.cat(batch_tensor))
        #    else:
                #print(batch_tensor)
        #        self.batch_tensor.update(Batch.from_data_list(batch_tensor))


        if not self._is_warmup:
            #print("ready")
            # ID, the list of what nets and how many are found in the data sent so to best batch
            self.ready_queue.put((self.id, self.getNetNums()))

    def processBatch(self):
        if not self._is_warmup:
            self.batch_ready.wait()
            self.batch_ready.clear()

        for i in range(len(self.games)):
            self._check_pause()
            index = self.batch_indices[i] if self._is_arena else i
            self._mcts(i).process_results(
                self.games[i],
                self.value_tensor[index].data.numpy(),
                self.policy_tensor[index].data.numpy(),
                # No adding root noise or temp when no doing a fast itteration 
                #  as we want the best move from out network
                False if self._is_arena or self.fast else self.args.add_root_noise,
                False if self._is_arena or self.fast else self.args.add_root_temp
            )

    def playMoves(self):
        toRem = []

        for i in range(len(self.games)):
            self._check_pause()
            self.temps[i] = self.args.temp_scaling_fn(
                self.temps[i], self.games[i].turns, self.game_cls.max_turns()
            ) if not self._is_arena else self.args.arenaTemp
            #print(self.temps[i])
            #print()
            #print(self._mcts(i), self.games[i], self.temps[i])
            policy = self._mcts(i).probs(self.games[i], self.temps[i])
            action = np.random.choice(self.games[i].action_size(), p=policy)
            if not self.fast and not self._is_arena:
                self.histories[i].append((
                    self.games[i].clone(),
                    self._mcts(i).probs(self.games[i])
                ))

            _ = [mcts.update_root(self.games[i], action) for mcts in self.mcts[i][1]]

            self.games[i].play_action(action)
            if self.args.mctsResetThreshold and self.games[i].turns >= self.next_reset[i]:
                self.mcts[i] = self._get_mcts()
                self.next_reset[i] = self.games[i].turns + self.args.mctsResetThreshold

            winstate = self.games[i].win_state()
            #print(winstate)
            if winstate.any():
                #print(i)
                self.result_queue.put((self.games[i].clone(), winstate, self.id, self.listToCompute[self.getNetIndex(i)]))
                lock = self.games_played.get_lock()
                lock.acquire()
                if self.games_played.value < self.args.gamesPerIteration:
                    self.games_played.value += 1
                    lock.release()
                    if not self._is_arena:
                        for hist in self.histories[i]:
                            self._check_pause()
                            if self.args.symmetricSamples:
                                data = hist[0].symmetries(hist[1], winstate)
                            else:
                                data = ((hist[0], hist[1], winstate),)

                            for state, pi, true_winstate in data:
                                self._check_pause()
                                self.output_queue.put((
                                    state.observation(), pi, np.array(true_winstate, dtype=np.float32)
                                ))

                    toRem.append(i)
                else:
                    lock.release()

        toRem.reverse()
        for i in toRem:
            self.removeFromComputing(i)

        # To ensure that all games are played in the correct order (i.e with the first player 
        #  in the tuple acting first)
        if self.whichPlayerNext == 0 or self._is_arena:
            for _ in range(self.batch_size - len(self.games)):
                self.addNextToCompute()
