import pyximport; pyximport.install()

from torch import multiprocessing as mp

from alphazero.Coach import Coach, get_args
from alphazero.NNetWrapper import NNetWrapper as nn
from alphazero.envs.alphaboop.alphaboop import Game
from alphazero.GenericPlayers import RawMCTSPlayer
from alphazero.utils import dotdict

args = get_args(dotdict({
    'run_name': 'boop_fpu',
    'workers': 7,
    'startIter': 1,
    'numIters': 1000,
    'numWarmupIters': 1,
    'process_batch_size': 512,
    'train_batch_size': 512,
    # should preferably be a multiple of process_batch_size and workers
    'gamesPerIteration': 512 * 4,
    'symmetricSamples': True,
    'skipSelfPlayIters': None,
    'selfPlayModelIter': None,
    'numMCTSSims': 300,
    'numFastSims': 40,
    'probFastSim': 0.75,
    'compareWithBaseline': True,
    'arenaCompare': 32 * 4,
    'arena_batch_size': 128,
    'arenaTemp': 1,
    'arenaMCTS': True,
    'baselineCompareFreq': 10,
    'compareWithPast': False, #elo caclulation enabled to this is not needed
    'pastCompareFreq': 10,
    'cpuct': 2.75,
    'fpu_reduction': 0.4,
    'load_model': True,
    'root_policy_temp': 1.3,
    'root_noise_frac': 0.3,
    "_num_players":2,
    #Elo
    'eloMCTS': 25,
    'eloGames':10,
    'eloMatches':10,
    'calculateElo': True
}),
    model_gating=False,
    max_gating_iters=None,
    max_moves=100,

    lr=0.02,
    num_channels=128,
    depth=10,
    value_head_channels=64,
    policy_head_channels=64,
    value_dense_layers=[2048, 1024, 512],
    policy_dense_layers=[2048, 1024, 512]
)
args.scheduler_args.milestones = [75, 150]


if __name__ == "__main__":
    nnet = nn(Game, args)
    c = Coach(Game, nnet, args)
    c.learn()
