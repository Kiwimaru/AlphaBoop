import numpy, pyximport

pyximport.install(setup_args={'include_dirs': numpy.get_include()})

from alphazero.Arena import Arena
from alphazero.GenericPlayers import *
from alphazero.NNetWrapper import NNetWrapper as NNet

# from alphazero.envs.miniBoop.miniBoop import Game as Game
# from alphazero.envs.miniBoop.train import args

from alphazero.envs.alphaboop.alphaboop import Game
from alphazero.envs.alphaboop.alphaboop import display as displayGame

from alphazero.envs.alphaboop.train import args

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""


def calculateSingleMove():
    pieces = np.array([[ 8, 8, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 0]])

    grey = 0
    orange = 1

    g = Game()
    g.set_board(pieces, grey)

    displayGame(g)
    args.numMCTSSims = 6400
    nn1 = NNet(Game, args)
    nn1.load_checkpoint('', 'AlphaBoop-109.pkl')

    alphaBoop = MCTSPlayer(nn1, args=args, print_policy=True)
    print("Action: " + str(alphaBoop.play(g)))

if __name__ == '__main__':
    args.numMCTSSims = 1600
    # args.arena_batch_size = 64
    calculateSingleMove()

    # # nnet players
    # nn1 = NNet(Game, args)
    # nn1.load_checkpoint('', 'AlphaMiniBoop.pkl')
    # # player1 = nn1.process
    # # policy, value = nn1.predict(np.array([player1, player2, colour, turn], dtype=np.float32))
    # # print(policy)
    #
    # alphaBoop = MCTSPlayer(nn1, args=args)  # , print_policy=True)
    #
    # human = HumanMiniBoopPlayer()
    #
    # # Human goes first
    # players = [alphaBoop, human]
    #
    # # Bot goes first
    # # players = [human, alphaBoop]
    #
    # arena = Arena(players, Game, use_batched_mcts=False, args=args, display=print)
    # wins, draws, winrates = arena.play_games(1, verbose=True)
    # for i in range(len(wins)):
    #     print(f'player{i+1}:\n\twins: {wins[i]}\n\twin rate: {winrates[i]}')
    # print('draws: ', draws)

    #
    # nn1 = NNet(Game, args)
    # nn1.load_checkpoint('', 'AlphaBoop-109.pkl')
    #
    # nn2 = NNet(Game, args)
    # nn2.load_checkpoint('', 'AlphaBoop-109.pkl')
    #
    # alphaBoop = MCTSPlayer(nn1, args=args)  # , print_policy=True)
    # alphaBoop2 = MCTSPlayer(nn2, args=args)  # , print_policy=True)
    #
    # human = HumanMiniBoopPlayer()
    #
    # # # Human goes first
    # players = [alphaBoop2, human]
    #
    # # Bot goes first
    # # players = [human, alphaBoop2]
    #
    # arena = Arena(players, Game, use_batched_mcts=False, args=args, display=displayGame)
    # wins, draws, winrates = arena.play_games(1, verbose=True)
    # for i in range(len(wins)):
    #     print(f'player{i + 1}:\n\twins: {wins[i]}\n\twin rate: {winrates[i]}')
    # print('draws: ', draws)
