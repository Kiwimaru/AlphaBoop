import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as geo_nn
import torch
import torch_geometric as geo_torch

from alphazero.Game import GameState
from alphazero.utils import dotdict
import numpy as np

# 1x1 convolution
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=False)

# 3*3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

# fully connected layers
def mlp(
    input_size: int,
    layer_sizes: list,
    output_size: int,
    output_activation=nn.Identity,
    activation=nn.ELU,
    dropout=None,
    batchnorm=False,
):
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
        # Only apply dropout and batchnorm after middle layers
        if i != len(sizes) - 2:
            if dropout != None:
                layers += [nn.Dropout(dropout, False)]
            if batchnorm:
                layers += [geo_torch.nn.norm.GraphNorm(in_channels=sizes[i+1])]
    return nn.Sequential(*layers)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResidualBlock, self).__init__()

        stride = 1
        if downsample:
            stride = 2
            self.conv_ds = conv1x1(in_channels, out_channels, stride)
            self.bn_ds = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)


    def forward(self, x):
        residual = x
        out = x
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample:
            residual = self.conv_ds(x)
            residual = self.bn_ds(residual)
        out += residual
        return out


class ResNet(nn.Module):
    def __init__(self, game_cls: GameState, args: dotdict):
        super(ResNet, self).__init__()
        # game params
        self.channels, self.board_x, self.board_y = game_cls.observation_size()
        self.action_size = game_cls.action_size()
        self.vst         = args.value_softmax_temperature
        self.pst         = args.policy_softmax_temperature

        self.conv1 = conv3x3(self.channels, args.num_channels)
        self.bn1 = nn.BatchNorm2d(args.num_channels)

        self.res_layers = []
        for _ in range(args.depth):
            self.res_layers.append(
                ResidualBlock(args.num_channels, args.num_channels)
            )
        self.resnet = nn.Sequential(*self.res_layers)

        self.v_conv = conv1x1(args.num_channels, args.value_head_channels)
        self.v_bn = nn.BatchNorm2d(args.value_head_channels)
        self.v_fc = mlp(
            self.board_x*self.board_y*args.value_head_channels,
            args.value_dense_layers,
            game_cls.num_players() + game_cls.has_draw(),
            activation=nn.Identity
        )

        self.pi_conv = conv1x1(args.num_channels, args.policy_head_channels)
        self.pi_bn = nn.BatchNorm2d(args.policy_head_channels)
        self.pi_fc = mlp(
            self.board_x*self.board_y*args.policy_head_channels,
            args.policy_dense_layers,
            self.action_size,
            activation=nn.Identity
        )

    def forward(self, s, _):
        # s: batch_size x num_channels x board_x x board_y
        s = s.view(-1, self.channels, self.board_x, self.board_y)
        s = F.relu(self.bn1(self.conv1(s)))
        s = self.resnet(s)

        v = self.v_conv(s)
        v = self.v_bn(v)
        v = torch.flatten(v, 1)
        v = self.v_fc(v)

        pi = self.pi_conv(s)
        pi = self.pi_bn(pi)
        pi = torch.flatten(pi, 1)
        pi = self.pi_fc(pi)

        return F.log_softmax(pi/self.pst, dim=1), F.log_softmax(v/self.vst, dim=1)


class FullyConnected(nn.Module):
    """
    Fully connected network which operates in the same way as NNetArchitecture.
    The fully_connected function is used to create the network, as well as the
    policy and value heads. Forward method returns log_softmax of policy and value head.
    """
    def __init__(self, game_cls: GameState, args: dotdict):
        super(FullyConnected, self).__init__()
        # get input size
        self.input_size = np.prod(game_cls.observation_size())
        self.vst        = args.value_softmax_temperature
        self.pst        = args.policy_softmax_temperature

        self.input_fc = mlp(
            self.input_size,
            args.input_fc_layers,
            args.input_fc_layers[-1],
            activation=nn.ReLU
        )
        self.v_fc = mlp(
            args.input_fc_layers[-1],
            args.value_dense_layers,
            game_cls.num_players() + game_cls.has_draw(),
            activation=nn.Identity
        )
        self.pi_fc = mlp(
            args.input_fc_layers[-1],
            args.policy_dense_layers,
            game_cls.action_size(),
            activation=nn.Identity
        )

    def forward(self, s, _):
        # s: batch_size x num_channels x board_x x board_y
        # reshape s for input_fc
        s = s.view(-1, self.input_size)
        s = self.input_fc(s)
        v = self.v_fc(s)
        pi = self.pi_fc(s)

        return F.log_softmax(pi/self.pst, dim=1), F.log_softmax(v/self.vst, dim=1)

class GraphNet(nn.Module):
    def __init__(self, game_cls: GameState, args: dotdict):
        super(GraphNet, self).__init__()
        assert args.depth > 0, "Can't make Graph Neural Network with 0 or less layers"
        assert args.value_head_channels == args.policy_head_channels
        

        self.channels        = game_cls.observation_size()[0]
        self.vst             = args.value_softmax_temperature
        self.pst             = args.policy_softmax_temperature
        
        # GIN(in_channels: int, hidden_channels: int, num_layers: int, 
        #    out_channels: Optional[int] = None, dropout: float = 0.0, 
        #    act: Optional[Union[str, Callable]] = 'relu', act_first: bool = False, 
        #    act_kwargs: Optional[Dict[str, Any]] = None, norm: Optional[Union[str, Callable]] = None, 
        #    norm_kwargs: Optional[Dict[str, Any]] = None, jk: Optional[str] = None, **kwargs
        # This is all the way up to 
        norm = geo_torch.nn.norm.BatchNorm(args.num_channels)
        self.GNNLayer  = geo_nn.GIN(self.channels, args.num_channels, args.depth, 
            out_channels = args.depth*args.num_channels, jk = "cat", norm = norm)
        
        self.CatLayers = geo_nn.JumpingKnowledge("cat")
        # This is the layer directly before the FC-v and FC-pi layers
        self.LinearLayer = mlp(self.channels + args.depth*args.num_channels,
            args.middle_layers, args.policy_head_channels, activation=nn.ReLU, dropout=None,
            batchnorm=True)

        self.v_fc = mlp(
            args.value_head_channels,
            args.value_dense_layers,
            game_cls.num_players() + game_cls.has_draw(),
            activation=nn.Identity
        )
        self.v_mean = geo_nn.pool.global_mean_pool

        self.pi_fc = mlp(
            args.policy_head_channels,
            args.policy_dense_layers,
            1,
            activation=nn.Identity
        )

    # Takes a torch_geometric.data.Data Object
    # if it is a batch must be given as a torch_geometric.data.Batch
    def forward(self, data, batch_size):
        #print(x)
        #print(edge_index)

        x = self.GNNLayer(data.x, data.edge_index)
        x = self.CatLayers([data.x,x])
        #if batch_size == 1:
        #    print(x)
        # If batching reshape x so that the the different graphs are seperate
        x_size = list(x.size())
        assert x_size[0] % batch_size == 0, f"Something has gone very wrong with batching {batch_size} graphs were inputed but the output contained a number of nodes that was not divisible by that; could be caused by inputing graphs of different sizes"
        x = x.view(*([batch_size, x_size[0]//batch_size]+x_size[1:]))

        #print(x.size())

        x = self.LinearLayer(x)

        v = self.v_fc(x)
        v = self.v_mean(v, batch=None)
        v = v.view(batch_size, -1)

        pi = self.pi_fc(x)
        pi = pi.view(batch_size, -1)

        return F.log_softmax(pi/self.pst, dim=1), F.log_softmax(v/self.vst, dim=1)