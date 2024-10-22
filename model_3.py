import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import add_self_loops, coalesce
from torch_scatter import scatter_max, scatter_add
from math import sqrt, log
from typing import Optional
from argparse import ArgumentParser
from test_tube import HyperOptArgumentParser
import random
import os
import time
import numpy as np
import torch_sparse.tensor

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from util import gumbel_softmax, spmm, scatter

import torch.optim as optim

# from torchrl.data import ReplayBuffer, ListStorage
# from utils import plot_learning_curve, create_directory

def add_model_args(parent_parser: Optional[ArgumentParser]=None, hyper: bool=False) -> ArgumentParser:
    if parent_parser is not None:
        parser = HyperOptArgumentParser(parents=[parent_parser], add_help=False, conflict_handler='resolve') if hyper else ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler='resolve')
    else:
        parser = HyperOptArgumentParser( add_help=False, conflict_handler='resolve') if hyper else ArgumentParser(add_help=False, conflict_handler='resolve')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_units', type=int, default=64)
    parser.add_argument('--use_aux_loss', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_agents', type=int, default=4)
    parser.add_argument('--num_steps', type=int, default=32)
    parser.add_argument('--reduce', type=str, default='sum', help="Options are ['sum', 'mean', 'max', 'log', 'sqrt']")
    parser.add_argument('--epochs', type=int, default=350)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--self_loops', action='store_true', default=False)
    parser.add_argument('--node_readout', action='store_true', default=False)
    parser.add_argument('--use_step_readout_lin', action='store_true', default=False)
    parser.add_argument('--gumbel_temp', type=float, default=1.0)
    parser.add_argument('--gumbel_min_temp', type=float, default=1.0/16)
    parser.add_argument('--gumbel_warmup', type=int, default=-1)
    parser.add_argument('--gumbel_decay_epochs', type=int, default=100)
    parser.add_argument('--min_lr_mult', type=float, default=1e-7)
    parser.add_argument('--weight_decay', type=float, default=1e-8)
    parser.add_argument('--num_pos_attention_heads', type=int, default=1)
    parser.add_argument('--clip_grad', type=float, default=-1.0)
    parser.add_argument('--readout_mlp', action='store_true', default=False)
    parser.add_argument('--post_ln', action='store_true', default=False)
    parser.add_argument('--attn_dropout', type=float, default=0.0)
    parser.add_argument('--no_time_cond', action='store_true', default=False)
    parser.add_argument('--mlp_width_mult', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='leaky_relu')
    parser.add_argument('--negative_slope', type=float, default=0.01)
    parser.add_argument('--input_mlp', action='store_true', default=False)
    parser.add_argument('--attn_width_mult', type=int, default=1)
    parser.add_argument('--importance_init', action='store_true', default=False)
    parser.add_argument('--random_agent', action='store_true', default=False)
    parser.add_argument('--test_argmax', action='store_true', default=False)
    parser.add_argument('--global_agent_pool', action='store_true', default=False)
    parser.add_argument('--agent_global_extra', action='store_true', default=False)
    parser.add_argument('--basic_global_agent', action='store_true', default=False)
    parser.add_argument('--basic_agent', action='store_true', default=False)
    parser.add_argument('--bias_attention', action='store_true', default=False)
    parser.add_argument('--visited_decay', type=float, default=0.9)
    parser.add_argument('--sparse_conv', action='store_true', default=False)
    parser.add_argument('--mean_pool_only', action='store_true', default=False)
    parser.add_argument('--edge_negative_slope', type=float, default=0.2)
    parser.add_argument('--final_readout_only', action='store_true', default=False)

    return parser

class TimeEmbedding(nn.Module):
    # https://github.com/w86763777/pytorch-ddpm/blob/master/model.py
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb

# Define the DQN model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DeepQNetwork(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim):
        super(DeepQNetwork, self).__init__()
 
        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.q = nn.Linear(fc2_dim, action_dim)
 
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(device)
 
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
 
        q = self.q(x)
 
        return q
 
    def save_checkpoint(self, epoch, checkpoint_file):
        checkpoint = {'epoch': epoch,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()}
        torch.save(checkpoint, checkpoint_file)
 
    def load_checkpoint(self, checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
 
 
class DQN:
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim,
                 gamma=0.99, tau=0.005, epsilon=1.0, eps_end=0.01, eps_dec=5e-4,
                 max_size=1000000, batch_size=256):
        self.tau = tau
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.batch_size = batch_size
        self.action_space = [i for i in range(action_dim)]
        self.loss_dqn = []
        self.loss_buffer = []
        self.mean_loss = []
        self.total_reward = []
 
        self.q_eval = DeepQNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                   fc1_dim=fc1_dim, fc2_dim=fc2_dim)
        self.q_target = DeepQNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                     fc1_dim=fc1_dim, fc2_dim=fc2_dim)
        self.memory = []
        # self.memory = ReplayBuffer(state_dim=state_dim, action_dim=action_dim,
        #                            max_size=max_size, batch_size=batch_size)
 
        self.update_network_parameters(tau=1.0)
 
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
 
        for q_target_params, q_eval_params in zip(self.q_target.parameters(), self.q_eval.parameters()):
            q_target_params.data.copy_(tau * q_eval_params + (1 - tau) * q_target_params)
 
    def choose_action(self, state):
        actions = self.q_eval.forward(state)
        return actions
    
    def sample_memory(self):
        if len(self.memory) > self.batch_size:
            sample =  random.sample(self.memory, self.batch_size)
        else:
            sample = self.memory
        q = torch.stack([x[0] for x in sample]).to(device)
        target = torch.stack([x[1] for x in sample]).detach()
        return q, target
    
    def learn(self, state, action, reward, state_, done):
        self.memory.append((state, action, reward, state_, done))
        # self.memory.store_transition(state, action, reward, state_, done)
 
        if len(self.memory) > self.batch_size:
            states, targets = self.sample_memory()
            self.q_eval.optimizer.zero_grad()
            loss = self.q_eval.loss(states, targets).to(device)
            loss.backward()
            self.q_eval.optimizer.step()
            self.loss_dqn.append(loss.item())
            self.update_network_parameters()
 
            self.epsilon = self.epsilon - self.eps_dec if self.epsilon > \
                self.eps_min else self.eps_min

    def save_models(self,  episode, checkpoint_path):
        self.q_eval.save_checkpoint(episode, checkpoint_path + 'DQN_Q_eval_{}.pth'.format(episode))
        self.q_target.save_checkpoint(episode, checkpoint_path + 'DQN_Q_target_{}.pth'.format(episode))
 

class AgentNet(nn.Module):
    def __init__(self, num_features, hidden_units, num_out_classes, dropout, num_steps, num_agents, reduce, node_readout, use_step_readout_lin, 
                num_pos_attention_heads, readout_mlp, self_loops, post_ln=False,  attn_dropout=0.0, no_time_cond=False, mlp_width_mult=1,
                activation_function='leaky_relu', negative_slope=0.01, input_mlp=False, attn_width_mult=1, importance_init=False, random_agent=False,
                test_argmax=False, global_agent_pool=False, agent_global_extra=False, basic_global_agent=False, basic_agent=False, bias_attention=False,
                visited_decay=0.9, sparse_conv=False, num_edge_features=0, mean_pool_only=False, edge_negative_slope=0.2, regression=False, final_readout_only=False,
                ogb_mol=False, qm9=False):
        super(AgentNet, self).__init__()

        self.dim = hidden_units
        self.node_dim = hidden_units * 2
        self.dropout = dropout
        self.num_steps = num_steps
        self.num_agents = num_agents
        self.reduce = reduce
        self.node_readout = node_readout
        self.use_step_readout_lin = use_step_readout_lin
        self.num_pos_attention_heads = num_pos_attention_heads
        self.readout_mlp = readout_mlp
        self.post_ln = post_ln
        self.attn_dropout = attn_dropout
        self.time_cond = not no_time_cond
        self.activation_function = activation_function
        self.negative_slope = negative_slope
        self.input_mlp = input_mlp
        self.test_argmax = test_argmax
        self.global_agent_pool = global_agent_pool
        self.agent_global_extra = agent_global_extra
        self.basic_global_agent = basic_global_agent
        self.basic_agent = basic_agent
        self.bias_attention = bias_attention
        self.visited_decay = visited_decay
        self.sparse_conv = sparse_conv
        self.mean_pool_only = mean_pool_only
        self.edge_negative_slope = edge_negative_slope
        self.final_readout_only = final_readout_only

        self.num_edge_features = num_edge_features

        self.self_loops = self_loops
        self.importance_init = importance_init

        self.random_agent = random_agent

        self.num_out_classes = num_out_classes

        self.regression = regression

        self.ogb_mol = ogb_mol
        self.qm9 = qm9

        self.temp = 2.0/3.0

        if self.activation_function == 'gelu':
            activation = nn.GELU() 
        elif self.activation_function == 'relu':
            activation = nn.ReLU() 
        else:
            activation = nn.LeakyReLU(negative_slope=self.negative_slope) # NOTE make negative_slope into a param

        # Have learnable global BSEU [back, stay, explored, unexplored] params
        if self.basic_global_agent or self.basic_agent or self.bias_attention:
            self.back_param = nn.Parameter(torch.tensor([1.0], requires_grad=True))
            self.stay_param = nn.Parameter(torch.tensor([1.0], requires_grad=True))
            self.explored_param = nn.Parameter(torch.tensor([1.0], requires_grad=True))
            self.unexplored_param = nn.Parameter(torch.tensor([1.0], requires_grad=True))
            if self.basic_agent:
                self.basic_agent_attn_mlp = nn.Sequential((nn.Identity() if self.post_ln else nn.LayerNorm(self.dim)), nn.Linear(self.dim, self.dim*mlp_width_mult), activation, nn.Dropout(self.dropout), nn.Linear(self.dim*mlp_width_mult, 4, bias=False))

        # Embed time step
        if self.time_cond:
            self.time_emb = TimeEmbedding(self.num_steps + 1, self.dim, self.dim * mlp_width_mult)

        # Input projection
        if self.input_mlp:
            self.input_proj = nn.Sequential(nn.Linear(num_features, self.dim*2), activation, nn.Linear(self.dim*2, self.dim))
        else:
            self.input_proj = nn.Sequential(nn.Linear(num_features, self.dim))

        if self.ogb_mol:
            self.bond_encoder = BondEncoder(self.dim)
            self.atom_encoder = AtomEncoder(self.dim)
            self.num_edge_features = self.dim
            edge_dim = self.dim
        elif self.qm9:
            self.edge_nn = nn.Sequential(nn.Linear(self.num_edge_features, self.dim*mlp_width_mult), activation, nn.Dropout(self.dropout), nn.Linear(self.dim*mlp_width_mult, self.dim * self.dim))
            edge_dim = self.num_edge_features
        elif self.num_edge_features > 0:
            self.edge_input_proj = nn.Sequential(nn.Linear(num_edge_features, self.dim*2), activation, nn.Linear(self.dim*2, self.dim))
            edge_dim = self.dim
        else:
            self.edge_input_proj = nn.Sequential(nn.Identity())
            edge_dim = 0

        self.node_mem_init = torch.nn.Parameter(torch.zeros(self.dim, requires_grad=True))

        # Agent embeddings
        self.agent_emb = nn.Embedding(self.num_agents, self.dim)

        # Attention params for first node position selection (if used)
        if self.importance_init:
            self.init_key = nn.Sequential((nn.Identity() if self.post_ln else nn.LayerNorm(self.dim)), nn.Linear(self.dim, self.dim*attn_width_mult*self.num_pos_attention_heads))
            self.init_query = nn.Sequential((nn.Identity() if self.post_ln else nn.LayerNorm(self.dim)), nn.Linear(self.dim, self.dim*attn_width_mult*self.num_pos_attention_heads))
            self.init_attn_lin = nn.Sequential(nn.Linear(self.num_pos_attention_heads, 1))

        # Layer norms
        if self.post_ln:
            self.agent_ln = nn.LayerNorm(self.dim)
            self.node_ln = nn.LayerNorm(self.dim)
            self.conv_ln = nn.LayerNorm(self.dim)
        else: 
            self.agent_ln = nn.Identity()
            self.node_ln = nn.Identity()
            self.conv_ln = nn.Identity()

        # Use global information exchange between agents (akin to a virtual node in GNNs)
        if self.global_agent_pool:
            extra_global_dim = self.dim
        else:
            extra_global_dim = 0

        # Node and agent update
        self.agent_node_edge_lin = None
        if edge_dim > 0:
            self.agent_node_ln = nn.Sequential((nn.Identity() if self.post_ln else nn.LayerNorm(self.dim+edge_dim)), nn.Linear(self.dim+edge_dim, self.dim), (nn.LeakyReLU(self.edge_negative_slope) if self.edge_negative_slope > 0 else nn.ReLU()))
        else:
            self.agent_node_ln = nn.Identity()
        self.message_val = nn.Sequential((nn.Identity() if self.post_ln else nn.LayerNorm(self.dim+edge_dim)), nn.Linear(self.dim+edge_dim, self.dim), (nn.LeakyReLU(self.edge_negative_slope) if self.edge_negative_slope > 0 else nn.ReLU()))
        self.conv_mlp = nn.Sequential((nn.Identity() if self.post_ln else nn.LayerNorm(self.dim*2)), nn.Linear(self.dim*2, self.dim*2 * mlp_width_mult), activation, nn.Dropout(self.dropout), nn.Linear(self.dim*2 * mlp_width_mult, self.dim), nn.Dropout(self.dropout))
        self.node_mlp = nn.Sequential(nn.Identity() if self.post_ln else (nn.LayerNorm(self.dim*2 + extra_global_dim)), nn.Linear(self.dim*2 + extra_global_dim, self.dim*2 * mlp_width_mult), activation, nn.Dropout(self.dropout), nn.Linear(self.dim*2 * mlp_width_mult, self.dim), nn.Dropout(self.dropout)) 
        self.agent_mlp = nn.Sequential((nn.Identity() if self.post_ln else nn.LayerNorm(self.dim*2 + edge_dim + (self.dim if self.agent_global_extra else 0))), nn.Linear(self.dim*2 + edge_dim + (self.dim if self.agent_global_extra else 0), self.dim*2 * mlp_width_mult), activation, nn.Dropout(self.dropout), nn.Linear(self.dim*2 * mlp_width_mult, self.dim), nn.Dropout(self.dropout)) 
        if self.use_step_readout_lin:
            self.step_readout_mlp = nn.Sequential((nn.Identity() if self.post_ln else nn.LayerNorm(self.dim+(0 if self.time_cond else 1))), nn.Linear(self.dim+(0 if self.time_cond else 1), self.dim*2 * mlp_width_mult), activation, nn.Dropout(self.dropout), nn.Linear(self.dim*2 * mlp_width_mult, self.dim*2), nn.Dropout(self.dropout))  
            out_dim = 2*self.dim
        else:
            self.step_readout_mlp = nn.Sequential((nn.Identity() if self.post_ln else nn.LayerNorm(self.dim+(0 if self.time_cond else 1))), nn.Linear(self.dim+(0 if self.time_cond else 1), self.dim*2), nn.Dropout(self.dropout)) 
            out_dim = 2*self.dim
        if self.global_agent_pool:
            self.global_agent_pool_mlp = nn.Sequential((nn.Identity() if self.post_ln else nn.LayerNorm(self.dim)), nn.Linear(self.dim, self.dim * mlp_width_mult), activation, nn.Dropout(self.dropout), nn.Linear(self.dim * mlp_width_mult, self.dim), nn.Dropout(self.dropout)) 
        self.final_mlp = nn.Sequential(nn.Linear(out_dim*2, out_dim*2), activation, nn.Linear(out_dim*2, self.num_out_classes)) 
        self.final_mlp_time = nn.Sequential(activation, nn.Linear(self.dim * mlp_width_mult, self.dim*2))

        # Add time emb projections
        if self.time_cond:
            self.node_mlp_time = nn.Sequential(activation, nn.Linear(self.dim * mlp_width_mult, self.dim*2 + extra_global_dim))
            self.agent_mlp_time = nn.Sequential(activation, nn.Linear(self.dim * mlp_width_mult, self.dim*2 + edge_dim + (self.dim if self.agent_global_extra else 0))) #  + extra_global_dim
            self.step_readout_mlp_time = nn.Sequential(activation, nn.Linear(self.dim * mlp_width_mult, self.dim))
            self.conv_mlp_time = nn.Sequential(activation, nn.Linear(self.dim * mlp_width_mult, self.dim*2))
            if self.global_agent_pool:
                self.global_agent_pool_mlp_time = nn.Sequential(activation, nn.Linear(self.dim * mlp_width_mult, self.dim))

        # Agent jump
        self.key = nn.Sequential((nn.Identity() if self.post_ln else nn.LayerNorm((self.dim)*2 + edge_dim)), nn.Linear((self.dim)*2 + edge_dim, self.dim*attn_width_mult*self.num_pos_attention_heads), nn.Identity())
        self.query = nn.Sequential((nn.Identity() if self.post_ln else nn.LayerNorm(self.dim)), nn.Linear(self.dim, self.dim*attn_width_mult*self.num_pos_attention_heads))
        self.attn_lin = nn.Sequential(nn.Linear(self.num_pos_attention_heads, 1))

        if self.node_readout:
            if self.readout_mlp:
                self.fc = nn.Sequential(nn.Linear(out_dim*(1 if self.mean_pool_only else 2) + self.dim, self.dim*2), activation, nn.Dropout(self.dropout), nn.Linear(self.dim*2, self.num_out_classes))
            else:
                self.fc = nn.Linear(out_dim*(1 if self.mean_pool_only else 2) + self.dim, self.num_out_classes)
        else:
            if self.readout_mlp:
                self.fc = nn.Sequential(nn.Linear(out_dim*(1 if self.mean_pool_only else 2), self.dim*2), activation, nn.Dropout(self.dropout), nn.Linear(self.dim*2, self.num_out_classes))
            else:
                self.fc = nn.Linear(out_dim*(1 if self.mean_pool_only else 2), self.num_out_classes)

          # define dqn model to choose action
        if self.ogb_mol:
            state_dim = self.dim*3 + 128
            dqn_batch_size = 64
        elif self.qm9:
            state_dim = self.dim*3 + 5
            dqn_batch_size = 32
        else:
            state_dim = self.dim*3
            dqn_batch_size = 32
        action_dim = 1

        self.dqn = DQN(alpha=0.0001, state_dim=state_dim, action_dim=action_dim,
                fc1_dim=128, fc2_dim=64, gamma=0.99, tau=0.005, epsilon=1.0,
                eps_end=0.05, eps_dec=5e-4, max_size=1000000, batch_size=dqn_batch_size)
        
        self.reset_parameters()

    def reset_parameters(self):
        # Have learnable global BSEU [back, stay, explored, unexplored] params
        if self.bias_attention: # Bias the parameters towards exploration
            nn.init.constant_(self.back_param, 0.0)
            nn.init.constant_(self.stay_param, -1.0)
            nn.init.constant_(self.explored_param, 0.0)
            nn.init.constant_(self.unexplored_param, 5.0)
        elif self.basic_global_agent or self.basic_agent:
            nn.init.constant_(self.back_param, 0.0)
            nn.init.constant_(self.stay_param, 0.0)
            nn.init.constant_(self.explored_param, 0.0)
            nn.init.constant_(self.unexplored_param, 0.0)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.activation_function == 'gelu':
                    nn.init.xavier_uniform_(m.weight)
                elif self.activation_function == 'relu':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                else:
                    nn.init.kaiming_uniform_(m.weight, a=self.negative_slope, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                m.reset_parameters()
            elif isinstance(m, nn.Embedding):
                m.reset_parameters()

    def save_checkpoint(self, epoch, optimizer, scheduler, checkpoint_file):
        checkpoint = {'epoch': epoch,
                        'agent_model': self.state_dict(),
                        'dqn_eval': self.dqn.q_eval.state_dict(),
                        'dqn_target': self.dqn.q_target.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()}
        torch.save(checkpoint, checkpoint_file)                     
    
    def load_checkpoint(self, checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, edge_feat: Optional[torch.Tensor] = None, fgs: Optional[torch.Tensor] = None,node_num: Optional[torch.Tensor] = None,cycles: Optional[torch.Tensor] = None, max_degree_node: Optional[torch.Tensor] = None):
        batch_size = (batch.max() + 1).item()
        num_nodes_batch = x.size(0)
        num_agents_batch = int(self.num_agents*batch_size)

        if self.time_cond:
            time_emb = self.time_emb(torch.zeros(1, device=x.device, dtype=torch.long))
        
        if self.ogb_mol:
            init_node_emb = self.atom_encoder(x)
        else:
            init_node_emb = self.input_proj(x)
        del x
        node_emb = init_node_emb

        if self.ogb_mol:
            edge_emb = self.bond_encoder(edge_feat)
        elif self.qm9:
            edge_emb = edge_feat
        elif edge_feat is not None:
            edge_emb = self.edge_input_proj(edge_feat)
        else:
            edge_emb = None

        # initialize dqn variables
        self.dqn.memory = []
        self.dqn.loss_dqn = []
        self.dqn.loss_buffer = []
        self.dqn.mean_loss = []
        self.dqn.total_reward = []
        comparison_old = None



        # Add self loops to let agent stay on a current node
        edge_index_sl = edge_index.clone()
        edge_emb_sl = edge_emb if edge_emb is None else edge_emb.clone()
        if self.self_loops:
            edge_index_sl, edge_emb_sl = add_self_loops(edge_index_sl, edge_attr=edge_emb_sl, num_nodes=num_nodes_batch)
        if edge_emb_sl is not None:
            edge_index_sl, edge_emb_sl = coalesce(edge_index_sl, edge_emb_sl, num_nodes_batch)
            edge_index, edge_emb = coalesce(edge_index, edge_emb, num_nodes_batch)
        else:
            edge_index_sl = coalesce(edge_index_sl, None, num_nodes_batch)
            edge_index = coalesce(edge_index, None, num_nodes_batch)

        # Set agent embeddings
        agent_emb = self.agent_emb(torch.arange(self.num_agents, device=edge_index.device)).unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, self.dim)

        # Set agent positions using attention
        if self.importance_init:
            agent_neighbour = torch.stack([
                                            torch.arange(num_agents_batch, device=batch.device).view(batch_size, self.num_agents).repeat_interleave(scatter_add(torch.ones_like(batch), batch), dim=0).view(-1),
                                            torch.arange(num_nodes_batch, device=batch.device).repeat_interleave(self.num_agents)
                                          ])
            # Move agents
            Q = self.init_query(agent_emb).reshape(agent_emb.size(0), self.num_pos_attention_heads, -1)
            K = self.init_key(init_node_emb.clone()).reshape(num_nodes_batch, self.num_pos_attention_heads, -1)
            attn_score = (Q[agent_neighbour[0]] * K[agent_neighbour[1]]).sum(dim=-1).view(-1) / sqrt(Q.size(-1))
            if self.num_pos_attention_heads > 1:
                attn_score = self.init_attn_lin(attn_score.view(agent_neighbour.size(1), self.num_pos_attention_heads))
            agent_neighbour_attention_value = gumbel_softmax(attn_score, agent_neighbour[0], num_nodes=num_agents_batch, hard=True, tau=(self.temp if self.training or not self.test_argmax else 1e-6))
            del Q, K, attn_score

            # Get updated agent positions
            indices = scatter_max(agent_neighbour_attention_value, agent_neighbour[0], dim=0, dim_size=num_agents_batch)[1]
            if indices.max() >= agent_neighbour_attention_value.size(0):
                # Try again as scatter_max sometimes randomly fails even though imputs are good???
                torch.cuda.synchronize(device=agent_neighbour_attention_value.device)
                indices = scatter_max(agent_neighbour_attention_value, agent_neighbour[0], dim=0, dim_size=num_agents_batch)[1]
                if indices.max() >= agent_neighbour_attention_value.size(0):
                    print(i, agent_neighbour_attention_value.max(), indices.max(), indices.argmax())
                    print(agent_neighbour_attention_value)
                    print(agent_neighbour)
                    print(agent_neighbour)
                    raise ValueError # Make sure agents are not placed 'out of bounds', this should not be possible here
            agent_pos = agent_neighbour[1][indices]
            agent_node = torch.stack([torch.arange(agent_pos.size(0), device=agent_pos.device), agent_pos], dim=0) # N_agents x N_nodes adjacency
            agent_node_attention_value = agent_neighbour_attention_value[indices] # NOTE multiply node emb with this to attach gradients when getting node agent is on
            del indices
        # Set agent positions randomly
        else:
            max_nodes_in_graph = scatter_add(torch.ones_like(batch, dtype=torch.float), batch).max().long().item()
            rand = torch.randint(0, int((max_nodes_in_graph - 1)*1e+6), [agent_emb.size(0)], device=edge_index.device) / int(max_nodes_in_graph*1e+6) # always < 1, node out of bounds (idx = num_nodes_batch) never picked with the floor
            node_counts = scatter_add(torch.ones_like(batch), batch, dim=0)
            step = node_counts.view(-1, 1).expand(-1, self.num_agents).reshape(-1)
            start = node_counts.cumsum(-1).view(-1, 1).expand(-1, self.num_agents).reshape(-1) - step
            agent_pos = torch.floor(rand * step + start).long()
            if agent_pos.max() >= num_nodes_batch:
                print(rand.max())
                raise ValueError # Make sure agents are not placed 'out of bounds'
            agent_node = torch.stack([torch.arange(agent_pos.size(0), device=agent_pos.device), agent_pos], dim=0) # N_agents x N_nodes adjacency
            agent_node_attention_value = torch.ones(agent_pos.size(0), dtype=torch.float, device=agent_pos.device)
            del max_nodes_in_graph, rand, node_counts, step, start
        node_agent = torch.stack([agent_node[1], agent_node[0]]) # Transpose with no coalesce
        node_agent_attn_value = agent_node_attention_value
        if edge_feat is not None:
            edge_taken_emb = torch.zeros(agent_pos.size(0), edge_emb.size(-1), device=edge_emb.device)

        out = torch.zeros(batch_size, self.num_out_classes, device=edge_index.device)

        adj = torch_sparse.tensor.SparseTensor(row=edge_index_sl[0], col=edge_index_sl[1], value=edge_emb_sl,
                                sparse_sizes=(num_nodes_batch, num_nodes_batch), is_sorted=False)
        del edge_index_sl
        if self.sparse_conv:
            adj_no_self_loop = torch_sparse.tensor.SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_emb,
                                sparse_sizes=(num_nodes_batch, num_nodes_batch), is_sorted=False)
            del edge_index

        # # Fastest option to get agent -> neighbour adjacency?
        agent_neighbour_coo = torch_sparse.tensor.__getitem__(adj, agent_node[1]).coo()
        agent_neighbour = torch.stack(agent_neighbour_coo[:2])
        agent_neighbour_edge_emb = agent_neighbour_coo[2]
        del agent_neighbour_coo
        # print(agent_neighbour[0].unique().size(0))
        if agent_neighbour[0].unique().size(0) != num_agents_batch: # Check if all agents have a neighbor
            raise ValueError

        # Track visited nodes
        if self.basic_global_agent or self.basic_agent or self.bias_attention:
            visited_nodes = torch.zeros(num_nodes_batch, self.num_agents, dtype=torch.float, device=agent_neighbour.device)
            visited_nodes.scatter_(0, agent_pos.view(batch_size, self.num_agents), torch.ones(batch_size, self.num_agents, device=agent_pos.device))
        
        
        for i in range(self.num_steps+1):
            # Get time for current step
            time_emb = self.time_emb(torch.tensor([i], device=node_emb.device, dtype=torch.long))

            # In the first iteration just update the starting node/agent embeddings
            if i > 0: 
                # Fastest option to get agent -> neighbour adjacency?
                agent_neighbour_coo = torch_sparse.tensor.__getitem__(adj, agent_node[1]).coo()
                agent_neighbour = torch.stack(agent_neighbour_coo[:2])
                agent_neighbour_edge_emb = agent_neighbour_coo[2]
                del agent_neighbour_coo
                if agent_neighbour[0].unique().size(0) != num_agents_batch: # Check if all agents have a neighbor
                    print(i, agent_neighbour[0].unique().size(0), agent_neighbour[0], agent_neighbour[1])
                    raise ValueError       
                
                # Fill in neighbor attention scores using the learned logits for [back, stay, explored, unexplored]
                attn_score = torch.zeros_like(agent_neighbour[0], dtype=torch.float)

                # Update tracked positions
                visited_nodes = visited_nodes * self.visited_decay
                visited_nodes.scatter_(0, agent_pos.view(batch_size, self.num_agents), torch.ones(batch_size, self.num_agents, device=agent_pos.device))

                # Get tracked values for new neighbors
                neighbors_visited = visited_nodes[agent_neighbour[1]].gather(1, (agent_neighbour[0] % self.num_agents).unsqueeze(1)).squeeze(1)

                mask_old = neighbors_visited < 1.0 # Disregard the current node
                
                if edge_feat is not None:
                    state = torch.cat([agent_emb[agent_neighbour[0]], node_emb[agent_neighbour[1]], agent_neighbour_edge_emb, node_emb[agent_node[1]][agent_neighbour[0]]], dim=-1)
                else:
                    state = torch.cat([agent_emb[agent_neighbour[0]], node_emb[agent_neighbour[1]], node_emb[agent_node[1]][agent_neighbour[0]]], dim=-1)
                # use a aggreator layer to process the state
                # state_aggr = nn.Sequential(nn.Linear(state.size(-1), self.dim*3), nn.ReLU())

                action_value = self.dqn.choose_action(state)
                # flatten the attention value
                attn_score = action_value.flatten()

                agent_neighbour_attention_value = gumbel_softmax(attn_score, agent_neighbour[0], num_nodes=num_agents_batch, hard=True, tau=(self.temp if self.training or not self.test_argmax else 1e-6), i=i)

                if (np.random.random() < self.dqn.epsilon):
                    _, neighbors_counts = agent_neighbour[0].unique(return_counts=True)
                    b = neighbors_counts.tolist()
                    segment_indices = [list(range(sum(b[:i]), sum(b[:i]) + count)) for i, count in enumerate(b)]
                    indices = torch.tensor([random.choice(segment) for segment in segment_indices], device=agent_neighbour[0].device,dtype = agent_neighbour[0].dtype)
                    q = attn_score[indices]
  
                else:     
                    # print('q',end = '')
                    q = scatter_max(attn_score, agent_neighbour[0], dim=0, dim_size=num_agents_batch)[0]
                
                    indices = scatter_max(agent_neighbour_attention_value, agent_neighbour[0], dim=0, dim_size=num_agents_batch)[1] # NOTE: could convert this to boolean tensor instead

                    if indices.max() >= agent_neighbour_attention_value.size(0):
                        # Try again as scatter_max sometimes randomly fails even though imputs are good???
                        torch.cuda.synchronize(device=agent_neighbour_attention_value.device)
                        indices = scatter_max(agent_neighbour_attention_value, agent_neighbour[0], dim=0, dim_size=num_agents_batch)[1]
                        if indices.max() >= agent_neighbour_attention_value.size(0):
                            print(i, agent_neighbour_attention_value.max(), indices.max(), indices.argmax())
                            print(agent_neighbour_attention_value)
                            print(agent_neighbour)
                            print(agent_neighbour)
                            raise ValueError # Make sure agents are not placed 'out of bounds', this should not be possible here
    
                agent_pos = agent_neighbour[1][indices]
                if agent_pos.max() >= num_nodes_batch:
                    print(i, agent_pos, agent_neighbour)
                    raise ValueError # Make sure agents are not placed 'out of bounds', this should not be possible here
                if indices.size(0) != num_agents_batch: # Check if all agents have a neighbor
                    print(i, agent_pos.unique().size(0), agent_pos)
                    raise ValueError
                agent_node = torch.stack([torch.arange(agent_pos.size(0), device=agent_pos.device), agent_pos], dim=0) # N_agents x N_nodes adjacency
                agent_node_attention_value = torch.ones_like(agent_pos, dtype=torch.float) if self.random_agent else agent_neighbour_attention_value[indices] # NOTE multiply node emb with this to attach gradients when getting node agent is on
                node_agent = torch.stack([agent_node[1], agent_node[0]]) # Transpose with no coalesce
                node_agent_attn_value = agent_node_attention_value
                if edge_feat is not None:
                    edge_taken_emb = agent_neighbour_edge_emb.clone()[indices]
                
                # create reward
                reward = torch.zeros_like(agent_pos, dtype=torch.float)
                # reward = torch.full_like(agent_pos, -1, dtype=torch.float)

                # if agent walks to a node that has been visited, set the reward to -1, otherwise 0
                mask_old_agent = mask_old[indices]
                reward[mask_old_agent] = -1.0

                if fgs is not None:
                    # if agent walks to a node of functional gruops, reward = 1.0.
                    reward[fgs[agent_pos] == 1] += 1.0
                    visited_nodes_update = visited_nodes.scatter_(0, agent_pos.view(batch_size, self.num_agents), torch.ones(batch_size, self.num_agents, device=agent_pos.device))
                    sliced_visited_nodes = torch.split(visited_nodes_update.T, node_num.tolist(),dim=1)
                    sliced_fgs = torch.split(fgs, node_num.tolist(),dim=0)
                    comparison = [torch.all(torch.abs(sliced_visited_node - sliced_fg) < 0.3, dim=1) for sliced_visited_node, sliced_fg in zip(sliced_visited_nodes, sliced_fgs)]
                    comparison_idx = torch.cat(comparison, dim=0)
                    if comparison_old is not None:
                        reward[comparison_idx ^ comparison_old] += 3.0
                        # only reward the node the first time it visited all functional groups
                    comparison_old = comparison_idx
                # print('reward_3', reward, reward.size())

                if cycles is not None:
                    reward[cycles[agent_pos] == 1] += 1.0
                if max_degree_node is not None:
                    reward[max_degree_node[agent_pos] == 1] += 1.0

                del indices

            # Update node embeddings
            active_nodes = torch.unique(agent_pos)
            if self.agent_node_edge_lin is not None:
                agent_ln = F.leaky_relu(self.agent_node_ln(agent_emb) + self.agent_node_edge_lin(edge_taken_emb * agent_node_attention_value.unsqueeze(-1)), negative_slope=self.edge_negative_slope)
            else:
                if edge_feat is not None:
                    agent_cat = torch.cat([agent_emb, edge_taken_emb * agent_node_attention_value.unsqueeze(-1)], dim=-1)
                else:
                    agent_cat = agent_emb
                agent_ln = self.agent_node_ln(agent_cat)
                del agent_cat
            if self.global_agent_pool:
                global_agent = agent_ln.view(batch_size, self.num_agents, -1).mean(dim=1, keepdim=False)
                node_update = torch.cat([node_emb[active_nodes], spmm(node_agent, node_agent_attn_value, num_nodes_batch, num_agents_batch, agent_ln, reduce=self.reduce)[active_nodes], global_agent[batch[active_nodes]]], dim=-1)
                node_update = node_update + self.node_mlp_time(time_emb)
            else:
                node_update = torch.cat([node_emb[active_nodes], spmm(node_agent, node_agent_attn_value, num_nodes_batch, num_agents_batch, agent_ln, reduce=self.reduce)[active_nodes]], dim=-1) + self.node_mlp_time(time_emb)
            del agent_ln
            node_emb[active_nodes] = self.node_ln(node_emb[active_nodes] + self.node_mlp(node_update))
            del node_update

            # Do a convolution to get neighborhood info:
            if self.sparse_conv:
                active_edge_index_coo = torch_sparse.tensor.__getitem__(adj_no_self_loop, active_nodes).coo()
                active_edge_index = torch.stack(active_edge_index_coo[:2])
                active_edge_emb = active_edge_index_coo[2]
                del active_edge_index_coo
            else: # For small graphs just doing a conv on everything, but updating only active nodes is faster
                active_edge_index = edge_index
                active_edge_emb = edge_emb
            if self.qm9:
                # Do the NNConv - convolution (https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/nn_conv.html#NNConv)
                weight = self.edge_nn(active_edge_emb)
                weight = weight.view(-1, self.dim, self.dim)
                message_val = torch.matmul(node_emb[active_edge_index[1]].unsqueeze(1), weight).squeeze(1)
                node_update = torch.cat([node_emb[active_nodes], scatter(message_val, active_edge_index[0], dim=0, dim_size=num_nodes_batch, reduce=self.reduce)[active_nodes]], dim=-1)
                del weight, message_val
            elif edge_feat is not None:
                # Pre-process edge embeddings
                node_cat = torch.cat([node_emb[active_edge_index[1]], active_edge_emb], dim=-1)
                # Need to materialize the messages if we have edge features
                message_val = self.message_val(node_cat)
                node_update = torch.cat([node_emb[active_nodes], scatter(message_val, active_edge_index[0], dim=0, dim_size=num_nodes_batch, reduce=self.reduce)[active_nodes]], dim=-1)
                del message_val
            else:
                node_update = torch.cat([node_emb[active_nodes], spmm(active_edge_index, torch.ones_like(active_edge_index[0], dtype=torch.float), num_nodes_batch, num_nodes_batch, node_emb, reduce=self.reduce)[active_nodes]], dim=-1)
            node_emb[active_nodes] = self.conv_ln(node_emb[active_nodes] + self.conv_mlp(node_update))
            del node_update

            # Update agent embeddings
            if edge_feat is not None:
                agent_cat = torch.cat([agent_emb, node_emb[agent_pos] * agent_node_attention_value.unsqueeze(-1), edge_taken_emb * agent_node_attention_value.unsqueeze(-1)], dim=-1)
            else:
                agent_cat = torch.cat([agent_emb, node_emb[agent_pos] * agent_node_attention_value.unsqueeze(-1)], dim=-1)
            if self.global_agent_pool and self.agent_global_extra:
                agent_cat = torch.cat([agent_cat, global_agent.unsqueeze(1).expand(-1, self.num_agents, -1).reshape(num_agents_batch, -1)], dim=-1)
            agent_emb = self.agent_ln(agent_emb + self.agent_mlp(agent_cat + self.agent_mlp_time(time_emb)))
            del agent_cat

            # Readout
            if not self.final_readout_only:
                layer_out = self.step_readout_mlp(agent_emb.view(batch_size, self.num_agents, -1) + self.step_readout_mlp_time(time_emb))

                if self.mean_pool_only:
                    layer_out = torch.mean(layer_out, dim=1)
                else:
                    layer_out = torch.cat([torch.max(layer_out, dim=1)[0], torch.mean(layer_out, dim=1)], dim=-1)
                
                if self.node_readout:
                    layer_out = torch.cat([layer_out, global_mean_pool(node_emb, batch)], dim=-1)

                layer_out = F.dropout(self.fc(layer_out), p=self.dropout, training=self.training)
                
                out = out + layer_out
                del layer_out
            
            if i > 0:
                if edge_feat is not None:
                    next_state = torch.cat([agent_emb[agent_neighbour[0]], node_emb[agent_neighbour[1]], agent_neighbour_edge_emb, node_emb[agent_node[1]][agent_neighbour[0]]], dim=-1)
                else:
                    next_state = torch.cat([agent_emb[agent_neighbour[0]], node_emb[agent_neighbour[1]], node_emb[agent_node[1]][agent_neighbour[0]]], dim=-1)
                with torch.no_grad():
                    q_ = self.dqn.q_target.forward(next_state)
                    # agent_neighbour_attention_value = gumbel_softmax(action_value, agent_neighbour[0], num_nodes=num_agents_batch, hard=True, tau=(self.temp if self.training or not self.test_argmax else 1e-6), i=i)
                    q_2 = scatter_max(q_.flatten(), agent_neighbour[0], dim=0, dim_size=num_agents_batch)[0]
                    target = reward + self.dqn.gamma * q_2

                self.dqn.memory.append([q, target])
                # Calculate the mean of accumulated losses
                self.dqn.total_reward.append(reward.sum())
                            

        if self.final_readout_only:
            out = self.step_readout_mlp(agent_emb.view(batch_size, self.num_agents, -1) + self.step_readout_mlp_time(time_emb))
            if self.mean_pool_only:
                out = torch.mean(out, dim=1)
            else:
                out = torch.cat([torch.max(out, dim=1)[0], torch.mean(out, dim=1)], dim=-1)
            
            if self.node_readout:
                out = torch.cat([out, global_mean_pool(node_emb, batch)], dim=-1)

            out = self.fc(out)
        else:
            out = out / (self.num_steps+1)

        if not self.ogb_mol and not self.regression:
            out = F.log_softmax(out, dim=-1)
        
        # self.dqn.mean_loss = torch.mean(torch.stack(self.dqn.loss_buffer), dim=0)
        self.dqn.total_reward = torch.mean(torch.stack(self.dqn.total_reward), dim=0)
        # print('-------- self.dqn mean loss: ', self.dqn.mean_loss.item(), ', total reward: ', self.dqn.total_reward.item(), '--------')

        return out, 0