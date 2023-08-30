# This implementation is based on https://github.com/KarolisMart/DropGNN/blob/main/mpnn-qm9.py
import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.datasets import QM9
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.utils import remove_self_loops
import time
import pandas as pd
import os
from tqdm import tqdm
import ast
import numpy as np
import os
import socket

from util import cos_anneal, get_cosine_schedule_with_warmup, plot_learning_curve
from model_2 import AgentNet, add_model_args


parser = add_model_args()
parser.add_argument('--target', default=0)
parser.add_argument('--aux_loss', action='store_true', default=False)
parser.add_argument('--complete_graph', action='store_true', default=False)
parser.add_argument('--num_worders',type = int, default=10)
parser.add_argument('--gpu_id',type=str, default="0")
args = parser.parse_args()
print(args)
target = int(args.target)
print('---- Target: {} ----'.format(target))

dataset_path = 'data/MPNN-QM9/'
fgs_filename = os.path.join(dataset_path, 'fgs.csv')   
df = pd.read_csv(fgs_filename,usecols=['ID','fgs'])
df['fgs'] = df['fgs'].astype(str)
df['fgs'] = df['fgs'].apply(lambda x: ast.literal_eval(x))
dict = df.set_index('ID')['fgs'].to_dict()
print('fgs data loaded')


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
set_num_workers = args.num_worders
print('HostName: {}, LoginName: {}, Current GPU ID: {}/{}, Num of workers: {}/{}, CodeFile: {}'.format(
        socket.gethostname(), os.getlogin(), 
        os.environ['CUDA_VISIBLE_DEVICES'], torch.cuda.device_count(), 
        set_num_workers, torch.get_num_threads(), __file__), flush=True)

def add_attributes(dataset,dict):
    data_list = []
    for i, data in enumerate(dataset):
        node_num = data.x.size(0)
        fgs_onehot = torch.zeros(node_num)
        data.node_num = node_num
        if dict.get(data.name) != None:
            fgs = dict[data.name]
            fgs_onehot[fgs] = 1
        data.fgs = fgs_onehot
        data_list.append(data)
    
    new_dataset = dataset.__class__(root=dataset.root, transform=dataset.transform)
    new_dataset.data, new_dataset.slices = dataset.collate(data_list)
    return new_dataset


class MyTransform(object):
    def __call__(self, data):
        data.y = data.y[:, int(args.target)]  # Specify target: 0 = mu

        # node_num = data.x.size(0)
        # # make a one hot fgs vector
        # node_fgs = torch.zeros(node_num)
        # data.node_num = node_num
        
        # if dict.get(data.name) != None:
        #     fgs = dict[data.name]
        #     node_fgs[fgs] = 1
        #     data.fgs = node_fgs
        # else:
        #     data.fgs = node_fgs

        return data

if args.complete_graph:
    class Complete(object):
        def __call__(self, data):
            device = data.edge_index.device

            row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
            col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

            row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
            col = col.repeat(data.num_nodes)
            edge_index = torch.stack([row, col], dim=0)

            edge_attr = None
            if data.edge_attr is not None:
                idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
                size = list(data.edge_attr.size())
                size[0] = data.num_nodes * data.num_nodes
                edge_attr = data.edge_attr.new_zeros(size)
                edge_attr[idx] = data.edge_attr

            edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
            data.edge_attr = edge_attr
            data.edge_index = edge_index

            return data
    

else:
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', '1-QM9')
    dataset = QM9(path, transform=T.Compose([MyTransform(), T.Distance()]))


path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'MPNN-QM9')
transform = T.Compose([MyTransform(), Complete(), T.Distance(norm=False)])
dataset = QM9(path)

dataset = add_attributes(dataset,dict)
dataset.transform = transform
print('example data:',dataset[1],dataset[1].fgs,dataset[1].node_num)

dataset = dataset.shuffle()

# Normalize targets to mean = 0 and std = 1.
tenpercent = int(len(dataset) * 0.1)
mean = dataset.data.y[tenpercent:].mean(dim=0)
std = dataset.data.y[tenpercent:].std(dim=0)
dataset.data.y = (dataset.data.y - mean) / std

test_dataset = dataset[:tenpercent]
val_dataset = dataset[tenpercent:2 * tenpercent]
train_dataset = dataset[2 * tenpercent:]
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=set_num_workers,pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=set_num_workers,pin_memory=True)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=set_num_workers,pin_memory=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

use_aux_loss = args.aux_loss

model = AgentNet(num_features=dataset.num_features, hidden_units=args.hidden_units, num_out_classes=1, dropout=args.dropout, num_steps=args.num_steps,
                num_agents=args.num_agents, reduce=args.reduce, node_readout=args.node_readout, use_step_readout_lin=args.use_step_readout_lin,
                num_pos_attention_heads=args.num_pos_attention_heads, readout_mlp=args.readout_mlp, self_loops=args.self_loops, post_ln=args.post_ln,
                attn_dropout=args.attn_dropout, no_time_cond=args.no_time_cond, mlp_width_mult=args.mlp_width_mult, activation_function=args.activation_function,
                negative_slope=args.negative_slope, input_mlp=args.input_mlp, attn_width_mult=args.attn_width_mult, importance_init=args.importance_init,
                random_agent=args.random_agent, test_argmax=args.test_argmax, global_agent_pool=args.global_agent_pool, agent_global_extra=args.agent_global_extra,
                basic_global_agent=args.basic_global_agent, basic_agent=args.basic_agent, bias_attention=args.bias_attention, visited_decay=args.visited_decay,
                sparse_conv=args.sparse_conv, mean_pool_only=args.mean_pool_only, edge_negative_slope=args.edge_negative_slope,
                final_readout_only=args.final_readout_only, num_edge_features=5, regression=True, qm9=True).to(device)


optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup, args.epochs, min_lr_mult=args.min_lr_mult)

mean, std = mean[target].to(device), std[target].to(device)

def train(epoch):
    model.train()
    loss_all = 0
    rewards = []
    verbose = True

    # for data in train_loader:
    for step, data in enumerate(tqdm(train_loader, desc="Iteration", disable=not verbose)):
        # print('step:',step,'of',len(train_loader))
        data = data.to(device)
        optimizer.zero_grad()
        pred, aux_pred, dqn = model(data.x, data.edge_index, data.batch, data.edge_attr, data.fgs, data.node_num)

        dqn.q_eval.optimizer.zero_grad()

        loss = F.mse_loss(pred.view(-1), data.y)
        if use_aux_loss:
            aux_loss = F.mse_loss(aux_pred.view(-1), data.y.unsqueeze(1).expand(-1,aux_pred.size(1)).clone().view(-1))
            loss = 0.75*loss + 0.25*aux_loss
        loss_all += loss.item() * data.num_graphs

        tol_loss = dqn.mean_loss + loss
            
        tol_loss.backward()

        optimizer.step()
        
        dqn.q_eval.optimizer.step()
        dqn.epsilon = dqn.epsilon - dqn.eps_dec if dqn.epsilon > dqn.eps_min else dqn.eps_min
        dqn.update_network_parameters()

        rewards.append(dqn.total_reward)

    return loss_all / len(train_loader.dataset), sum(rewards)/len(rewards), dqn

def test(loader):
    model.eval()
    error = 0

    for data in loader:
        data = data.to(device)
        pred, _, dqn = model(data.x, data.edge_index, data.batch, data.edge_attr, data.fgs, data.node_num)
        error += ((pred.view(-1) * std) -
                  (data.y * std)).abs().sum().item() # MAE
    return error / len(loader.dataset)

# del dict
print(model.__class__.__name__)
best_val_error = None
losses = []
val_errors = []
rewards = []

save_every = 50
save_path = 'qm9/' + 'qm9_2' +'/'+args.target+'/' 
# + time.strftime("%m%d-%H%M") + '/'
checkpoint_path = save_path + 'checkpoint/'
curve_path = save_path + 'figs/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
    os.makedirs(checkpoint_path)
    os.makedirs(curve_path)

for epoch in range(1, args.epochs+1):
    torch.cuda.reset_peak_memory_stats(0)
    start = time.time()
    lr = scheduler.optimizer.param_groups[0]['lr']

    print('Epoch: {:03d}/{:03d}, LR: {:7f}, StartTime: {}'.format(epoch, args.epochs, lr, time.asctime()), flush=True)

    if args.gumbel_warmup < 0:
        gumbel_warmup = args.warmup
    else:
        gumbel_warmup = args.gumbel_warmup
    model.temp = cos_anneal(gumbel_warmup, gumbel_warmup + args.gumbel_decay_epochs, args.gumbel_temp, args.gumbel_min_temp, epoch)
    loss, reward, dqn = train(epoch)
    val_error = test(val_loader)
    scheduler.step()#val_error

    if best_val_error is None:
        best_val_error = val_error
    if val_error <= best_val_error:
        test_error = test(test_loader)
        best_val_error = val_error
    
    print('Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Validation MAE: {:.7f}, Test MAE: {:.7f}, Reward: {:.3f}, Time: {:.4f}, Mem: {:.3f}, Cached: {:.3f}'.format(
        epoch, lr, loss, val_error, test_error, reward, time.time() - start, torch.cuda.max_memory_allocated()/1024.0**3, torch.cuda.max_memory_reserved()/1024.0**3), flush=True)
    losses.append(loss)
    val_errors.append(val_error)
    rewards.append(reward)

    if epoch % save_every == 0:
        checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}
        torch.save(checkpoint, checkpoint_path + 'AgentNet_{}.pt'.format(epoch))
        dqn.save_models(epoch, checkpoint_path)
        print('{} epoch Checkpoint saved to {}'.format(epoch, checkpoint_path))

print('---------------- Final Result ----------------')
print('Best Validation MAE: {:.7f}'.format(best_val_error))
print('Test MAE: {:.7f}'.format(test_error))
print('----------------------------------------------')


plot_learning_curve(np.arange(1, args.epochs+1), losses, 'Learning Curve', 'losses', curve_path +'losses.png')
plot_learning_curve(np.arange(1, args.epochs+1), val_errors, 'Learning Curve', 'val_errors', curve_path +'val_errors.png')
plot_learning_curve(np.arange(1, args.epochs+1), rewards, 'Learning Curve', 'rewards', curve_path+'rewards.png')