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

from util import cos_anneal, get_cosine_schedule_with_warmup
from model_3 import AgentNet, add_model_args

import matplotlib.pyplot as plt
 
 
def plot_learning_curve(episodes, records, title, ylabel, figure_file):
    plt.figure()
    plt.plot(episodes, records, linestyle='-', color='r')
    plt.title(title)
    plt.xlabel('episode')
    plt.ylabel(ylabel)
 
    plt.show()
    plt.savefig(figure_file)
    plt.close()




import os
os.environ['CUDA_VISIBLE_DEVICES'] = "5"
print("torch.cuda.device_count() {}".format(torch.cuda.device_count()))

parser = add_model_args()
parser.add_argument('--target', default=0)
parser.add_argument('--aux_loss', action='store_true', default=False)
parser.add_argument('--complete_graph', action='store_true', default=False)
args = parser.parse_args()
print(args)
target = int(args.target)
print('---- Target: {} ----'.format(target))

dataset_path = 'data/MPNN-QM9/'
fgs_filename = os.path.join(dataset_path, 'fgs.csv')   
df = pd.read_csv(fgs_filename,usecols=['ID','fgs'])
df['fgs'] = df['fgs'].astype(str)
df['fgs'] = df['fgs'].apply(lambda x: ast.literal_eval(x))
print('FGS csv Loaded')
dict = df.set_index('ID')['fgs'].to_dict()


class MyTransform(object):
    def __init__(self, dict):
        self.dict = dict
    def __call__(self, data):
        data.y = data.y[:, int(args.target)]  # Specify target: 0 = mu

        node_num = data.x.size(0)
        # make a one hot fgs vector
        node_fgs = torch.zeros(node_num)
        
        if dict.get(data.name) != None:
            fgs = dict[data.name]
            node_fgs[fgs] = 1
            data.fgs = node_fgs
        else:
            data.fgs = node_fgs
        # fgs = str(fgs)
        # print(fgs,type(fgs))
        # print(fgs)
        # print(fgs.iloc[0])
        # fgs = ast.literal_eval(fgs)
        # print(data.fgs)
        

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
    
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'MPNN-QM9')
    transform = T.Compose([MyTransform(df), Complete(), T.Distance(norm=False)])
    dataset = QM9(path, transform=transform)
else:
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', '1-QM9')
    dataset = QM9(path, transform=T.Compose([MyTransform(), T.Distance()]))


# print(len(dataset))
# for data in dataset[130800:]:
#     print('fgs', data)
#     print('fgs', data.fgs)
#     print('fgs', data.x)

# import sys
# sys.exit()

# # read dataframe
# dataset_path = 'data/MPNN-QM9/'
# fgs_filename = os.path.join(dataset_path, 'fgs.csv')   
# df = pd.read_csv(fgs_filename,usecols=['ID','fgs'])
# print('FGS csv Loaded')

# print(df['fgs'][10],type(df['fgs'][10]))

# # add fgs to dataset
# print('Adding fgs to dataset...')

# for data in tqdm(dataset[:15]):
#     fgs = df[df['ID'] == data.name]['fgs']
#     fgs = ast.literal_eval(fgs.iloc[0])
#     print(fgs,type(fgs))
#     data.fgs = torch.IntTensor(fgs)
#     print(data.fgs)


# # see revised dataset inside
# for data in dataset[:10]:
#     print(data.fgs)

dataset = dataset.shuffle()

print(torch.get_num_threads(),type(torch.get_num_threads()))
set_num_workers =  torch.get_num_threads()
print('set_num_workers:', set_num_workers)

# Normalize targets to mean = 0 and std = 1.
tenpercent = int(len(dataset) * 0.1)
mean = dataset.data.y[tenpercent:].mean(dim=0)
std = dataset.data.y[tenpercent:].std(dim=0)
dataset.data.y = (dataset.data.y - mean) / std

test_dataset = dataset[:tenpercent]
val_dataset = dataset[tenpercent:2 * tenpercent]
train_dataset = dataset[2 * tenpercent:]
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=set_num_workers)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=set_num_workers)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=set_num_workers)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# # 获取当前GPU名字
# gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
# # 获取当前GPU总显存
# props = torch.cuda.get_device_properties(device)
# total_memory = props.total_memory / 1e9

# gpu_info = "当前GPU 型号是：{}，可用总显存为：{} GB".format(gpu_name, total_memory)
# print(gpu_info, gpu_name)

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

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        pred, aux_pred,dqn = model(data.x, data.edge_index, data.batch, data.edge_attr, data.fgs,data.name)

        dqn.q_eval.optimizer.zero_grad()

        loss = F.mse_loss(pred.view(-1), data.y)
        if use_aux_loss:
            aux_loss = F.mse_loss(aux_pred.view(-1), data.y.unsqueeze(1).expand(-1,aux_pred.size(1)).clone().view(-1))
            loss = 0.75*loss + 0.25*aux_loss
        loss.backward(retain_graph=True)
        loss_all += loss.item() * data.num_graphs

        loss_dqn = dqn.mean_loss + loss
            
        loss_dqn.backward(retain_graph=True)

        optimizer.step()
        
        dqn.q_eval.optimizer.step()
        dqn.update_network_parameters()

        reward = dqn.total_reward

    return loss_all / len(train_loader.dataset), reward

def test(loader):
    model.eval()
    error = 0

    for data in loader:
        data = data.to(device)
        pred, _, _ = model(data.x, data.edge_index, data.batch, data.edge_attr, data.fgs)
        error += ((pred.view(-1) * std) -
                  (data.y * std)).abs().sum().item() # MAE
    return error / len(loader.dataset)

# del dict
print(model.__class__.__name__)
best_val_error = None
losses = []
val_errors = []
rewards = []

for epoch in range(1, args.epochs+1):
    torch.cuda.reset_peak_memory_stats(0)
    start = time.time()
    lr = scheduler.optimizer.param_groups[0]['lr']
    if args.gumbel_warmup < 0:
        gumbel_warmup = args.warmup
    else:
        gumbel_warmup = args.gumbel_warmup
    model.temp = cos_anneal(gumbel_warmup, gumbel_warmup + args.gumbel_decay_epochs, args.gumbel_temp, args.gumbel_min_temp, epoch)
    loss, reward = train(epoch)
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

print('---------------- Final Result ----------------')
print('Best Validation MAE: {:.7f}'.format(best_val_error))
print('Test MAE: {:.7f}'.format(test_error))
print('----------------------------------------------')


save_path = 'log/qm9/results/'
plot_learning_curve(np.arange(1, args.epochs+1), losses, 'Learning Curve', 'losses', save_path+args.dataset+'_losses.png')
plot_learning_curve(np.arange(1, args.epochs+1), val_errors, 'Learning Curve', 'val_errors', save_path+args.dataset+'_val_errors.png')
plot_learning_curve(np.arange(1, args.epochs+1), rewards, 'Learning Curve', 'rewards', save_path+args.dataset+'_rewards.png')