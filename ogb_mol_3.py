# Based on https://github.com/snap-stanford/ogb/tree/master/examples/graphproppred/mol
import torch
from torch_geometric.loader import DataLoader
from test_tube.hpc import SlurmCluster
from torch_geometric.data import Dataset
import random
import os
import ast
import socket
import time
import pandas as pd
import torch_geometric.transforms as T
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

from util import cos_anneal, get_cosine_schedule_with_warmup, plot_learning_curve
from model_3 import AgentNet, add_model_args

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()

def add_attributes(dataset,fgs):
    data_list = []
    for i, data in enumerate(dataset):
        node_num = data.x.size(0)
        fgs_onehot = torch.zeros(node_num)
        data.node_num = data.x.size(0)
        fgs_onehot[fgs[i]] = 1
        data.fgs = fgs_onehot
        data_list.append(data)
    
    new_dataset = dataset.__class__(root=dataset.root, name=dataset.name, transform=dataset.transform,
                                pre_transform=dataset.pre_transform)
    new_dataset.data, new_dataset.slices = dataset.collate(data_list)
    return new_dataset


def train(model, device, loader, optimizer, task_type, use_aux_loss=False):
    model.train()

    for batch in tqdm(loader, desc="Iteration", disable=not args.bar):
        batch = batch.to(device)
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            if use_aux_loss:
                pred, aux_pred = model(x = batch.x, edge_index = batch.edge_index, batch = batch.batch, edge_feat = batch.edge_attr, fgs = batch.fgs, node_num = batch.node_num)
            else:
                pred, _, = model(x = batch.x, edge_index = batch.edge_index, batch = batch.batch, edge_feat = batch.edge_attr, fgs = batch.fgs, node_num = batch.node_num) 
            
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            if "classification" in task_type: 
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            if use_aux_loss:
                if "classification" in task_type: 
                    aux_loss = cls_criterion(aux_pred[is_labeled.unsqueeze(1).expand(-1,aux_pred.size(1),-1)].to(torch.float32).view(-1), batch.y.to(torch.float32)[is_labeled].unsqueeze(1).expand(-1,aux_pred.size(1)).clone().view(-1))
                else:
                    aux_loss = reg_criterion(aux_pred[is_labeled.unsqueeze(1).expand(-1,aux_pred.size(1),-1)].to(torch.float32).view(-1), batch.y.to(torch.float32)[is_labeled].unsqueeze(1).expand(-1,aux_pred.size(1)).clone().view(-1))
                loss = 0.75*loss + 0.25*aux_loss
                
            optimizer.zero_grad()
            # model.dqn.q_eval.optimizer.zero_grad()
         
            q,target = model.dqn.sample_memory()
            dqn_loss = F.mse_loss(q, target)
            tol_loss = loss + dqn_loss * args.discount

            tol_loss.backward()

            optimizer.step()
            # dqn.q_eval.optimizer.step()
            model.dqn.epsilon = model.dqn.epsilon - model.dqn.eps_dec if model.dqn.epsilon > model.dqn.eps_min else model.dqn.eps_min
            model.dqn.update_network_parameters()


def eval(model, device, loader, evaluator, use_aux_loss=False):
    model.eval()
    y_true = []
    y_pred = []
    rewards = []

    for batch in tqdm(loader, desc="Iteration",disable=not args.bar):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred, _ = model(x = batch.x, edge_index = batch.edge_index, batch = batch.batch, edge_feat = batch.edge_attr, fgs = batch.fgs, node_num = batch.node_num)
            reward = model.dqn.total_reward/len(batch.y)
            rewards.append(reward.detach().cpu())
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict), sum(rewards)/len(rewards)


def main(args, cluster=None):

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    print('HostName: {}, GPU ID: {}/{}, StartTime: {}, LoginName: {}, CodeFile: {}'.format(
        socket.gethostname(), os.environ['CUDA_VISIBLE_DEVICES'], torch.cuda.device_count(), time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), os.getlogin(),__file__), flush=True)
    
    print(args, flush=True)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    # Seed things
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load fgs data
    if args.dataset == 'ogbg-molhiv':
        fgs_filename = os.path.join('dataset/ogbg_molhiv/fgs.csv') 
    elif args.dataset == 'ogbg-molpcba':
        fgs_filename = os.path.join('dataset/ogbg_molpcba/fgs.csv')  
    df = pd.read_csv(fgs_filename,usecols=['fgs'])
    df['fgs'] = df['fgs'].astype(str)
    df['fgs'] = df['fgs'].apply(lambda x: ast.literal_eval(x))
    fgs = df['fgs'].tolist()
    print('------Functional Groups Data Loaded------', flush=True)

    # add fgs to data as a node attribute
    pyg_dataset = PygGraphPropPredDataset(name = args.dataset)
    dataset = add_attributes(pyg_dataset,fgs)

    if args.feature == 'full':
        pass
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:,:2]
        dataset.data.edge_attr = dataset.data.edge_attr[:,:2]
    
    ### automatic dataloading and splitting
    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    model = AgentNet(num_features=dataset.num_features, hidden_units=args.hidden_units, num_out_classes=dataset.num_tasks, dropout=args.dropout, num_steps=args.num_steps,
                    num_agents=args.num_agents, reduce=args.reduce, node_readout=args.node_readout, use_step_readout_lin=args.use_step_readout_lin,
                    num_pos_attention_heads=args.num_pos_attention_heads, readout_mlp=args.readout_mlp, self_loops=args.self_loops, post_ln=args.post_ln,
                    attn_dropout=args.attn_dropout, no_time_cond=args.no_time_cond, mlp_width_mult=args.mlp_width_mult, activation_function=args.activation_function,
                    negative_slope=args.negative_slope, input_mlp=args.input_mlp, attn_width_mult=args.attn_width_mult, importance_init=args.importance_init,
                    random_agent=args.random_agent, test_argmax=args.test_argmax, global_agent_pool=args.global_agent_pool, agent_global_extra=args.agent_global_extra,
                    basic_global_agent=args.basic_global_agent, basic_agent=args.basic_agent, bias_attention=args.bias_attention, visited_decay=args.visited_decay,
                    sparse_conv=args.sparse_conv, mean_pool_only=args.mean_pool_only, edge_negative_slope=args.edge_negative_slope,
                    final_readout_only=args.final_readout_only, ogb_mol=True).to(device)


    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': args.lr},
        {'params': model.dqn.q_eval.parameters(), 'lr': args.lr_dqn}
    ], weight_decay=args.weight_decay)

    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup, args.epochs, min_lr_mult=args.min_lr_mult)

    valid_curve = []
    test_curve = []
    train_curve = []
    reward_curve = []
    save_every = 50
    save_path = os.path.join('results', args.dataset , args.save_path)
    checkpoint_path = os.path.join(save_path, 'checkpoints/')
    curve_path = os.path.join(save_path, 'curves/')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(checkpoint_path)
        os.makedirs(curve_path)

    for epoch in range(1, args.epochs + 1):
        torch.cuda.reset_peak_memory_stats(0)
        print("=====Epoch {}=====".format(epoch))
        # print('Training...')
        lr, lr_dqn = [param_group['lr'] for param_group in scheduler.optimizer.param_groups]

        if args.gumbel_warmup < 0:
            gumbel_warmup = args.warmup
        else:
            gumbel_warmup = args.gumbel_warmup
        model.temp = cos_anneal(gumbel_warmup, gumbel_warmup + args.gumbel_decay_epochs, args.gumbel_temp, args.gumbel_min_temp, epoch)
        train(model, device, train_loader, optimizer, dataset.task_type, use_aux_loss=args.use_aux_loss)
        scheduler.step()
        # scheduler_dqn.step()

        # print('Evaluating...')
        train_perf, train_reward= eval(model, device, train_loader, evaluator, use_aux_loss=args.use_aux_loss)
        train_curve.append(train_perf[dataset.eval_metric])
        reward_curve.append(int(train_reward))
        print({'train_loss': train_perf, 'train_reward' : train_reward.item(), 'lr': lr, 'lr_dqn': lr_dqn}, flush=True)

        if args.fast is None or (args.fast is not None and epoch % args.fast == 0):
            valid_perf, valid_reward = eval(model, device, valid_loader, evaluator, use_aux_loss=args.use_aux_loss)
            test_perf, test_reward = eval(model, device, test_loader, evaluator, use_aux_loss=args.use_aux_loss)
            valid_curve.append(valid_perf[dataset.eval_metric])
            test_curve.append(test_perf[dataset.eval_metric])
            print({'Train': [train_perf, train_reward.item()] , 'Validation': [valid_perf,valid_reward.item()], 'Test': [test_perf,test_reward.item()], 'LR': lr, 'Mem': round(torch.cuda.max_memory_allocated()/1024.0**3, 3), 'Cached': round(torch.cuda.max_memory_reserved()/1024.0**3, 3)})
        

        if epoch % save_every == 0:
            model.save_checkpoint(epoch,optimizer,scheduler,checkpoint_path + 'DAgent_' + args.dataset + '_{}.pt'.format(epoch))
            print('{} epoch Checkpoint saved to {}'.format(epoch, checkpoint_path))

    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)

    print('Finished training!')
    print('Best tain score: {}'.format(best_train))
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))

    plot_learning_curve(np.arange(1, args.epochs+1), train_curve, 'Learning Curve', 'train', curve_path + 'train_curve.png')
    plot_learning_curve(np.arange(1, len(valid_curve)+1), valid_curve, 'Learning Curve', 'valid', curve_path + 'valid_curve.png')
    plot_learning_curve(np.arange(1, len(test_curve)+1), test_curve, 'Learning Curve', 'test', curve_path + 'test_curve.png')
    plot_learning_curve(np.arange(1, args.epochs+1), reward_curve , 'Learning Curve', 'rewards', curve_path + 'reward_curve.png')

if __name__ == "__main__":
    # AgentNet params
    parser = add_model_args(None, hyper=True)

    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                        help='dataset name (default: ogbg-molhiv)')

    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')

    parser.add_argument('--slurm', action='store_true', default=False)
    parser.add_argument('--grid_search', action='store_true', default=False)
    parser.add_argument('--gpu_jobs', action='store_true', default=False)
    # Run all seeds
    parser.opt_list('--seed', type=int, default=0, tunable=True, options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    parser.add_argument('--gpu_id', type=str, default='0', help = 'specify which gpu on server to use')
    parser.add_argument('--lr_dqn', type=float, default=0.0001)
    parser.add_argument('--discount', type=float, default=0.2, help = 'discount factor for DQN loss')
    parser.add_argument('--save_path', type=str, default='0915', help = 'specify which gpu on server to use')
    parser.add_argument('--fast', type = int, default = None, help = 'for fast training, only evaluate validation every n epochs')
    parser.add_argument('--bar', action = 'store_true', default = False, help = 'display a progress bar for training and evaluation')
    args = parser.parse_args()

    if args.slurm:
        print('Launching SLURM jobs')
        log_path = 'slurm_log/'
        cluster = SlurmCluster(
            hyperparam_optimizer=args,
            log_path=log_path,
            python_cmd='python'
        )
        cluster.job_time = '48:00:00'
        
        cluster.memory_mb_per_node = '24G'

        job_name = f'{args.dataset}_{args.num_agents}_{args.num_steps}_{args.hidden_units}_{args.lr}_{args.edge_negative_slope}_{args.batch_size}_all_seeds'
        if args.gpu_jobs:
            cluster.per_experiment_nb_cpus = 2
            cluster.per_experiment_nb_gpus = 1
            cluster.optimize_parallel_cluster_gpu(main, nb_trials=None, job_name=job_name, job_display_name=args.dataset)
        else:
            cluster.per_experiment_nb_cpus = 8
            cluster.optimize_parallel_cluster_cpu(main, nb_trials=None, job_name=job_name, job_display_name=args.dataset)
    elif args.grid_search:
        for hparam_trial in args.trials(None):
            main(hparam_trial)
    else:
        main(args)