# DQN Enhanced Agent-based GNNs for Target-Oriented Graph Classification
This project reformalised the agent walking strategy of AgentNet(https://arxiv.org/abs/2206.11010) as a reinforcement learning problem and implemented a Deep Q-network to find the optimal agent transition. By combining the reinforcement learning with graph neural networks, we achieve better performance on graph classification tasks with sepcific target-oriented agents.


## Getting Started

### Prerequisites
* python
* pytorch
* networkx
* matplotlib
* pyg
* numpy

### Installing
* Clone this repo
* Install dependencies: `conda env create -f environment.yml`

## Runinng The Code
The code is organized into 3 files:
* `graph_classification.py` - runs the real-world graph classification experiments
* `ogb_mol.py` - runs the OGB dataset experiments
* `qm9.py` - runs the QM9 dataset experiments

The number of agents and steps can be adjusted by changing the `--num_agents` and `--num_steps` flags. The number of agents should be equal to the mean number of nodes of the datasets. The following table displays the statistics of each dataset used in the experiments.


| Dataset      | \# Graphs | Mean \# Nodes | Max \# Nodes | Min \# Nodes | Mean Deg. | Max Deg. |
|--------------|-----------|---------------|--------------|--------------|-----------|----------|
| MUTAG        | 188       | 17.9          | 28           | 10           | 2.2       | 4        |
| PTC          | 344       | 25.6          | 109          | 2            | 2.0       | 4        |
| PROTEINS     | 1113      | 39.1          | 620          | 4            | 3.73      | 25       |
| IMDB-B       | 1000      | 19.8          | 136          | 12           | 9.8       | 135      |
| IMDB-M       | 1500      | 13.0          | 89           | 7            | 10.1      | 88       |
| DD           | 1178      | 284.3         | 5748         | 30           | 5.0       | 19       |
| OGB-MolHIV   | 41127     | 25.5          | 222          | 2            | 2.2       | 10       |
| OGB-MolPCBA  | 437929    | 26.0          | 332          | 1            | 2.2       | 5        |
| QM9          | 130831    | 18.0          | 29           | 3            | 2.0       | 5        |



### TU dataset 
The following command will run training on the TU datasets. The results report the accuracy score.

`python graph_classification_3.py --dataset 'MUTAG' --verbose --dropout 0.0 --num_agents 18 --num_steps 16 --batch_size 32 --reduce 'log' --hidden_units 128 --epochs 350 --warmup 0 --gumbel_decay_epochs 500 --clip_grad 1.0 --weight_decay 0.01 --self_loops --lr 0.0001 --mlp_width_mult 2 --model_type agent --input_mlp --activation_function 'leaky_relu' --negative_slope 0.01 --gumbel_min_temp 0.66666667 --gumbel_temp 0.66666667 --global_agent_pool --bias_attention --discount 0 --lr_dqn 0.0001`

The dataset can be `[MUTAG, PTC_GIN, PROTEINS, IMDB-BINARY, IMDB-MULTI, DD]`. The learning rate of DQN and the ratio of DQN loss accumulated in the total loss can be modified with the `--discount` and `--lr_dqn` flags.
To run the standard grid search on the TU graph classification datasets use the following command with additional `--slurm --gpu_jobs` or `--grid_search` flags.

### OGB
The following command will run training on the OGB datasets. The results report the ROC-AUC score.

`python ogb_mol_3.py --dataset 'ogbg-molhiv' --dropout 0.0 --num_agents 26 --num_steps 16 --batch_size 64 --reduce 'log' --hidden_units 128 --epochs 100 --warmup 0 --gumbel_decay_epochs 50 --clip_grad 1.0 --weight_decay 0.01 --self_loops --lr 0.0001 --mlp_width_mult 2 --input_mlp --activation_function 'leaky_relu' --negative_slope 0.01 --gumbel_min_temp 0.66666667 --gumbel_temp 0.66666667 --global_agent_pool --bias_attention --discount 0 --lr_dqn 0.0001`

The dataset can be `[ogbg-molhiv, ogbg-molpcba]`. To run the code in a fast mode which evaluate every 50 epoch and only report the training loss every epoch, add the `--fast 50` flag. To eable the training progress bar for better monitoring, add the `--bar` flag. Add the `--slurm --gpu_jobs` flags to run all 10 random seeds. 

### QM9
The following command will run training on the QM9. The results report the MAE loss.

`python qm9_3.py --target 0 --dropout 0.0 --num_agents 18 --num_steps 8 --batch_size 32 --reduce 'log' --hidden_units 128 --epochs 350 --warmup 0 --gumbel_decay_epochs 50 --clip_grad 1.0 --weight_decay 0.01 --self_loops --lr 0.0001 --mlp_width_mult 2 --input_mlp --activation_function 'leaky_relu' --negative_slope 0.01 --gumbel_min_temp 0.66666667 --gumbel_temp 0.66666667 --global_agent_pool --bias_attention --edge_negative_slope 0.01 --readout_mlp --complete_graph  --discount 0 --lr_dqn 0.0001`

The targets range from `0` - `11`, selected by the `--target` flag.

## Role of each file

- `model_3.py`: This file contains the implementation of the DQN-AgentNet model.
- `graph_classification_3.py`: This file trains the model on the TU datasets.
- `ogb_mol_3.py`: This file trains the model on the OGB datasets.
- `qm9_3.py`: This file trains the model on the QM9 datasets.

- `results/`: This folder contains the results of the experiments.
  - `checkpoints`: This folder contains the saved models.
  - `curves`: This folder contains the learning curves of the experiments.

- `data/`: These folders contain the datasets used in the experiments.
  - `fgs/`: This folder contains the functional group csv files extracted from the datasets.

- `fgs_process.py`: This file extracts the functional groups from the given datasets.
- `utils.py`: This file contains the utility functions used in the experiments.
