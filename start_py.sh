
# initialize conda
.  /opt/anaconda3/etc/profile.d/conda.sh
conda activate AgentNet
export LD_LIBRARY_PATH=/home/uceeyc6/.conda/envs/AgentNet/lib

# run synetic data
nohup python synthetic.py --dataset 'fourcycles' --num_seeds 10 --verbose --dropout 0.0 --num_agents 16 --num_steps 16 --batch_size 200 --reduce 'log' --hidden_units 128 --epochs 10000 --warmup 0 --gumbel_decay_epochs 1 --clip_grad 1.0 --weight_decay 0.01 --self_loops --lr 0.0001 --mlp_width_mult 2 --model_type agent --input_mlp --activation_function 'leaky_relu' --negative_slope 0.01 --gumbel_min_temp 0.66666667 --gumbel_temp 0.66666667 --global_agent_pool --bias_attention >log/synthetic/agent_fourcycles.log &

# run real data
# nohup python graph_classification.py --dataset 'PROTEINS' --verbose --dropout 0.0 --num_agents 39 --num_steps 16 --batch_size 32 --reduce 'log' --hidden_units 128 --epochs 350 --warmup 0 --gumbel_decay_epochs 500 --clip_grad 1.0 --weight_decay 0.01 --self_loops --lr 0.0001 --mlp_width_mult 2 --model_type agent --input_mlp --activation_function 'leaky_relu' --negative_slope 0.01 --gumbel_min_temp 0.66666667 --gumbel_temp 0.66666667 --global_agent_pool --bias_attention >log/graph_classification/agent_PROTEINS.log &

nohup python graph_classification.py --dataset 'MUTAG' --verbose --dropout 0.0 --num_agents 18 --num_steps 16 --batch_size 32 --reduce 'log' --hidden_units 128 --epochs 20 --warmup 0 --gumbel_decay_epochs 500 --clip_grad 1.0 --weight_decay 0.01 --self_loops --lr 0.0001 --mlp_width_mult 2 --model_type agent --input_mlp --activation_function 'leaky_relu' --negative_slope 0.01 --gumbel_min_temp 0.66666667 --gumbel_temp 0.66666667 --global_agent_pool --bias_attention >log/test/agent_q.log &

# run rl code
python graph_classification_rl.py --dataset 'MUTAG' --verbose --dropout 0.0 --num_agents 18 --num_steps 16 --batch_size 32 --reduce 'log' --hidden_units 128 --epochs 20 --warmup 0 --gumbel_decay_epochs 500 --clip_grad 1.0 --weight_decay 0.01 --self_loops --lr 0.0001 --mlp_width_mult 2 --model_type agent --input_mlp --activation_function 'leaky_relu' --negative_slope 0.01 --gumbel_min_temp 0.66666667 --gumbel_temp 0.66666667 --global_agent_pool --bias_attention

nohup python graph_classification_rl.py --dataset 'MUTAG' --verbose --dropout 0.0 --num_agents 18 --num_steps 16 --batch_size 32 --reduce 'log' --hidden_units 128 --epochs 350 --warmup 0 --gumbel_decay_epochs 500 --clip_grad 1.0 --weight_decay 0.01 --self_loops --lr 0.0001 --mlp_width_mult 2 --model_type agent --input_mlp --activation_function 'leaky_relu' --negative_slope 0.01 --gumbel_min_temp 0.66666667 --gumbel_temp 0.66666667 --global_agent_pool --bias_attention >log/graph_classification_rl/agent_RL_MUTAG.log &

nohup python graph_classification_rl.py --dataset 'PTC' --verbose --dropout 0.0 --num_agents 18 --num_steps 16 --batch_size 32 --reduce 'log' --hidden_units 128 --epochs 350 --warmup 0 --gumbel_decay_epochs 500 --clip_grad 1.0 --weight_decay 0.01 --self_loops --lr 0.0001 --mlp_width_mult 2 --model_type agent --input_mlp --activation_function 'leaky_relu' --negative_slope 0.01 --gumbel_min_temp 0.66666667 --gumbel_temp 0.66666667 --global_agent_pool --bias_attention >log/graph_classification_rl/agent_RL_MUTAG.log &


# run qm9 dataset
nohup python qm9_1.py --target 0 --dropout 0.0 --num_agents 18 --num_steps 8 --batch_size 32 --reduce 'log' --hidden_units 128 --epochs 250 --warmup 0 --gumbel_decay_epochs 50 --clip_grad 1.0 --weight_decay 0.01 --self_loops --lr 0.0001 --mlp_width_mult 2 --input_mlp --activation_function 'leaky_relu' --negative_slope 0.01 --gumbel_min_temp 0.66666667 --gumbel_temp 0.66666667 --global_agent_pool --bias_attention --edge_negative_slope 0.01 --readout_mlp --complete_graph >log/qm9/agent_qm9_1_rl.log &
python qm9_2.py --target 0 --dropout 0.0 --num_agents 18 --num_steps 8 --batch_size 32 --reduce 'log' --hidden_units 128 --epochs 350 --warmup 0 --gumbel_decay_epochs 50 --clip_grad 1.0 --weight_decay 0.01 --self_loops --lr 0.0001 --mlp_width_mult 2 --input_mlp --activation_function 'leaky_relu' --negative_slope 0.01 --gumbel_min_temp 0.66666667 --gumbel_temp 0.66666667 --global_agent_pool --bias_attention --edge_negative_slope 0.01 --readout_mlp --complete_graph

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113


python qm9.py --target 0 --dropout 0.0 --num_agents 18 --num_steps 8 --batch_size 32 --reduce 'log' --hidden_units 128 --epochs 350 --warmup 0 --gumbel_decay_epochs 50 --clip_grad 1.0 --weight_decay 0.01 --self_loops --lr 0.0001 --mlp_width_mult 2 --input_mlp --activation_function 'leaky_relu' --negative_slope 0.01 --gumbel_min_temp 0.66666667 --gumbel_temp 0.66666667 --global_agent_pool --bias_attention --edge_negative_slope 0.01 --readout_mlp --complete_graph

nohup python qm9_2.py --target 0 --dropout 0.0 --num_agents 18 --num_steps 8 --batch_size 32 --reduce 'log' --hidden_units 128 --epochs 350 --warmup 0 --gumbel_decay_epochs 50 --clip_grad 1.0 --weight_decay 0.01 --self_loops --lr 0.0001 --mlp_width_mult 2 --input_mlp --activation_function 'leaky_relu' --negative_slope 0.01 --gumbel_min_temp 0.66666667 --gumbel_temp 0.66666667 --global_agent_pool --bias_attention --edge_negative_slope 0.01 --readout_mlp --complete_graph >log/qm9/agent_qm9_2.log &

nohup python qm9_2.py --target 0 --dropout 0.0 --num_agents 18 --num_steps 8 --batch_size 16 --reduce 'log' --hidden_units 128 --epochs 350 --warmup 0 --gumbel_decay_epochs 50 --clip_grad 1.0 --weight_decay 0.01 --self_loops --lr 0.0001 --mlp_width_mult 2 --input_mlp --activation_function 'leaky_relu' --negative_slope 0.01 --gumbel_min_temp 0.66666667 --gumbel_temp 0.66666667 --global_agent_pool --bias_attention --edge_negative_slope 0.01 --readout_mlp --complete_graph >log/qm9/agent_qm9_2.log &

python qm9_2.py --target 0 --dropout 0.0 --num_agents 18 --num_steps 8 --batch_size 8 --reduce 'log' --hidden_units 128 --epochs 100 --warmup 0 --gumbel_decay_epochs 50 --clip_grad 1.0 --weight_decay 0.01 --self_loops --lr 0.0001 --mlp_width_mult 2 --input_mlp --activation_function 'leaky_relu' --negative_slope 0.01 --gumbel_min_temp 0.66666667 --gumbel_temp 0.66666667 --global_agent_pool --bias_attention --edge_negative_slope 0.01 --readout_mlp --complete_graph

nohup python qm9_2.py --target 0 --dropout 0.0 --num_agents 18 --num_steps 8 --batch_size 8 --reduce 'log' --hidden_units 128 --epochs 100 --warmup 0 --gumbel_decay_epochs 50 --clip_grad 1.0 --weight_decay 0.01 --self_loops --lr 0.0001 --mlp_width_mult 2 --input_mlp --activation_function 'leaky_relu' --negative_slope 0.01 --gumbel_min_temp 0.66666667 --gumbel_temp 0.66666667 --global_agent_pool --bias_attention --edge_negative_slope 0.01 --readout_mlp --complete_graph >log/qm9/agent_qm9_2.log & 

nohup python qm9_0.py --target 0 --dropout 0.0 --num_agents 18 --num_steps 8 --batch_size 32 --reduce 'log' --hidden_units 128 --epochs 100 --warmup 0 --gumbel_decay_epochs 50 --clip_grad 1.0 --weight_decay 0.01 --self_loops --lr 0.0001 --mlp_width_mult 2 --input_mlp --activation_function 'leaky_relu' --negative_slope 0.01 --gumbel_min_temp 0.66666667 --gumbel_temp 0.66666667 --global_agent_pool --bias_attention --edge_negative_slope 0.01 --readout_mlp --complete_graph >log/qm9/agent_qm9_0.log & 

python qm9_3.py --target 0 --dropout 0.0 --num_agents 18 --num_steps 8 --batch_size 32 --reduce 'log' --hidden_units 128 --epochs 100 --warmup 0 --gumbel_decay_epochs 50 --clip_grad 1.0 --weight_decay 0.01 --self_loops --lr 0.0001 --mlp_width_mult 2 --input_mlp --activation_function 'leaky_relu' --negative_slope 0.01 --gumbel_min_temp 0.66666667 --gumbel_temp 0.66666667 --global_agent_pool --bias_attention --edge_negative_slope 0.01 --readout_mlp --complete_graph

nohup python qm9_3.py --target 0 --dropout 0.0 --num_agents 18 --num_steps 8 --batch_size 32 --reduce 'log' --hidden_units 128 --epochs 100 --warmup 0 --gumbel_decay_epochs 50 --clip_grad 1.0 --weight_decay 0.01 --self_loops --lr 0.0001 --mlp_width_mult 2 --input_mlp --activation_function 'leaky_relu' --negative_slope 0.01 --gumbel_min_temp 0.66666667 --gumbel_temp 0.66666667 --global_agent_pool --bias_attention --edge_negative_slope 0.01 --readout_mlp --complete_graph >log/qm9/agent_qm9_3.log &

nohup python qm9_3.py --target 1 --dropout 0.0 --num_agents 18 --num_steps 8 --batch_size 32 --reduce 'log' --hidden_units 128 --epochs 100 --warmup 0 --gumbel_decay_epochs 50 --clip_grad 1.0 --weight_decay 0.01 --self_loops --lr 0.0001 --mlp_width_mult 2 --input_mlp --activation_function 'leaky_relu' --negative_slope 0.01 --gumbel_min_temp 0.66666667 --gumbel_temp 0.66666667 --global_agent_pool --bias_attention --edge_negative_slope 0.01 --readout_mlp --complete_graph >log/qm9/target1/agent_qm9_3_1.log &


# run ogb-molhiv
nohup python ogb_mol.py --dataset 'ogbg-molhiv' --dropout 0.0 --num_agents 26 --num_steps 16 --batch_size 64 --reduce 'log' --hidden_units 128 --epochs 100 --warmup 0 --gumbel_decay_epochs 50 --clip_grad 1.0 --weight_decay 0.01 --self_loops --lr 0.0001 --mlp_width_mult 2 --input_mlp --activation_function 'leaky_relu' --negative_slope 0.01 --gumbel_min_temp 0.66666667 --gumbel_temp 0.66666667 --global_agent_pool --bias_attention >log/ogb-mol/agent_ogb-molhiv_0.log &
nohup python ogb_mol.py --dataset 'ogbg-molpcba' --dropout 0.0 --num_agents 26 --num_steps 16 --batch_size 64 --reduce 'log' --hidden_units 128 --epochs 100 --warmup 0 --gumbel_decay_epochs 50 --clip_grad 1.0 --weight_decay 0.01 --self_loops --lr 0.0001 --mlp_width_mult 2 --input_mlp --activation_function 'leaky_relu' --negative_slope 0.01 --gumbel_min_temp 0.66666667 --gumbel_temp 0.66666667 --global_agent_pool --bias_attention >log/ogb-mol/agent_ogb-molpcba_0.log &

nohup python ogb_mol_1.py --dataset 'ogbg-molhiv' --dropout 0.0 --num_agents 26 --num_steps 16 --batch_size 64 --reduce 'log' --hidden_units 128 --epochs 100 --warmup 0 --gumbel_decay_epochs 50 --clip_grad 1.0 --weight_decay 0.01 --self_loops --lr 0.0001 --mlp_width_mult 2 --input_mlp --activation_function 'leaky_relu' --negative_slope 0.01 --gumbel_min_temp 0.66666667 --gumbel_temp 0.66666667 --global_agent_pool --bias_attention --num_workers 5 >log/ogb-mol/agent_ogb-molhiv_1.log &


nohup python qm9_0_zinc.py --target 0 --dropout 0.0 --num_agents 37 --num_steps 8 --batch_size 128 --reduce 'log' --hidden_units 128 --epochs 100 --warmup 0 --gumbel_decay_epochs 50 --clip_grad 1.0 --weight_decay 0.01 --self_loops --lr 0.0001 --mlp_width_mult 2 --input_mlp --activation_function 'leaky_relu' --negative_slope 0.01 --gumbel_min_temp 0.66666667 --gumbel_temp 0.66666667 --global_agent_pool --bias_attention --edge_negative_slope 0.01 --readout_mlp --complete_graph >log/qm9/agent_qm9_0_zinc.log & 

nohup python ogb_mol_2.py --dataset 'ogbg-molhiv' --dropout 0.0 --num_agents 26 --num_steps 16 --batch_size 64 --reduce 'log' --hidden_units 128 --epochs 100 --warmup 0 --gumbel_decay_epochs 50 --clip_grad 1.0 --weight_decay 0.01 --self_loops --lr 0.0001 --mlp_width_mult 2 --input_mlp --activation_function 'leaky_relu' --negative_slope 0.01 --gumbel_min_temp 0.66666667 --gumbel_temp 0.66666667 --global_agent_pool --bias_attention >log/ogb-mol/agent_ogb-molhiv_2.log &

python ogb_mol_3.py --dataset 'ogbg-molhiv' --dropout 0.0 --num_agents 26 --num_steps 16 --batch_size 64 --reduce 'log' --hidden_units 128 --epochs 100 --warmup 0 --gumbel_decay_epochs 50 --clip_grad 1.0 --weight_decay 0.01 --self_loops --lr 0.0001 --mlp_width_mult 2 --input_mlp --activation_function 'leaky_relu' --negative_slope 0.01 --gumbel_min_temp 0.66666667 --gumbel_temp 0.66666667 --global_agent_pool --bias_attention >log/ogb-mol/agent_ogb-molhiv_3.log &

python qm9_3.py --target 3 --dropout 0.0 --num_agents 18 --num_steps 8 --batch_size 32 --reduce 'log' --hidden_units 128 --epochs 100 --warmup 0 --gumbel_decay_epochs 50 --clip_grad 1.0 --weight_decay 0.01 --self_loops --lr 0.0001 --mlp_width_mult 2 --input_mlp --activation_function 'leaky_relu' --negative_slope 0.01 --gumbel_min_temp 0.66666667 --gumbel_temp 0.66666667 --global_agent_pool --bias_attention --edge_negative_slope 0.01 --readout_mlp --complete_graph >log/qm9/new/agent_qm9_3_3.log