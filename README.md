# ACT: Action Chunking with Transformers


## Installation

```
conda create -n aloha -f environment.yml
conda activate aloha
cd act/detr && pip install -e .
```


## Training with Teleoperation Data from Dex-Retargeting

Use `visualize_episodes.py` to generate videos and qpos plots for quick sanity checking of your data.  
The dataset path for each task is configured in `constants.py`.  
Evaluation results will be stored in a subfolder in the directory specified by `ckpt_dir`.

```
# for visualization
python visualize_episodes.py --dataset_dir <data save dir> --episode_idx 0
# for training
python imitate_episodes.py --ckpt_dir results/sim_transfer_cube_scripted task_name=sim_transfer_cube_scripted policy_class=ACT kl_weight=10 chunk_size=100 hidden_dim=512 batch_size=8 dim_feedforward=3200 num_epochs=200 lr=1e-5 seed=0
# for evaluation
<training command> --eval --temporal_agg
```

## In-Context Learning

This fork supports **test-time conditioning** without updating policy weights: fix a CVAE latent `z` for the whole rollout (`--latent_z_sample`), fix simulator object pose (and optionally `fixed_init_qpos` in the same list form as dataset qpos) via Hydra overrides, **search visual FiLM** (gamma/beta) with black-box optimization (`optimize_film_params.py`), and record best-vs-baseline rollouts (`render_film_rollout_videos.py`). `dataset_stats.pkl` must live next to the checkpoint when evaluating.

### LLM API key (safe storage)

Some modes (e.g. `optimize_film_params.py --method llm`) require an OpenAI-compatible API key. **Do not hardcode keys in code.** Set them via environment variables (recommended) or a local `.env` file.

```
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://openai.rc.asu.edu/v1"   # optional
```

For local development, you can copy `.env.example` to `.env` and fill in `OPENAI_API_KEY`. `.env` is ignored by git.

```
# Eval with a fixed latent z for the entire episode (dim = latent_z_dim, default 32)
python imitate_episodes.py --eval --temporal_agg --ckpt_dir <ckpt_dir> \
  --latent_z_sample "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0" \
  task_name=<task> policy_class=ACT kl_weight=10 chunk_size=100 hidden_dim=512 batch_size=8 dim_feedforward=3200 num_epochs=200 lr=1e-5 seed=0

# Eval with fixed object pose (vector length is task-specific; add fixed_init_qpos=[...] if needed)
python imitate_episodes.py --eval --ckpt_dir <ckpt_dir> \
  fixed_object_pose=[0.1,0.5,0.05,1,0,0,0] \
  task_name=<task> policy_class=ACT kl_weight=10 chunk_size=100 hidden_dim=512 batch_size=8 dim_feedforward=3200 num_epochs=200 lr=1e-5 seed=0

# Search FiLM (gamma/beta) with ARS or CMA-ES under fixed pose; optional full-dim FiLM: --rsp_subspace_dim 0
pip install cma matplotlib   # only if using --method cma
python optimize_film_params.py --ckpt <ckpt_dir>/policy_best.ckpt --task_name <sim_task> \
  --fixed_object_pose "0.1,0.5,0.05,1,0,0,0" --method ars --ars_iters 50 --ars_pairs 4 \
  --output_dir tmp/film_search --parallel 4

python optimize_film_params.py --ckpt <ckpt_dir>/policy_best.ckpt --task_name <sim_task> \
  --fixed_object_pose "0.1,0.5,0.05,1,0,0,0" --method cma --cma_maxiter 50 --cma_popsize 8 \
  --output_dir tmp/film_search_cma

# After a search: record two videos (optimized FiLM vs identity gamma=1,beta=0) using run_meta.json + best_film_only.pt
python render_film_rollout_videos.py --run_dir tmp/film_search
```


<div align="center">
  <h1 align="center"> 「Original」 </h1>
</div>

### *New*: [ACT tuning tips](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit?usp=sharing)
TL;DR: if your ACT policy is jerky or pauses in the middle of an episode, just train for longer! Success rate and smoothness can improve way after loss plateaus.

#### Project Website: https://tonyzhaozh.github.io/aloha/

This repo contains the implementation of ACT, together with 2 simulated environments:
Transfer Cube and Bimanual Insertion. You can train and evaluate ACT in sim or real.
For real, you would also need to install [ALOHA](https://github.com/tonyzhaozh/aloha).

### Updates:
You can find all scripted/human demo for simulated environments [here](https://drive.google.com/drive/folders/1gPR03v05S1xiInoVJn7G7VJ9pDCnxq9O?usp=share_link).


### Repo Structure
- ``imitate_episodes.py`` Train and Evaluate ACT
- ``policy.py`` An adaptor for ACT policy
- ``detr`` Model definitions of ACT, modified from DETR
- ``sim_env.py`` Mujoco + DM_Control environments with joint space control
- ``ee_sim_env.py`` Mujoco + DM_Control environments with EE space control
- ``scripted_policy.py`` Scripted policies for sim environments
- ``constants.py`` Constants shared across files
- ``utils.py`` Utils such as data loading and helper functions
- ``visualize_episodes.py`` Save videos from a .hdf5 dataset


### Installation

```
conda create -n aloha -f environment.yml
conda activate aloha
cd act/detr && pip install -e .
```

### Example Usages

To set up a new terminal, run:

```
conda activate aloha
cd <path to act repo>
```

### Simulated experiments

We use ``sim_transfer_cube_scripted`` task in the examples below. Another option is ``sim_insertion_scripted``.
To generated 50 episodes of scripted data, run:

```
python3 record_sim_episodes.py \
--task_name sim_transfer_cube_scripted \
--dataset_dir <data save dir> \
--num_episodes 50
```

To can add the flag ``--onscreen_render`` to see real-time rendering.
To visualize the episode after it is collected, run

```
python3 visualize_episodes.py --dataset_dir <data save dir> --episode_idx 0
```

To train ACT:
    
```
python3 imitate_episodes.py --ckpt_dir results/sim_transfer_cube_scripted   task_name=sim_transfer_cube_scripted   policy_class=ACT kl_weight=10 chunk_size=100 hidden_dim=512 batch_size=8   dim_feedforward=3200 num_epochs=200 lr=1e-5 seed=0
```

To evaluate the policy, run the same command but add ``--eval``. This loads the best validation checkpoint.
The success rate should be around 90% for transfer cube, and around 50% for insertion.
To enable temporal ensembling, add flag ``--temporal_agg``.

```
python3 imitate_episodes.py --ckpt_dir results/sim_transfer_cube_scripted   task_name=sim_transfer_cube_scripted   policy_class=ACT kl_weight=10 chunk_size=100 hidden_dim=512 batch_size=8   dim_feedforward=3200 num_epochs=200 lr=1e-5 seed=0 --eval --temporal_agg
```

Videos will be saved to ``<ckpt_dir>`` for each rollout.
You can also add ``--onscreen_render`` to see real-time rendering during evaluation.

For real-world data where things can be harder to model, train for at least 5000 epochs or 3-4 times the length after the loss has plateaued.
Please refer to [tuning tips](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit?usp=sharing) for more info.

