"""Real-robot task configs. imitate_episodes.py imports TASK_CONFIGS from here for any
task_name that doesn't start with 'sim_' (mirrors the original ALOHA repo's layout, so no
change to imitate_episodes.py's import is needed)."""

DATA_DIR = 'data'

TASK_CONFIGS = {
    # UR arm (6 dof) + HMF proto5 right hand (18 dof), state/action = combined joint positions.
    # Built by convert_teleop_dataset.py from data/20260901_good_data_90hz/ + merge_teleop_dataset.py.
    # See NOTE (or ask) for the qpos[t+1]-as-action caveat: there's no independent commanded-joint
    # channel in the source data, so action is a hindsight shift of observed qpos, not a true command.
    "real_pick_yellow_bottle": {
        "dataset_dir": DATA_DIR + "/real_pick_yellow_bottle/good_0901_c20",
        "num_episodes": 20,
        "episode_len": 1578,  # longest episode in good_20; used for eval rollout cap, not training
        "camera_names": ["left", "top"],
        "state_dim": 24,
        "action_dim": 24,
    },
}
