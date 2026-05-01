import pathlib

### Task parameters
DATA_DIR = 'data'
DEFAULT_STATE_DIM = 14
STATE_DIM_ALLEGRO = 22
ROOT_DIM = 6   # allegro hand root 6dof (x,y,z,rx,ry,rz)
FINGER_DIM = 16  # allegro finger joints
ENV_FAMILY_METAWORLD = "metaworld"
ENV_FAMILY_ALLEGRO = "allegro"
ENV_FAMILY_HMF_PROTO5_HAND = "hmf_proto5_hand"
DEX_ALLEGRO_XML_PATH = '/mnt/1tb1/xuechao/MuJoCo-Asset-Pipeline/asset/scene/freejoint/teleop_scene_left_077_rubiks_cube/teleop_scene_left_077_rubiks_cube.xml'
# DEX_ALLEGRO_XML_PATH = "/mnt/1tb1/xuechao/MuJoCo-Asset-Pipeline/asset/scene/freejoint/teleop_scene_left_035_power_drill/teleop_scene_left_035_power_drill.xml"
HMF_PROTO5_STATE_DIM = 24
HMF_PROTO5_ACTION_DIM = 23  # mocap_pos(3) + mocap_quat(4, wxyz) + finger_ctrl(16)
HMF_PROTO5_CTRL_DIM = 16

HMF_PROTO5_RANDOM_RESET_CONFIGS = {
    "pick_place_v3": {
        "random_obj_goal": [
            {
                "name": "obj",
                "type": "body",
                "position_ranges": [[-0.2, 0.2], [0.5, 0.8], [0.66, 0.66]],
            },
            {
                "name": "goal",
                "type": "site",
                "position_ranges": [[-0.3, 0.3], [0.5, 0.8], [0.7, 0.9]],
            },
        ],
    },
    "drawer": {
        "random_obj_goal": [
            {
                "name": "drawer",
                "type": "body",
                "position_ranges": [[-0.2, 0.2], [0.8, 1.0], [0.6, 0.6]],
            },
            {
                "name": "goal",
                "type": "site",
                "position_ranges": [[-0.3, 0.3], [0.5, 0.8], [0.7, 0.9]],
            },
        ],
        "task_reset_joint": {
            "enabled": True,
            "name": "goal_slidey",
            "value": 0.0,
        },
    },
    "basketball": {
        "random_obj_goal": [
            {
                "name": "ball",
                "type": "body",
                "position_ranges": [[-0.2, 0.2], [0.5, 0.7], [0.6, 0.6]],
            },
            {
                "name": "goal",
                "type": "body",
                "position_ranges": [[-0.3, 0.3], [0.7, 0.8], [0.6, 0.6]],
            },
        ],
    },
}

SIM_TASK_CONFIGS = {
    "sim_transfer_cube_scripted": {
        "dataset_dir": DATA_DIR + "/sim_transfer_cube_scripted",
        "num_episodes": 50,
        "episode_len": 400,
        "camera_names": ["top"],
        "env_family": ENV_FAMILY_METAWORLD,
    },
    "sim_transfer_cube_human": {
        "dataset_dir": DATA_DIR + "/sim_transfer_cube_human",
        "num_episodes": 50,
        "episode_len": 400,
        "camera_names": ["top"],
        "env_family": ENV_FAMILY_METAWORLD,
    },
    "sim_insertion_scripted": {
        "dataset_dir": DATA_DIR + "/sim_insertion_scripted",
        "num_episodes": 50,
        "episode_len": 400,
        "camera_names": ["top"],
        "env_family": ENV_FAMILY_METAWORLD,
    },
    "sim_insertion_human": {
        "dataset_dir": DATA_DIR + "/sim_insertion_human",
        "num_episodes": 50,
        "episode_len": 500,
        "camera_names": ["top"],
        "env_family": ENV_FAMILY_METAWORLD,
    },
    "sim_dexgrasp_cube_teleop_wrong_offset": {
        "dataset_dir": DATA_DIR + "/sim_dexgrasp_cube_teleop/20260304_123721",
        "num_episodes": 51,
        "episode_len": 400,
        "camera_names": ["default_cam", "wrist_cam"],
        "state_dim": STATE_DIM_ALLEGRO,
        "env_family": ENV_FAMILY_ALLEGRO,
    },
    "sim_dexgrasp_cube_teleop_minimal": {
        "dataset_dir": DATA_DIR + "/sim_dexgrasp_cube_teleop/20260306_152210",
        "num_episodes": 5,
        "episode_len": 400,
        "camera_names": ["default_cam", "wrist_cam"],
        "state_dim": STATE_DIM_ALLEGRO,
        "env_family": ENV_FAMILY_ALLEGRO,
    },
    "sim_dexgrasp_cube_teleop": {
        "dataset_dir": DATA_DIR + "/sim_dexgrasp_cube_teleop/20260306_153749",
        "num_episodes": 49,
        "episode_len": 400,
        "camera_names": ["default_cam", "wrist_cam"],
        "state_dim": STATE_DIM_ALLEGRO,
        "env_family": ENV_FAMILY_ALLEGRO,
    },
    "sim_dexgrasp_pca_cube_teleop": {
        "dataset_dir": DATA_DIR + "/sim_dexgrasp_cube_teleop/20260306_153749",
        "num_episodes": 49,
        "episode_len": 400,
        "camera_names": ["default_cam", "wrist_cam"],
        "state_dim": STATE_DIM_ALLEGRO,
        "action_dim": ROOT_DIM + 3,  # root 6 + finger 3 PCs
        "pca_finger_dim": 3,
        "env_family": ENV_FAMILY_ALLEGRO,
    },
    # HMF proto5 tasks share the same env family but load task-specific XMLs.
    "sim_hmf_proto5_pick_place_v3": {
        "dataset_dir": DATA_DIR + "/sim_hmf_proto5_teleop/pick_place_v3/20260427_175415_50traj",
        "num_episodes": 50,
        "episode_len": 400,
        "camera_names": ["topview", "corner"],
        "state_dim": HMF_PROTO5_STATE_DIM,
        "action_dim": HMF_PROTO5_ACTION_DIM,
        "env_family": ENV_FAMILY_HMF_PROTO5_HAND,
        "xml_path": "/home/lab/Documents/proto5_description/mjcf/hmf_hand_proto5_release_right_ur7e_scene_pick_place_v3.xml",
        "random_reset": HMF_PROTO5_RANDOM_RESET_CONFIGS["pick_place_v3"],
    },
    "sim_hmf_proto5_drawer": {
        "dataset_dir": DATA_DIR + "/sim_hmf_proto5_teleop/drawer/20260429_141927_30traj",
        "num_episodes": 30,
        "episode_len": 400,
        "camera_names": ["topview", "corner"],
        "state_dim": HMF_PROTO5_STATE_DIM,
        "action_dim": HMF_PROTO5_ACTION_DIM,
        "env_family": ENV_FAMILY_HMF_PROTO5_HAND,
        "xml_path": "/home/lab/Documents/proto5_description/mjcf/hmf_hand_proto5_release_right_ur7e_scene_drawer.xml",
        "random_reset": HMF_PROTO5_RANDOM_RESET_CONFIGS["drawer"],
    },
    "sim_hmf_proto5_basketball": {
        "dataset_dir": DATA_DIR + "/sim_hmf_proto5_teleop/basketball/20260429_152808_20traj",
        "num_episodes": 20,
        "episode_len": 400,
        "camera_names": ["topview", "corner"],
        "state_dim": HMF_PROTO5_STATE_DIM,
        "action_dim": HMF_PROTO5_ACTION_DIM,
        "env_family": ENV_FAMILY_HMF_PROTO5_HAND,
        "xml_path": "/home/lab/Documents/proto5_description/mjcf/hmf_hand_proto5_release_right_ur7e_scene_basketball.xml",
        "random_reset": HMF_PROTO5_RANDOM_RESET_CONFIGS["basketball"],
    },
}

### Simulation envs fixed constants
DT = 0.02
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]

XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets/' # note: absolute path

# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = 0.3083
MASTER_GRIPPER_JOINT_CLOSE = -0.6842
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

############################ Helper functions ############################

MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE)/2
