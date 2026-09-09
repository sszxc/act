"""Forward kinematics for the UR7e arm + HMF proto5 hand (24 DOF), via MuJoCo.

Loads the robot MJCF (sibling repo) and reads RHand_PALM_LINK's world pose after mj_forward.
JOINT_NAMES is the order used by the raw teleop data (trajectories/combined/joint_names_json,
verified against data_0901/data_0902) -- NOT the MJCF's own qpos order (the hand's joints are
laid out differently there), so every value is written into the model by joint name.

Task-space split (per user decision): only the 6 arm joints become a palm pose (pos + 6D
rotation, Zhou et al.); the remaining 18 (wrist yaw/pitch + 16 finger joints) stay joint-space.
"""
import os
import numpy as np
import mujoco

# Honda_proto5_description is bind-mounted to ~/Honda_proto5_description inside the
# docker container (see run_docker.sh), not under ~/code/ like on the host.
MJCF_PATH = os.path.expanduser(
    "~/Honda_proto5_description/mjcf/hmf_hand_proto5_release_right_ur7e_scene_template.xml"
)
PALM_BODY = "RHand_PALM_LINK"

JOINT_NAMES = [
    "RArm_shoulder_pan_joint", "RArm_shoulder_lift_joint", "RArm_elbow_joint",
    "RArm_wrist_1_joint", "RArm_wrist_2_joint", "RArm_wrist_3_joint",
    "RHand_WRZ_joint", "RHand_WRY_joint",
    "RHand_T1Z_joint", "RHand_T1Y_joint", "RHand_T2Y_joint", "RHand_T3Y_joint",
    "RHand_I1Z_joint", "RHand_I1Y_joint", "RHand_I2Y_joint", "RHand_I3Y_joint",
    "RHand_M1Z_joint", "RHand_M1Y_joint", "RHand_M2Y_joint", "RHand_M3Y_joint",
    "RHand_R1Z_joint", "RHand_R1Y_joint", "RHand_R2Y_joint", "RHand_R3Y_joint",
]
N_ARM = 6                                  # first N_ARM entries above become the task-space pose
N_HAND = len(JOINT_NAMES) - N_ARM           # remaining entries stay joint-space (18)
POSE_DIM = 9                                # pos(3) + rot6d(6)
TASKSPACE_DIM = POSE_DIM + N_HAND           # 27

_model = None
_data = None
_qposadr = None


def _load():
    global _model, _data, _qposadr
    if _model is None:
        _model = mujoco.MjModel.from_xml_path(MJCF_PATH)
        _data = mujoco.MjData(_model)
        _qposadr = np.array([_model.joint(n).qposadr[0] for n in JOINT_NAMES])
    return _model, _data, _qposadr


def fk_palm(qpos24):
    """qpos24 (24,), JOINT_NAMES order -> (pos (3,), R (3,3)) of RHand_PALM_LINK in world frame."""
    model, data, qposadr = _load()
    data.qpos[qposadr] = qpos24
    mujoco.mj_forward(model, data)
    bid = model.body(PALM_BODY).id
    return data.xpos[bid].copy(), data.xmat[bid].reshape(3, 3).copy()


def rotmat_to_6d(R):
    """Zhou et al. continuous rotation repr: first two columns of R, flattened to (6,)."""
    return np.asarray(R)[:, :2].reshape(-1).copy()


def sixd_to_rotmat(v6):
    """Inverse of rotmat_to_6d via Gram-Schmidt. rotmat_to_6d flattens the (3,2) column-pair
    row-major (numpy's default), i.e. v6 = [R00,R01,R10,R11,R20,R21] -- interleaved, NOT
    [col0; col1] -- so recovering the two columns needs reshape(3,2), not a 3/3 split."""
    m = np.asarray(v6, dtype=np.float64).reshape(3, 2)
    a1, a2 = m[:, 0], m[:, 1]
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=1)


def rotmat_to_quat_wxyz(R):
    """(3,3) rotation matrix -> (4,) quaternion (w,x,y,z), via mujoco (same convention as
    data.xquat elsewhere in this file). Used to report a task-space prediction as a pose for
    UDP/downstream consumers, instead of the internal rot6d training representation."""
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.asarray(R, dtype=np.float64).reshape(-1))
    return quat


def task_state(qpos24):
    """qpos24 (24,) -> 27-dim absolute task-space state: palm pos(3) + palm rot6d(6) + the
    remaining 18 joint values (wrist yaw/pitch + fingers), unchanged."""
    qpos24 = np.asarray(qpos24)
    pos, R = fk_palm(qpos24)
    return np.concatenate([pos, rotmat_to_6d(R), qpos24[N_ARM:]])


def task_state_batch(qpos_TxN):
    """qpos_TxN (T, 24) -> (T, 27), row-by-row (mujoco has no batched CPU FK)."""
    qpos_TxN = np.asarray(qpos_TxN)
    return np.stack([task_state(qpos_TxN[t]) for t in range(qpos_TxN.shape[0])], axis=0)


def task_action_delta_batch(qpos_start24, qpos_targets24):
    """27-dim delta action for each target row, relative to qpos_start24 (mirrors the existing
    action_repr='delta' semantics: delta relative to the chunk's start qpos, not frame-to-frame).
    pos delta is in world frame; rotation delta is target-in-start-frame (R_start.T @ R_target),
    encoded as 6D; the trailing 18 dims are a plain qpos subtraction (still joint-space)."""
    qpos_start24 = np.asarray(qpos_start24)
    qpos_targets24 = np.asarray(qpos_targets24)
    pos0, R0 = fk_palm(qpos_start24)
    K = qpos_targets24.shape[0]
    out = np.zeros((K, TASKSPACE_DIM), dtype=np.float64)
    for k in range(K):
        pos1, R1 = fk_palm(qpos_targets24[k])
        out[k, :3] = pos1 - pos0
        out[k, 3:9] = rotmat_to_6d(R0.T @ R1)
    out[:, 9:] = qpos_targets24[:, N_ARM:] - qpos_start24[N_ARM:]
    return out


if __name__ == "__main__":
    # rotmat_to_6d/sixd_to_rotmat/rotmat_to_quat_wxyz round-trip, random rotations (this
    # caught a real bug once: sixd_to_rotmat assumed [col0;col1] concatenation, but
    # reshape(-1) on a (3,2) array interleaves the two columns instead -- silent NaNs on
    # decode, invisible to any check that never calls sixd_to_rotmat).
    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(200):
        R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        R = R * np.linalg.det(R)  # flip to a proper rotation (det=+1) if qr gave a reflection
        R_back = sixd_to_rotmat(rotmat_to_6d(R))
        max_err = max(max_err, np.abs(R - R_back).max())
    print(f"rotmat<->6d round-trip max abs err over 200 random rotations: {max_err:.2e} "
          f"({'OK' if max_err < 1e-8 else 'FAIL'})")
    print("quat at identity (expect [1,0,0,0]):", rotmat_to_quat_wxyz(np.eye(3)))

    # Cheap sanity checks -- run after any change to JOINT_NAMES/MJCF_PATH/PALM_BODY.
    zero = np.zeros(len(JOINT_NAMES))
    pos, R = fk_palm(zero)
    print("palm pose at qpos=0:")
    print("  pos  ", pos)
    # Expected (from the mocap default in hmf_hand_proto5_release_right_ur7e_mocap.xml, which
    # was set to match the robot's home pose): pos ~= [0.8172, 0.3984, 0.6548].
    expected_home_pos = np.array([0.8172, 0.3984, 0.6548])
    print("  matches mocap-default home pos:", np.allclose(pos, expected_home_pos, atol=1e-3))

    print("\nper-arm-joint perturbation (+0.1 rad), palm displacement:")
    for i in range(N_ARM):
        q = zero.copy()
        q[i] = 0.1
        p, _ = fk_palm(q)
        print(f"  {JOINT_NAMES[i]:28s} d_pos={p - pos}")
