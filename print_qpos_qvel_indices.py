import mujoco
import numpy as np

def get_qpos_qvel_mapping(model):
    mapping = []

    for j in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        body_id = model.jnt_bodyid[j]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        joint_type = model.jnt_type[j]

        # --- qpos info ---
        qpos_start = model.jnt_qposadr[j]
        if joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            qpos_dof = 1
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            qpos_dof = 4   # quaternion
        elif joint_type == mujoco.mjtJoint.mjJNT_FREE:
            qpos_dof = 7   # xyz + quat
        else:
            raise ValueError("Unknown joint type")
        qpos_indices = np.arange(qpos_start, qpos_start + qpos_dof)

        # --- qvel info ---
        qvel_start = model.jnt_dofadr[j]
        if joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            qvel_dof = 1
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            qvel_dof = 3   # angular velocity
        elif joint_type == mujoco.mjtJoint.mjJNT_FREE:
            qvel_dof = 6   # linear + angular velocity
        qvel_indices = np.arange(qvel_start, qvel_start + qvel_dof)

        mapping.append({
            "joint_id": j,
            "joint_name": joint_name,
            "body_id": body_id,
            "body_name": body_name,
            "joint_type": joint_type,
            "qpos_indices": qpos_indices,
            "qvel_indices": qvel_indices,
        })
    
    return mapping

model = mujoco.MjModel.from_xml_path("3D-Diffusion-Policy/diffusion_policy_3d/env/diana/assets/setup_final_poncho.xml")
mapping = get_qpos_qvel_mapping(model)

for m in mapping:
    print(f"Joint {m['joint_name']} (body {m['body_name']}):")
    print(f"  qpos indices: {m['qpos_indices']}")
    print(f"  qvel indices: {m['qvel_indices']}")
