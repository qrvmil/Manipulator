import mujoco
import numpy as np
import time
from collections import namedtuple

IKResult = namedtuple('IKResult', ['qpos', 'err_norm', 'steps', 'success'])

def qpos_from_site_pose_simple(model, data, 
                               site_name,
                               target_pos=None,
                               target_quat=None,
                               joint_indices=None,
                               tol=1e-6,
                               max_steps=100,
                               step_size=0.1):
    """
    Args:
        model: MuJoCo model (mujoco.MjModel)
        data: MuJoCo data (mujoco.MjData)
        site_name: site name (string)
        target_pos: target position [x, y, z] or None
        target_quat: target orientation [w, x, y, z] or None
        joint_indices: joint indices for control or None (all joints)
        tol: tolerance
        max_steps: maximum number of iterations
        step_size: step size of optimization
    
    Returns:
        IKResult with fields qpos, err_norm, steps, success
    """
    
    if target_pos is None and target_quat is None:
        raise ValueError("target_pos or target_quat is required")
    
    try:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    except:
        raise ValueError(f"Site '{site_name}' not found in model")
    
    if joint_indices is None:
        joint_indices = list(range(min(6, model.nq)))
    
    original_qpos = data.qpos.copy()
    
    if target_pos is not None and target_quat is not None:
        nconstr = 6
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        error = np.zeros(6)
    elif target_pos is not None:
        nconstr = 3
        jacp = np.zeros((3, model.nv))
        jacr = None
        error = np.zeros(3)
    else:
        nconstr = 3
        jacp = None
        jacr = np.zeros((3, model.nv))
        error = np.zeros(3)
    
    success = False
    
    for step in range(max_steps):
        mujoco.mj_forward(model, data)
        
        current_pos = data.site_xpos[site_id].copy()
        current_mat = data.site_xmat[site_id].copy().reshape(3, 3)
        
        error_norm = 0.0
        
        if target_pos is not None:
            pos_error = target_pos - current_pos
            error[:3] = pos_error
            error_norm += np.linalg.norm(pos_error)
        
        if target_quat is not None:
            current_quat = np.zeros(4)
            mujoco.mju_mat2Quat(current_quat, current_mat.flatten())

            quat_error = np.zeros(4)
            neg_current_quat = np.zeros(4)
            mujoco.mju_negQuat(neg_current_quat, current_quat)
            mujoco.mju_mulQuat(quat_error, target_quat, neg_current_quat)
            
            rot_error = np.zeros(3)
            mujoco.mju_quat2Vel(rot_error, quat_error, 1.0)
            
            if target_pos is not None:
                error[3:6] = rot_error
            else:
                error[:3] = rot_error
            error_norm += np.linalg.norm(rot_error)
        
        if error_norm < tol:
            success = True
            break
        
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        
        if target_pos is not None and target_quat is not None:
            jac = np.vstack([jacp, jacr])
        elif target_pos is not None:
            jac = jacp
        else:
            jac = jacr
        
        jac_joints = jac[:, joint_indices]
        
        try:
            regularization = 1e-4
            A = jac_joints.T @ jac_joints + regularization * np.eye(len(joint_indices))
            b = jac_joints.T @ error
            delta_q = np.linalg.solve(A, b)
            
            max_delta = step_size
            delta_norm = np.linalg.norm(delta_q)
            if delta_norm > max_delta:
                delta_q *= max_delta / delta_norm
            
            for i, joint_idx in enumerate(joint_indices):
                data.qpos[joint_idx] += delta_q[i]
                
        except np.linalg.LinAlgError:
            print(f"Error in solving system on step {step}")
            break
    
    return IKResult(
        qpos=data.qpos.copy(),
        err_norm=error_norm,
        steps=step,
        success=success
    )
