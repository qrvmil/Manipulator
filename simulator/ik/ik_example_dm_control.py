import numpy as np
import mujoco
from dm_control import mujoco as dm_mujoco
from inverse_kinematics import qpos_from_site_pose
    
def create_physics_from_model_data(model, data):
    
    physics = dm_mujoco.Physics.from_xml_path('model/scene.xml')
    
    physics.data.qpos[:] = data.qpos[:]
    physics.data.qvel[:] = data.qvel[:]
    
    physics.forward()
    
    return physics

def solve_ik_example():

    model = mujoco.MjModel.from_xml_path('model/scene.xml')
    data = mujoco.MjData(model)
    
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    physics = create_physics_from_model_data(model, data)
    
    try:
        site_id = physics.model.name2id('pinch', 'site')
    except:
        return
    
    current_pos = physics.named.data.site_xpos['pinch'].copy()
    
    target_pos = current_pos + np.array([0.1, 0.0, 0.0])
    
    joint_names = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate']
    
    result = qpos_from_site_pose(
        physics=physics,
        site_name='pinch',
        target_pos=target_pos,
        joint_names=joint_names,
        tol=1e-6,
        max_steps=100
    )
    
    if result.success:
        print(f"   Number of iterations: {result.steps}")
        print(f"   Final error: {result.err_norm:.6f}")

        data.qpos[:len(result.qpos)] = result.qpos
        mujoco.mj_forward(model, data)
        
        return result.qpos
    else:
        return None

if __name__ == "__main__":
    solve_ik_example()


 