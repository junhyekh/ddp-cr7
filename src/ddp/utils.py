from pathlib import Path
from typing import Optional
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import joblib
import numpy as np

import pink

"""
    G1 loading based on https://github.com/Gepetto/example-robot-data
"""

class G1Loader:
    path: Optional[str] = None
    urdf_filename: str = "g1_29dof_rev_1_0.urdf"
    has_rotor_parameters = False
    free_flyer = True
    model_path = None

    def __init__(self, rpy=None):
        if self.path is None:
            self.path = Path(__file__).parents[2] / 'data/robots/g1'
        urdf_path = Path(self.path) / self.urdf_filename
        print(urdf_path)
        if not urdf_path.exists():
            raise ValueError(f"No such {self.urdf_filename} in {Path(self.path).resolve()}")
        
        self.model_path = str(urdf_path.resolve().parent)
        builder = RobotWrapper.BuildFromURDF
        self.robot = builder(
                urdf_path,
                self.model_path,
                pin.JointModelFreeFlyer() if self.free_flyer else None,
            )

        self.srdf_path = None
        self.robot.q0 = pin.neutral(self.robot.model)
        self.robot.urdf = urdf_path
        # left_foot_id = self.robot.model.getFrameId('left_ankle_roll_link')
        # right_foot =self.robot.model.getFrameId('right_ankle_roll_link')
        # print(self.robot.data.oMf[left_foot_id],
        # self.robot.data.oMf[right_foot]
        # )
        cfg = pink.Configuration(self.robot.model, self.robot.data,
                        self.robot.q0)

        pelvis_from_rf = cfg.get_transform_frame_to_world(
            'right_ankle_roll_link')
        self.robot.q0[2] = 0.028531 + -pelvis_from_rf.translation[-1]
        if rpy is not None:
            self.robot.q0[1] = -0.15
            R = pin.Quaternion(pin.utils.rpyToMatrix(0.0, 0.0, -np.pi / 2))
            self.robot.q0[3:7] = np.asarray([R.x,R.y,R.z,R.w])

        if self.free_flyer:
            self.addFreeFlyerJointLimits()

    def addFreeFlyerJointLimits(self):
        ub = self.robot.model.upperPositionLimit
        ub[:7] = 1
        self.robot.model.upperPositionLimit = ub
        lb = self.robot.model.lowerPositionLimit
        lb[:7] = -1
        self.robot.model.lowerPositionLimit = lb

link_names = {
        "left_hip_pitch_joint": 1,
        "left_hip_roll_joint": 1,
        "left_hip_yaw_joint": 1,
        "left_knee_joint": 4,
        "left_ankle_pitch_joint": 7,
        "left_ankle_roll_joint": 7,
        "right_hip_pitch_joint": 2,
        "right_hip_roll_joint": 2,
        "right_hip_yaw_joint": 2,
        "right_knee_joint": 5,
        "right_ankle_pitch_joint": 8,
        "right_ankle_roll_joint": 8,
        "waist_yaw_joint": 0,
        "waist_roll_joint": 0,
        "waist_pitch_joint": 0,
        "left_shoulder_pitch_joint": 16,
        "left_shoulder_roll_joint": 16,
        "left_shoulder_yaw_joint": 16,
        "left_elbow_joint": 18,
        "right_shoulder_pitch_joint": 17,
        "right_shoulder_roll_joint": 17,
        "right_shoulder_yaw_joint": 17,
        "right_elbow_joint": 19,
    }
smpl_joint_names = [
    "PELVIS", 
    "L_HIP", 
    "R_HIP", 
    "SPINE1", 
    "L_KNEE", 
    "R_KNEE", 
    "SPINE2", 
    "L_ANKLE", 
    "R_ANKLE", 
    "SPINE3", 
    "L_FOOT", 
    "R_FOOT", 
    "NECK", 
    "L_COLLAR", 
    "R_COLLAR", 
    "HEAD", 
    "L_SHOULDER", 
    "R_SHOULDER", 
    "L_ELBOW", 
    "R_ELBOW", 
    "L_WRIST", 
    "R_WRIST", 
    "L_HAND", 
    "R_HAND", 
]

class dataloader:
    path: Optional[str] = None
    def __init__(self):
        if self.path is None:
            self.path = Path(__file__).parents[2] / 'data/traj/ddp_cr7_data.pkl'
        self.data = joblib.load(self.path)
        print(self.data.keys())

    def get_data(self):
        smpl_joints = self.data['smpl_joints']
        # for link_name, smpl_idx in link_names.items():
        #     print(f"{link_name}: {smpl_joint_names[smpl_idx]}")
        ret = {}
        proceed = []
        for link_name, link_id in link_names.items():
            if link_id in proceed:
                continue
            else:
                ret[link_name]=smpl_joints[:, link_id, :]
                proceed.append(link_id)
                # if 'ankle' in link_name:
                #     ret[link_name][62:, 2] = 0.038531
        return ret
    
    def parse_contact(self, left_foot,
                      right_foot):
        return {
            left_foot: 1-self.data['contact'][..., 0],
            right_foot: 1-self.data['contact'][..., 1]
        }
        

def get_data(data_path: str):
    
    with open(data_path, "rb") as f:
        data = joblib.load(f)
    smpl_joints = data['smpl_joints']
    print("matched keys")
    for link_name, smpl_idx in link_names.items():
        print(f"{link_name}: {smpl_joint_names[smpl_idx]}")
    ret = {link_name: smpl_joints[:, link_id, :] for link_name, link_id in link_names.items()}
    return ret
        

if __name__ == "__main__":
    data_path = "/root/ddp-cr7/data/traj/ddp_cr7_data.pkl"
    data = get_data(data_path)
    print(data.keys())