from scipy.spatial.transform import Rotation, Slerp
import numpy as np

posArucoTable = {
    "20" : [-0.9, -0.6, 0.0],
    "21" : [0.9, -0.6, 0.0],
    "22" : [-0.9, -1.4, 0.0],
    "23" : [0.9, -1.4, 0.0]    
} # positon du centre des arucos sur la table ordre (repère STK) [m]

def mean_robust(rotations: Rotation, poses, arucosIds, max_var= 0.5):
    M = rotations.mean()
    diff = (M.inv() * rotations).magnitude()
    arucosIds_cleaned = []
    rotations_cleaned = []
    positions_cleaned = []
    for k in range(len(rotations)):
        if diff[k]<max_var:
            rotations_cleaned.append(Rotation.as_matrix(rotations[k]))
            positions_cleaned.append(poses[k])
            arucosIds_cleaned.append(arucosIds[k])
    if(len(rotations_cleaned)!=0):
        newM = Rotation.from_matrix(rotations_cleaned).mean()
        for k in range(len(positions_cleaned)):
            positions_cleaned[k] -= rotations_cleaned[k] @ posArucoTable[str(arucosIds_cleaned[k])]  
        posesMean = sum(positions_cleaned)/len(positions_cleaned)

        return newM, posesMean
    return Rotation.identity(), [0, 0, 0]

def rotation_kabsch(Rcaptured_world_table: Rotation):
    M = Rcaptured_world_table.as_matrix()
    theta = np.arctan2(M[1, 0]- M[0, 1], M[0, 0] + M[1, 1])
    Rflat_world_table = Rotation.from_euler("z", theta)
    return Rflat_world_table

def passe_bas_expo(prev_pos, pos, prev_R, R, ratio):
    posPB = np.array(prev_pos)*(1-ratio) + np.array(pos)*ratio

    Rc = Rotation.concatenate([prev_R, R])
    slerp = Slerp([0,1], Rc)
    RPB = slerp(ratio)
    return posPB, RPB