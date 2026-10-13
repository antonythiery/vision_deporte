from scipy.spatial.transform import Rotation
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
    newM = Rotation.from_matrix(rotations_cleaned).mean()
    for k in range(len(positions_cleaned)):
        positions_cleaned[k] -= rotations_cleaned[k] @ posArucoTable[str(arucosIds_cleaned[k])]  
    posesMean = sum(positions_cleaned)/len(positions_cleaned)

    return newM, posesMean