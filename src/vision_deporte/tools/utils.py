from scipy.spatial.transform import Rotation

def mean_robust(rotations: Rotation, max_var= 0.5):
    M = rotations.mean()
    diff = (M.inv() * rotations).magnitude()
    rotations_cleaned = []
    for k in range(len(rotations)):
        if diff[k]<max_var:
            rotations_cleaned.append(Rotation.as_matrix(rotations[k]))
    newM = Rotation.from_matrix(rotations_cleaned).mean()
    print(f"newM : {newM}")
    return newM