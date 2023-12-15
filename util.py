import collections
import numpy as np

IrcTuple = collections.namedTuple('IrcTuple', ['index', 'row', 'col'])
XyzTuple = collections.namedTuple('XyzTuple', ['x', 'y', 'z'])

# @ - Matrix multiplication
def irc2xyz(coord_irc, origin_xyz, vxSize_xyz, direction_a):
    cri_a = np.array(coord_irc)[::-1] # Swap the order while we convert to a NumPy array
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz) 
    coords_xyz = (direction_a @ (cri_a * vxSize_a)) + origin_a # The bottom three steps of the conversion
    return XyzTuple(*coords_xyz) # https://stackoverflow.com/questions/36901/what-does-double-star-asterisk-and-star-asterisk-do-for-parameters

def xyz2irc(coord_xyz, origin_xyz, vxSize_xyz, direction_a):
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz) 
    coord_a = np.array(coord_xyz)
    cri_a = ((coord_a - origin_a) @ np.linalg.inv(direction_a)) / vxSize_a # Inverse of the last three steps
    cri_a = np.round(cri_a) # Sneaks in proper rounding before converting to integers
    return IrcTuple(int(cri_a[2]), int(cri_a[1]), int(cri_a[0])) # Shuffles and converts to integers
