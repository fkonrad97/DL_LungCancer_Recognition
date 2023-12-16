import collections
import numpy as np

IrcTuple = collections.namedtuple('IrcTuple', ['index', 'row', 'col'])
XyzTuple = collections.namedtuple('XyzTuple', ['x', 'y', 'z'])

'''
* Flip the coordinates from IRC to CRI, to align with XYZ.
* Scale the indices with the voxel sizes.
* Matrix-multiply with the directions matrix, using @ in Python.
* Add the offset for the origin.
'''
# @ - Matrix multiplication
def irc2xyz(coord_irc, origin_xyz, vxSize_xyz, direction_a):
    cri_a = np.array(coord_irc)[::-1] # Swap the order while we convert to a NumPy array
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz) 
    coords_xyz = (direction_a @ (cri_a * vxSize_a)) + origin_a # The bottom three steps of the conversion: So basically, with vxSize_a, we scale the newly rotated cri and then multiply it with the direction matrix and finally add the offset: origin_a
    return XyzTuple(*coords_xyz) # https://stackoverflow.com/questions/36901/what-does-double-star-asterisk-and-star-asterisk-do-for-parameters

def xyz2irc(coord_xyz, origin_xyz, vxSize_xyz, direction_a):
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz) 
    coord_a = np.array(coord_xyz)
    cri_a = ((coord_a - origin_a) @ np.linalg.inv(direction_a)) / vxSize_a # Inverse of the last three steps
    cri_a = np.round(cri_a) # Sneaks in proper rounding before converting to integers
    return IrcTuple(int(cri_a[2]), int(cri_a[1]), int(cri_a[0])) # Shuffles and converts to integers
