import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import tifffile
from skimage.morphology import (erosion, dilation, opening, closing, ball)
from skimage.measure import label
from scipy.ndimage.morphology import binary_fill_holes
from pyevtk.hl import gridToVTK

#% INTERACTIVE VISUALIZATION FUNCTIONS - DO NOT WORK WITH INLINE FIGURES
def arrow_navigation(event,z,Z):
    '''
    Change z using arrow keys for interactive inspection.
    @author: vand at dtu dot dk
    '''
    if event.key == "up":
        z = min(z+1,Z-1)
    elif event.key == 'down':
        z = max(z-1,0)
    elif event.key == 'right':
        z = min(z+10,Z-1)
    elif event.key == 'left':
        z = max(z-10,0)
    elif event.key == 'pagedown':
        z = min(z+50,Z+1)
    elif event.key == 'pageup':
        z = max(z-50,0)
    return z


def inspect_vol(V, cmap=plt.cm.gray, vmin = None, vmax = None):
    """
    Inspect volumetric data.
    
    Parameters
    ----------
    V : 3D numpy array, it will be sliced along axis=0.  
    cmap : matplotlib colormap
        The default is plt.cm.gray.
    vmin and vmax: float
        color limits, if None the values are estimated from data.
        
    Interaction
    ----------
    Use arrow keys to change a slice.
    
    @author: vand at dtu dot dk
    """
    def update_drawing():
        ax.images[0].set_array(V[z])
        ax.set_title(f'slice z={z}/{Z}')
        fig.canvas.draw()

    def key_press(event):
        nonlocal z
        z = arrow_navigation(event,z,Z)
        update_drawing()

    Z = V.shape[0]
    z = (Z-1)//2
    fig, ax = plt.subplots()
    if vmin is None:
        vmin = np.min(V)
    if vmax is None:
        vmax = np.max(V)
    ax.imshow(V[z], cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(f'slice z={z}/{Z}')
    fig.canvas.mpl_connect('key_press_event', key_press)
    
    
def inspect_tifvol(filename, cmap=plt.cm.gray, vmin = None, vmax = None):
    ''' 
    Inspect volume saved as tif stack or collection of tifs.

    Parameters
    ----------
    filename : str
        A name of a stacked tif file or a name of a folder containing a
        collection of tif files.
    cmap : matplotlib colormap
        The default is plt.cm.gray.
    vmin and vmax: float
        color limits, if None the values are estimated from the middle slice.
        
    Interaction
    ----------
    Use arrow keys to change a slice.
 
    Author: vand@dtu.dk, 2021
    '''

    def update_drawing():
        I = readslice(z)
        ax.images[0].set_array(I)
        ax.set_title(f'slice z={z}/{Z}')
        fig.canvas.draw()

    def key_press(event):
        nonlocal z
        z = arrow_navigation(event,z,Z)
        update_drawing()

    if os.path.isdir(filename):
        D = sorted(glob.glob(filename + '/*.tif*'))
        Z = len(D)
        readslice = lambda z: tifffile.imread(D[z])
    else:
        tif = tifffile.TiffFile(filename)
        Z = len(tif.pages)
        readslice = lambda z: tifffile.imread(filename, key = z)
      
    z = (Z-1)//2
    I = readslice(z)
    fig, ax = plt.subplots()
    if vmin is None:
        vmin = np.min(I)
    if vmax is None:
        vmax = np.max(I)
    ax.imshow(I, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(f'slice z={z}/{Z}')
    fig.canvas.mpl_connect('key_press_event', key_press)
    
    
def load_tifvol(filename, sub=None):
    ''' 
    Load volume from tif stack or collection of tifs.

    Parameters
    ----------
    filename : str
        A name of a stacked tif file or a name of a folder containing a
        collection of tif files.
    sub : a list containing three array-likes with the slices to be loaded from
        each of the three dimendions.
    
    Returns
    -------
    3D numpy array.

    
    Author: vand@dtu.dk, 2021
    '''
  
    if os.path.isdir(filename):
        D = sorted(glob.glob(filename + '/*.tif*'))
        Z = len(D)
        readslice = lambda z: tifffile.imread(D[z])
    else:
        tif = tifffile.TiffFile(filename)
        Z = len(tif.pages)
        readslice = lambda z: tifffile.imread(filename, key = z)
      
    oneimage = readslice(0)
    dim = (Z,) + oneimage.shape 
    
    if sub is None:
        sub = [None, None, None]
    for i in range(3):
        if sub[i] is None:
            sub[i] = np.arange(dim[i])
        sub[i] = np.asarray(sub[i]) # ensure np as we reshape later
    
    V = np.empty((len(sub[0]), len(sub[1]), len(sub[2])), dtype=oneimage.dtype)
    
    for i in range(len(sub[0])):
        I = readslice(sub[0][i])
        V[i] = I[sub[1].reshape((-1,1)), sub[2].reshape((1,-1))]
    
    return V
    

def save_tifvol(V, filename, stacked=True):
    '''
    Saves tifvol using tifffile. 
    Does not (yet) support resolution and xy axis flip.

    Parameters
    ----------
    V : 3D numpy array, it will saved in slices along axis=0.
    filename : str with filename.
    stacked : bool, default is True.
        Whether to save one stacked tif file or a collection of tifs.


    '''
    if stacked:
        tifffile.imwrite(filename, V[0])
        for  z in range(1, V.shape[0]):
            tifffile.imwrite(filename, V[z], append=True)
    else:
        nr_digits = len(str(V.shape[0]))
        nr_format = '{:0' + str(nr_digits) + 'd}'
        for z in range(V.shape[0]):
            tifffile.imwrite(filename + nr_format.format(z) + '.tif', V[z])
 

def getLargestCC(segmentation,connectivity=1):
    """
    Returns the largest connected component from binary np.ndarray
    
    Inputs:
        segmentation = binary volume
        connectivity = maximum number of orthogonal connections that defines
                       pixels as connected. For 2D: 1 is 4 neighbours, 2 is 8
                       neighbours. For 3D: 1 is 6 neighbours and 2 is 26 
                       neighbours
        
    Outputs:
        largestCC    = largest connected component 
    """
    labels = label(segmentation,connectivity=connectivity)
    assert( labels.max() != 0 )
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def ite_ball_morphology(vol,r,ites=3,morph_type='opening'):
    """
    Iteratively uses ball shaped morphological operations on a volume for
    improved speed. 

    Inputs:
        vol        = numpy 3d array of a volume to be morphologically filtered
        r          = radius of the filtering
        ites       = number of iterations to split the radius of filtering in.
                     more iterations are faster but less accurate. Will 
                     determine appropriate value if ites<1
        morph_type = type of morphological operation. Has to be either 
                     'opening', 'closing', 'erosion' or 'dilation'
                     
    Outputs:
        vol        = numpy 3d array of the filtered volume   
        
    Usage example:
        import numpy as np
        vol = np.random.rand(100,100,100)
        r = 10
        vol_filtered = ite_ball_morphology(vol,r)
    """
    assert(any(morph_type==morph_type_i for morph_type_i in ['opening', 'closing', 'erosion', 'dilation']))
    if ites<1:
        ites = round(r/2+0.5)
    ites = round(ites)
    footprint = ball(r/ites)
    for _ in range(ites):
        if morph_type=='opening' or morph_type=='erosion':
            vol = erosion(vol,footprint)
        elif morph_type=='closing' or morph_type=='dilation':
            vol = dilation(vol,footprint)
    for _ in range(ites):
        if morph_type=='opening':
            vol = dilation(vol,footprint)
        elif morph_type=='closing':
            vol = erosion(vol,footprint)
    return vol




def export_to_vtr(data,name='vol_data',
    base_folder='C:/Users/jakob/Desktop/WORK/QIM grundfos/2020 Grundfos Fibre Analysis/'):
    
    x = np.arange(0, data.shape[0]+1)
    y = np.arange(0, data.shape[1]+1)
    z = np.arange(0, data.shape[2]+1)
    
    gridToVTK(base_folder+name, x, y, z, cellData = {name: data.copy()})
    
def cat(arrays,axis=0,new_dim=False):
    """
    Very unsafe concatenation of arrays
    """
    n_dims = np.array([len(np.array(array).shape) for array in arrays]).max()
    if n_dims<axis:
        n_dims = axis
    cat_arrays = []
    for array in arrays:
        if np.size(array)>1:
            tmp = np.array(array).copy()
            tmp = np.expand_dims(tmp,axis=tuple(range(len(tmp.shape),n_dims)))
            cat_arrays.append(tmp)
    if new_dim or len(cat_arrays[0].shape)<=axis:
        for i in range(len(cat_arrays)):
            cat_arrays[i] = np.expand_dims(cat_arrays[i],axis=axis)
    SHAPE = np.array([list(array.shape) for array in cat_arrays]).max(0)
    for i in range(len(cat_arrays)):
        reps = SHAPE//(cat_arrays[i].shape)
        reps[axis] = 1
        cat_arrays[i] = np.tile(cat_arrays[i],reps)
    cat_arrays = np.concatenate(cat_arrays,axis=axis)
    return cat_arrays
    
#%%
# SOME EXAMPLES OF USING VOLTOOLS    
if __name__ == '__main__':     
    #Example with synthhetic data       
    import dummydata
    
    B = dummydata.binary_splatty((30, 150, 160), sigma=7, threshold=0, boundary=0)
    V = dummydata.binary_to_intensities(B)
    inspect_vol(V)
    
    filename = 'dummy.tiff'
    save_tifvol(V, filename)
    inspect_tifvol(filename)
    
    U = load_tifvol(filename, sub=[None,  range(30,121), range(5,156,2)])
    inspect_vol(U)
    
    
    #%% Example of loading a collection of tifs
    
    # Inspect tif volume without loading all images first
    filename = '../testing_data/walnut'
    inspect_tifvol(filename)
    
    # Load subsampled subvolume
    V = load_tifvol(filename, sub=[range(32,398,2),  range(60,371,2), range(50,351,2)])
    inspect_vol(V, vmin=0, vmax=100)
    
    
    #%% Example of loading a stacked tif file
    
    # Inspect tif volume without loading all images first
    filename = '../testing_data/cement2.tif'
    inspect_tifvol(filename)
    
    # Load subsampled subvolume
    V = load_tifvol(filename, sub=[range(100,250), range(100,280), range(100,300)])
    inspect_vol(V)
    
  





    


