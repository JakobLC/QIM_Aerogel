import numpy as np
import matplotlib.pyplot as plt

from voltools import (load_tifvol, 
                      inspect_vol,
                      getLargestCC)

import porespy as ps

import time
import glob

from scipy.ndimage import gaussian_filter, zoom, label
from scipy.ndimage.morphology import binary_fill_holes, distance_transform_edt
from scipy.ndimage.measurements import center_of_mass
from scipy.interpolate import interp1d




def remove_ring_artefact2(vol,return_distance_field=True,n_r=30,cutoff=10):
    """
    Normalization algorithm to remove bias from cylindrical aerogel volumes
    Inputs:
        vol = Aerogel intensity volume. Has to be oriented such that the first
              axis is the height of the Aerogel cylinder.
        return_distance_field = Should radial distance array be returned
        n_r = Number of radius based bins to use for bias field estimation
        cutoff = Number of slices to remove from the first axis due to boundary
                 artefacts
    Outputs:
        vol = Normalized volume
        D = (only if return_distance_field=True) Distance field array where the
            center of the cylinder is 0 ane the edge is 1 increasing linearly.
        
    """
    #Quantiles for normalization. Essentially assumes that the 0.025 quantile
    #is 0 in the volume which is important for multiplicative normalization
    #model
    Qs = np.quantile(vol.flatten(),[0.025,0.7])
    
    

    #standard deviation for filters
    g = np.mean(vol.shape)/500
    
    #init some lists for the rings at start and end of volume
    d = [cutoff,vol.shape[0]-1-cutoff,0,0,0,0]
    ring = [0,0]
    mu = [0,0]
    r = [0,0]
    #How large a fraction of the first and last ring should be considered the 
    #ring? larger value for QQ[0] because the first ring seems bigger in most
    #volumes
    QQ = [0.12,0.09]
    for i in range(2):
        #ensure smoothness
        ring[i] = gaussian_filter(vol[d[i]],sigma=g)
        #get outer ring (dark ring in the volumes)
        ring[i] = getLargestCC(ring[i]<np.quantile(ring[i],QQ[i]))
        #get inner part of ring
        ring[i] = binary_fill_holes(ring[i])*(1-ring[i])
        #find center of rings
        mu[i] = center_of_mass(ring[i])
        #find radius of rings based on assumption that the area-radius relation
        #holds like a perfect circle
        r[i] = np.sqrt(ring[i].sum()/np.pi)
    
    #find indices where we crop the volume since the pixels are not part of the
    #aerogel
    d[2] = int(np.floor(np.min((mu[0][0]-r[0],mu[1][0]-r[1]))))
    d[3] = int( np.ceil(np.max((mu[0][0]+r[0],mu[1][0]+r[1]))))
    d[4] = int(np.floor(np.min((mu[0][1]-r[0],mu[1][1]-r[1]))))
    d[5] = int( np.ceil(np.max((mu[0][1]+r[0],mu[1][1]+r[1]))))

    #coordinate array for construction of D
    X2, X1 = np.meshgrid(np.arange(d[4],d[5]),np.arange(d[2],d[3]))

    D = np.zeros((d[1]-d[0],d[3]-d[2],d[5]-d[4]))
    for i in range(d[1]-d[0]):
        t = i/(d[1]-d[0]-1)
        mu_tmp = [(1-t)*mu[0][0]+t*mu[1][0],
                  (1-t)*mu[0][1]+t*mu[1][1]]
        r_tmp = (1-t)*r[0]+t*r[1] #interpolating radius
        D[i] = np.sqrt((X1-mu_tmp[0])**2+(X2-mu_tmp[1])**2)/r_tmp
    
    #threshold radius values for the radius binnings for bias estimation
    thresh_r = np.linspace(0,1,n_r+1)
    #x is the radius (center bin values)
    x = 0.5*thresh_r[1:]+0.5*thresh_r[:-1]
    x = np.array([-1e-12,*list(x),1,D.max()+1e-12]) #add points at start and end of x to ensure all of D is covered
    #y is the 80% quantile intensity, used as estimation for the bias
    y = np.zeros(n_r+3)
    
    vol = vol[d[0]:d[1],d[2]:d[3],d[4]:d[5]]
    vol = (vol-Qs[0])/(Qs[1]-Qs[0])
    vol[vol<0] = 0
    
    for i in range(n_r):
        y[i+1] = np.quantile(vol[np.logical_and(thresh_r[i]<D,D<=thresh_r[i+1])],0.8)
        
    #add some points at the start and end of y to ensure all of D is covered (and for smooth interpolation)
    y[0] = y[1]
    slope = (y[n_r]-y[n_r-1])/(x[n_r]-x[n_r-1])
    y[n_r+1] = y[n_r]+slope*(x[n_r+1]-x[n_r])
    y[n_r+2] = y[n_r]+slope*(x[n_r+2]-x[n_r]) 
    
    #interpolated function
    f = interp1d(x, y, kind='cubic')
    #create bias
    bias = f(D)
    #mask is 1 inside the cylinder and 0 outside, but with smooth interpolation of border pixels
    mask = (1-D)*200
    mask[mask>1] = 1
    mask[mask<0] = 0
    #remove multiplicative bias
    vol = (1-mask)+vol*mask/(np.abs(bias)+1e-12)
    
    if return_distance_field: 
        return vol, D
    else:
        return vol


#Create "names" list and "folders" list where the following folder structure is 
#assumed:
    #F:/WORK/Aerogel/VO43aF/Recon_VO43aF_1
    #F:/WORK/Aerogel/VO43aF/Recon_VO43aF_2
    # ...
    #F:/WORK/Aerogel/VO43aF/Recon_VO43aF_10
#and similarly for the others:
    #F:/WORK/Aerogel/VO43bN/Recon_VO43bN_1
    # ...
    
folder_name = '/WORK/Aerogel/VO'

folders = glob.glob(folder_name+'*/*/')
order = [i+10*j for j in range(4) for i in [0]+list(range(2,10))+[1]]
folders = [folders[i] for i in order]
names = [name[name.find('VO',20):-1] for name in folders]

print(names)
#%% Code to inspect a single volume (the i'th)
i = 5
vol = load_tifvol(folders[i])
#vol = zoom(vol,0.5) #if you want to bin 2x2x2 pixels
inspect_vol(vol)
#%% Code to normalize a single volume and inspect it (the i'th)
i = 5
vol = load_tifvol(folders[i])
vol, D = remove_ring_artefact2(vol)
inspect_vol(vol)
#%% Normalize all 40 volumes and save them as .npy files
for i in range(len(folders)):
    tic = time.perf_counter()
    base_folder = 'F:/WORK/Aerogel/normed_vols3/'
    vol = load_tifvol(folders[i])
    vol, D = remove_ring_artefact2(vol)
    np.save(base_folder+names[i]+'.npy',vol)
    np.save(base_folder+names[i]+'_D.npy',D)
    print(f"timed at {time.perf_counter() - tic:0.4f} seconds. ite="+str(i))
#%% Code for inspecting some of the saved normalizations
base_folder = 'F:/WORK/Aerogel/normed_vols3/'
i = 1
vol = np.load(base_folder+names[i]+'.npy',allow_pickle=True)
D = np.load(base_folder+names[i]+'_D.npy',allow_pickle=True)
#%% inspect vol
inspect_vol(vol)
#%% inspect radius volume (D)
inspect_vol(D)
#%% Code to calculate characteristics for the 40 volumes and saving it in a 
#dictionary called stats_list.


#Number of bins for radius based characteristics
n_r = 30 
#Number of bins for distance to carbon-fibers based statistics
n_c = 30 
#Maximum distance to carbon fibers to calculate characteristics for
max_c = 150
#Maximum local thickness before considering it as empty space. E.g. if the
#distance is greater than 20 pixels to the nearest aerogel cell wall then it
#isn't included in characteristics
max_lt = 20 
#If band_bool[0] is True then Radius based characteristics are calculated only
#for the specific band (or radius), if false then it is cumulative for all
#radii less than or equal to the radius value. band_bool[1] is the same but
#for distance to carbon fibers
band_bool = [True,True]
#Maximum radius to calculate characteristics for. If radius is 1 then some
#parts of the dark ring are included so we set it abit lower
D_max = 0.975
#minimum number of pixels in connected components (cc) to be considered as 
#carbon fibers. We remove tiny thresholded cc
minimum_cc_sum = 10000

base_folder = 'F:/WORK/Aerogel/normed_vols3/'

#init
stats_list = []
thresh_r = np.linspace(0,D_max, n_r+1)
thresh_c = np.linspace(0,max_c,n_c+1)

for i in range(len(names)):
    tic = time.perf_counter()
    stats = {"rad": {"porosity": [], #fraction of aerogel cells
                 "LT_mean": [], #mean of local thickness
                 "LT_std": [], #std of local thickness
                 "LT_dist": [], #distribution of local thickness
                 "carbon_density": [],
                 "N": [] #Number of voxels
                 },
         "carbon": {"porosity": [], #fraction of ones
                 "LT_mean": [], #mean of local thickness
                 "LT_std": [], #std of local thickness
                 "LT_dist": [], #distribution of local thickness
                 "N": [] #Number of voxels
                 }
         }
    
    #load normalized vols
    vol = np.load(base_folder+names[i]+'.npy',allow_pickle=True)
    D = np.load(base_folder+names[i]+'_D.npy',allow_pickle=True)
    
    #Normalize intensities so different volumes are intensity consistent
    volg = gaussian_filter(vol,sigma=np.mean(vol.shape)/100)
    a = np.quantile(vol[np.logical_and(vol<volg,D<0.9)],0.5)
    a2 = np.quantile(vol[np.logical_and(vol<volg-(1-a)/10,D<0.9)],0.5)
    a2 -= 0.05/(1+np.exp((1-a2-0.05)/0.02))
    vol = (vol-1)*(1/(1-a2))+1
        
    del volg #release memory
    
    #Define CA volume as a segmentation of carbon fibers with 1=fiber, 0=not fiber
    #only if (i<20) meaning its in the aF or bN groups that contain fibers
    if i<20: 
        CA = np.logical_and(gaussian_filter(vol,sigma=np.mean(vol.shape)/200)<-0.4,D<D_max).astype(int)
        if minimum_cc_sum>0:
            #remove small connected components from the carbon fiber segmentation
            labelled_mask, num_labels = label(CA)
            _, counts = np.unique(labelled_mask,return_counts=True)
            for j in np.nonzero(counts>=minimum_cc_sum)[0][1:]:
                CA[labelled_mask == j] += 1
            CA = CA == 2
            del labelled_mask
        
    #Define LT volume as a binary segmentation with 0=air, 1=aerogel cell walls
    LT = np.logical_and(gaussian_filter(vol,sigma=np.mean(vol.shape)/600)<0.5,D<D_max)
    
    del vol
    
    if i<20: CA = distance_transform_edt(1-CA)-CA
    
    #Convert LT to local thickness volume
    LT = distance_transform_edt(1-LT)-LT
    LT = ps.filters.local_thickness(LT, mode='dt') 
    
    for j in range(n_r): #loop characteristics over different radii
        if band_bool[0]:
            mask = np.logical_and(np.logical_and(thresh_r[j]<D,D<=thresh_r[j+1]),LT<max_lt)
        else:
            mask = np.logical_and(D<=thresh_r[j+1],LT<max_lt)
        stats["rad"]["porosity"].append((LT<0.5)[mask].mean())
        stats["rad"]["LT_mean"].append(LT[mask].mean())
        stats["rad"]["LT_std"].append(LT[mask].std())
        stats["rad"]["LT_dist"].append(ps.metrics.pore_size_distribution(LT[mask], bins=100, log=False, voxel_size=1))
        if i<20: stats["rad"]["carbon_density"].append((CA<0.5)[mask].mean())
        stats["rad"]["N"].append(mask.sum())
    if i<20:
        for j in range(n_c): #loop characteristics over different distances to carbon fibers
            if band_bool[1]:
                mask = np.logical_and(np.logical_and(np.logical_and(thresh_c[j]<CA,CA<=thresh_c[j+1]),LT<max_lt),D<D_max)
            else:
                mask = np.logical_and(np.logical_and(CA<=thresh_c[j+1],LT<max_lt),D<D_max)
            stats["carbon"]["porosity"].append((LT<0.5)[mask].mean())
            stats["carbon"]["LT_mean"].append(LT[mask].mean())
            stats["carbon"]["LT_std"].append(LT[mask].std())
            stats["carbon"]["LT_dist"].append(ps.metrics.pore_size_distribution(LT[mask], bins=100, log=False, voxel_size=1))
            stats["carbon"]["N"].append(mask.sum())
    else: #if no carbon fibers (not i<20) then calculate a single value for carbon based characteristics
        mask = np.logical_and(LT<max_lt,D<D_max)
        stats["carbon"]["porosity"].append((LT<0.5)[mask].mean())
        stats["carbon"]["LT_mean"].append(LT[mask].mean())
        stats["carbon"]["LT_std"].append(LT[mask].std())
        stats["carbon"]["LT_dist"].append(ps.metrics.pore_size_distribution(LT[mask], bins=100, log=False, voxel_size=1))
        stats["carbon"]["N"].append(mask.sum())
    stats_list.append(stats)
    #save after each iteration incase of crash
    np.save('F:/WORK/Aerogel/stats_40_save3.npy',stats_list) 
    print(f"All {i+1}/{len(names)} completed in {time.perf_counter()-tic:0.4f} seconds.")
