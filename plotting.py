import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.ndimage import gaussian_filter
import porespy as ps

#%%
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
#%% init plotting
n_r = 30 
n_c = 30 
max_c = 150
D_max = 0.975
max_lt = 20
thresh_r = np.linspace(0,D_max, n_r+1)
thresh_c = np.linspace(0,max_c,n_c+1)
stats_list = np.load('F:/WORK/Aerogel/stats_40_save3.npy',allow_pickle=True)
subgroups = ['aF','bN','cF','dN']
#colgroups = plt.rcParams['axes.prop_cycle'].by_key()['color'] #not quite as nice colors
colgroups = ['#f58231','#3cb44b','#000075','#e6194B'] #nice colors
#%% Plot radius vs porosity
idx = range(4)
m = [0 for _ in range(len(idx))]
for j in idx:
    for i in range(10):
        plt.plot(thresh_r[1:],np.array(stats_list[j*10+i]["rad"]["porosity"]),color=colgroups[j],alpha=0.2)
        m[j] += np.array(stats_list[j*10+i]["rad"]["porosity"])/10
for j in idx: 
    plt.plot(thresh_r[1:],m[j],color=colgroups[j],label=subgroups[j], linewidth=2.5)
plt.xlabel('Radius')
plt.ylabel('Porosity')
plt.legend()
#%% Plot mean porosity vs sample number
mean_porosity = np.zeros((10,4))
for j in range(4):
    for i in range(10):
        mean_porosity[i,j] = np.mean(stats_list[j*10+i]["rad"]["porosity"])
for j in range(4):
    plt.plot(mean_porosity[:,j],'-o',color=colgroups[j])
plt.ylabel('Porosity')
plt.xlabel('Sample number')
plt.legend(subgroups)
#%%
m = [0,0]
for j in [0,1]:
    for i in range(10):
        plt.plot(thresh_r[1:],stats_list[j*10+i]["rad"]["carbon_density"],color=colgroups[j],alpha=0.2)
        m[j] += np.array(stats_list[j*10+i]["rad"]["carbon_density"])/10
for j in [0,1]: 
    plt.plot(thresh_r[1:],m[j],color=colgroups[j],label=subgroups[j], linewidth=2.5)
plt.xlabel('Radius')
plt.ylabel('Carbon fiber density')
plt.legend()
#%% plot mean local thickness vs radius
idx = range(4)
m = [0 for _ in range(len(idx))]
for j in idx:
    for i in range(10):
        plt.plot(thresh_r[1:],stats_list[j*10+i]["rad"]["LT_mean"],color=colgroups[j],alpha=0.2)
        m[j] += np.array(stats_list[j*10+i]["rad"]["LT_mean"])/10
for j in idx: 
    plt.plot(thresh_r[1:],m[j],color=colgroups[j],label=subgroups[j], linewidth=2.5)
plt.xlabel('Radius')
plt.ylabel('Mean Local thickness (pixels)')
plt.legend()
#%% Plot mean local thickness vs distance to nearest carbon fiber
idx = [0,1]
m = [0 for _ in range(4)]
for j in idx:
    for i in range(10):
        plt.plot(thresh_c[1:],stats_list[j*10+i]["carbon"]["LT_mean"],color=colgroups[j],alpha=0.2)
        m[j] += np.array(stats_list[j*10+i]["carbon"]["LT_mean"])/10
for j in idx: 
    plt.plot(thresh_c[1:],m[j],color=colgroups[j],label=subgroups[j], linewidth=2.5)
    
idx = [2,3]
for j in idx:
    for i in range(10):
        if False:
            plt.plot([thresh_c[1],thresh_c[-1]],
                 [np.mean(stats_list[j*10+i]["carbon"]["LT_mean"]) for _ in range(2)],
                 color=colgroups[j],
                 alpha=0.2)
        m[j] += np.array(stats_list[j*10+i]["carbon"]["LT_mean"]).mean()/10
for j in idx: 
    plt.plot([thresh_c[1],thresh_c[-1]],[m[j] for _ in range(2)],
             color=colgroups[j],
             label=subgroups[j], 
             linewidth=2.5)
plt.xlabel('Distance to carbon fibers (pixels)')
plt.ylabel('Mean local thickness (Pixels)')
plt.legend()
#%% Plot porosity vs distance to nearest carbon fiber
idx = [0,1]
m = [0 for _ in range(4)]
for j in idx:
    for i in range(10):
        plt.plot(thresh_c[1:],np.array(stats_list[j*10+i]["carbon"]["porosity"]),color=colgroups[j],alpha=0.2)
        m[j] += np.array(stats_list[j*10+i]["carbon"]["porosity"])/10
for j in idx: 
    plt.plot(thresh_c[1:],m[j],color=colgroups[j],label=subgroups[j], linewidth=2.5)
    
idx = [2,3]
for j in idx:
    for i in range(10):
        if False:
            plt.plot([thresh_c[1],thresh_c[-1]],
                 [np.mean(stats_list[j*10+i]["carbon"]["porosity"]) for _ in range(2)],
                 color=colgroups[j],
                 alpha=0.2)
        m[j] += np.array(stats_list[j*10+i]["carbon"]["porosity"]).mean()/10
for j in idx: 
    plt.plot([thresh_c[1],thresh_c[-1]],[m[j] for _ in range(2)],
             color=colgroups[j],
             label=subgroups[j], 
             linewidth=2.5)
plt.xlabel('Distance to carbon fibers (pixels)')
plt.ylabel('Porosity')
plt.legend()
#%% Plot image slices from all images
base_folder = 'F:/WORK/Aerogel/normed_vols3/'
k = 100
pics = []
for i in range(40):
    vol = np.load(base_folder+names[i]+'.npy',allow_pickle=True)
    pic = vol[k].copy()
    #vol.close()
    del vol
    pics.append(pic)
    print(i)
fig,ax = plt.subplots(ncols=10,nrows=4)
Q = [0.01,0.99]
bbox = np.array([0.0,0.5,0.4,0.6])
for j in range(10):
    for i in range(4):
        pic = pics[i*10+j]
        Qs = np.quantile(pic,Q)
        pic[pic<Qs[0]] = Qs[0]
        pic[pic>Qs[1]] = Qs[1]
        bbox = np.array([0.0,0.5,0.4,0.6])
        b = (np.array(pic.shape)[[0,0,1,1]]*bbox).astype(int)
        pic = pic[b[0]:b[1],b[2]:b[3]]
        ax[i,j].imshow(pic,cmap='gray')
        ax[i,j].axis('off')
#%%
plt.tight_layout(pad=0,h_pad=None,w_pad=None,rect=None)
#%% Create array containing nice looking histograms of local thickness distribution
#The 3D array bin_pdf contains the 40 different volumes across the 1st 
#dimension, the different volume radii for the 2nd dimension. The last 
#dimension is the dimension across which we store the local thickness 
#distribution for the specific volume index and radius index. Radii wrt. index
#are given by thresh_r. BIN_PDF is the same where we marginalize wrt. the 
#radii to get pure local thickness distributions for each volume. 
n_bins = 40
bin_edges = np.linspace(1,max_lt,n_bins+1)
bin_cen = bin_edges[1:]*0.5+bin_edges[:-1]*0.5 #centers of local thickness bins wrt. index
bin_pdf = np.zeros((40,n_r,n_bins))
BIN_PDF = np.zeros((40,n_bins))
for vol_idx in range(40):
    dist = stats_list[vol_idx]["rad"]["LT_dist"]
    N = np.array(stats_list[vol_idx]["rad"]["N"]) #Number of pixels in each radius bin
    for i in range(len(dist)):
        if N[i]>0:
            cent_tmp = dist[i].bin_centers[np.nonzero(dist[i].pdf)[0]]
            edge_tmp = np.log(cent_tmp)
            edge_tmp = [0]+np.exp(edge_tmp[1:]*0.5+edge_tmp[:-1]*0.5).tolist()+[max_lt]
            pdf_tmp = dist[i].pdf[np.nonzero(dist[i].pdf)[0]]
            pdf_tmp = pdf_tmp/(pdf_tmp.sum())
            for j in range(len(pdf_tmp)):
                use_idx = np.logical_and(edge_tmp[j]<bin_cen,bin_cen<=edge_tmp[j+1])
                if use_idx.sum()>0:
                    for use_i in np.nonzero(use_idx)[0]:    
                        bin_pdf[vol_idx][i][use_i] += pdf_tmp[j]/use_idx.sum()
                else:
                    use_i = np.abs(bin_cen-cent_tmp[j]).argmin()
                    bin_pdf[vol_idx][i][use_i] += pdf_tmp[j]
    
    BIN_PDF[vol_idx] = (bin_pdf[vol_idx]*(N.reshape(-1,1))).sum(0)/(N.sum())   

#%%
idx = range(4)
m = [0 for _ in range(len(idx))]
for j in idx:
    for i in range(10):
        plt.plot(bin_cen,BIN_PDF[j*10+i],color=colgroups[j],alpha=0.2)
        m[j] += BIN_PDF[j*10+i]/10
for j in idx: 
    plt.plot(bin_cen,m[j],color=colgroups[j],label=subgroups[j], linewidth=2.5)
plt.ylabel('Density')
plt.xlabel('Local thickness (pixels)')
plt.legend()
#%% Create array containing nice looking histograms of local thickness distribution

#DIFFERENCE between this and the other one: this one is based on the distance
#to carbon fiber based characteristics. 
#The 3D array bin_pdf contains the 40 different volumes across the 1st 
#dimension, the different carbon fiber distances for the 2nd dimension. The 
#last dimension is the dimension across which we store the local thickness 
#distribution for the specific volume index and carbon fiber index. Carbon 
#fiber distances wrt. index is given by thresh_c
n_bins = 40
bin_edges = np.linspace(1,max_lt,n_bins+1)
bin_cen = bin_edges[1:]*0.5+bin_edges[:-1]*0.5
bin_pdf = np.zeros((40,n_c,n_bins))
for vol_idx in range(40):
    dist = stats_list[vol_idx]["carbon"]["LT_dist"]
    N = np.array(stats_list[vol_idx]["carbon"]["N"])
    for i in range(len(dist)):
        if N[i]>0:
            cent_tmp = dist[i].bin_centers[np.nonzero(dist[i].pdf)[0]]
            edge_tmp = np.log(cent_tmp)
            edge_tmp = [0]+np.exp(edge_tmp[1:]*0.5+edge_tmp[:-1]*0.5).tolist()+[max_lt]
            pdf_tmp = dist[i].pdf[np.nonzero(dist[i].pdf)[0]]
            pdf_tmp = pdf_tmp/(pdf_tmp.sum()+1e-12)
            for j in range(len(pdf_tmp)):
                use_idx = np.logical_and(edge_tmp[j]<bin_cen,bin_cen<=edge_tmp[j+1])
                if use_idx.sum()>0:
                    for use_i in np.nonzero(use_idx)[0]:    
                        bin_pdf[vol_idx][i][use_i] += pdf_tmp[j]/use_idx.sum()
                else:
                    use_i = np.abs(bin_cen-cent_tmp[j]).argmin()
                    bin_pdf[vol_idx][i][use_i] += pdf_tmp[j]
