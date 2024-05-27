import os
import json
import nibabel as nb
from os.path import join as opj
from os.path import split as ops
from os import listdir as opl, stat
import glob
import shutil 
import copy, pprint
import pandas as pd
import numpy as np
import nilearn.image as ni
import nilearn.plotting as nplot
import matplotlib.pyplot as plt
import seaborn as sns

### Brain Space ###
from brainspace.gradient import GradientMaps
from brainspace.plotting import plot_hemispheres
from brainspace.mesh.mesh_io import read_surface
from brainspace.utils.parcellation import map_to_labels, map_to_mask
from brainspace.datasets import load_group_fc, load_parcellation, load_conte69
from brainspace.mesh.array_operations import map_pointdata_to_celldata
from brainspace.gradient import GradientMaps

from nilearn.input_data import NiftiLabelsMasker


##### function that makes surfaces
def make_volume_surface(j,converter,atlas, rad = 0.25):
    
    nthread = 35
    gm_pre_all = []
    gm_post_all = []
    gm_pre_all_vol = [] 
    gm_post_all_vol = [] 
    count = 0


    import nilearn.image as ni
    import nilearn.surface as nis

    rh = '/Users/joe/Cook Share Dropbox/Joseph Nashed/Coding_NC/Atlas/Surface/rh.mid_surface.rsl.gii'
    lh = '/Users/joe/Cook Share Dropbox/Joseph Nashed/Coding_NC/Atlas/Surface/lh.mid_surface.rsl.gii'

    wrh = '/Users/joe/Cook Share Dropbox/Joseph Nashed/Coding_NC/Atlas/Surface/rh.white_surface.rsl.gii'
    wlh = '/Users/joe/Cook Share Dropbox/Joseph Nashed/Coding_NC/Atlas/Surface/lh.white_surface.rsl.gii'


    RH = nis.load_surf_mesh(rh)
    LH = nis.load_surf_mesh(lh)
    WRH = nis.load_surf_mesh(wrh)
    WLH = nis.load_surf_mesh(wlh)

    coordinates = np.vstack((LH[0],RH[0]))
    faces = np.vstack((LH[1],RH[1]))
    wcoordinates = np.vstack((WLH[0],WRH[0]))
    wfaces = np.vstack((WLH[1],WRH[1]))

    whole_brain = nis.load_surf_mesh((coordinates,faces))
    white_brain = nis.load_surf_mesh((wcoordinates,wfaces))
    ################################### SURFACE ##################################

    volume1 = ni.index_img(converter.inverse_transform(j.reshape(1,j.size)),0)  
    surface1 = nis.vol_to_surf(volume1 ,whole_brain, interpolation='nearest', kind='ball', radius=rad, inner_mesh=  white_brain,mask_img = ni.binarize_img(atlas,0))#,depth=[0.6])    

    return volume1,surface1


##### Creates surface profections
def get_surface_comp(pre,post,converter1, Atlas_lim_roi,compon=10,rad=1):

    nthread = 35
    gm_pre_all = []
    gm_post_all = []
    gm_pre_all_vol = [] 
    gm_post_all_vol = [] 
    count = 0


    import nilearn.image as ni
    import nilearn.surface as nis

    rh = '/Users/joe/Cook Share Dropbox/Joseph Nashed/Coding_NC/Atlas/Surface/rh.mid_surface.rsl.gii'
    lh = '/Users/joe/Cook Share Dropbox/Joseph Nashed/Coding_NC/Atlas/Surface/lh.mid_surface.rsl.gii'

    wrh = '/Users/joe/Cook Share Dropbox/Joseph Nashed/Coding_NC/Atlas/Surface/rh.white_surface.rsl.gii'
    wlh = '/Users/joe/Cook Share Dropbox/Joseph Nashed/Coding_NC/Atlas/Surface/lh.white_surface.rsl.gii'

    RH = nis.load_surf_mesh(rh)
    LH = nis.load_surf_mesh(lh)
    WRH = nis.load_surf_mesh(wrh)
    WLH = nis.load_surf_mesh(wlh)

    coordinates = np.vstack((LH[0],RH[0]))
    faces = np.vstack((LH[1],RH[1]))
    wcoordinates = np.vstack((WLH[0],WRH[0]))
    wfaces = np.vstack((WLH[1],WRH[1]))

    whole_brain = nis.load_surf_mesh((coordinates,faces))
    white_brain = nis.load_surf_mesh((wcoordinates,wfaces))
    ################################### SURFACE ##################################

    grad_pre = [] 
    grad_post = []

    for i in range(0,compon):
        # i = compon
        g_pre = pre[:,i].T
        g_post = post[:,i].T
        pre_a1 = ni.index_img(converter1.inverse_transform(g_pre.reshape(1,g_pre.size)),0)  
        grad_pre.append(nis.vol_to_surf(pre_a1 ,whole_brain, interpolation='nearest', kind='ball', radius=rad, inner_mesh=  white_brain,mask_img = ni.binarize_img(Atlas_lim_roi,0)))#,depth=[0.6])    
        post_a1 = ni.index_img(converter1.inverse_transform(g_post.reshape(1,g_post.size)),0)
        grad_post.append(nis.vol_to_surf(post_a1 ,whole_brain, interpolation='nearest', kind='ball', radius=rad, inner_mesh=  white_brain,mask_img = ni.binarize_img(Atlas_lim_roi,0)))#,depth=[0.6])    

    gm_pre_all.append(grad_pre)   
    gm_post_all.append(grad_post) 
    gm_pre_all_vol.append(pre_a1)   
    gm_post_all_vol.append(post_a1) 
    return gm_pre_all,gm_pre_all_vol,gm_post_all,gm_post_all_vol,

##### Creates Covariance matrices
def get_cov(i,num):
    print(num)
    from sklearn.covariance import LedoitWolf
    cov = LedoitWolf().fit(i)
    return cov.covariance_

##### Creates eccentricity
def get_dists(x):
    # grads = x.filter(like='g')
    grads = x.filter(items=[1,2,3])
    centroid = grads.mean().values
    x['distance'] = np.linalg.norm(grads.values - centroid, axis=1)
    return x

##### gets the centroid for eccentricity
def get_centroid(x):
    # grads = x.filter(like='g')
    grads = x.filter(items=[1,2,3])
    centroid = grads.mean().values
    x['distance'] = np.linalg.norm(grads.values - centroid, axis=1)
    # x['centroid'] = centroid
    return x,centroid

##### plots 3d eg. for eccentricity
def plot_3d(x, y, z, color=None, ax=None, view_3d=(35, -110), **kwargs):
    """Plot 3D scatter plot of region loadings/weights
    Parameters
    ----------
    x, y, z : array-like
        Data to plot in X, Y, and Z dimensions, respectively
    color : array-like, optional
        Colours to assign each element in data. Must be same length as x, y, 
        and z. By default None
    ax : matplotlib.axes._subplots.Axes3DSubplot, optional
        Existing matplotlib axis. If None, a new Figure and axis will be 
        created. By default None
    view_3d : tuple, optional
        Sets initial view as (elevation, azimuth). By default None
    **kwargs : dict, optional
        Other keyword arguments for `matplotlib.pyplot.scatter`
    Returns
    -------
    matplotlib.axes._subplots.Axes3DSubplot
        3D scatter plot
    """
    if ax is None:
        fig = plt.figure(figsize=(4, 4), frameon=False)
        ax = fig.add_subplot(projection='3d')
    
    ax.scatter(xs=x, ys=y, zs=z, c=color, **kwargs)
    ax.set(xlabel='PC1', ylabel='PC2', zlabel='PC3')
    if view_3d is not None:
        ax.view_init(elev=view_3d[0], azim=view_3d[1])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)
    return ax

##### takes the eccentricity and individual scores and makes a dataframe
def make_dataframe(arr,Time = 'Pre',Group = 'Good'):
    
    ROIs = ['ROI_'+str(i) for i in range(0,arr.T.shape[1])] 
    Pre = [Time for i in range(0,arr.T.shape[0])] 
    grp = [Group for i in range(0,arr.T.shape[0])] 
    subj = [i for i in range(0,arr.T.shape[0])] 
    df = pd.DataFrame((arr.T),columns = ROIs)
    df.insert(0,'Time_Point',Pre)
    df.insert(0,'Group',grp)
    df.insert(0,'Subject',subj)
    return df



# Load Data
path_to_yeo = '/Users/joe/Cook Share Dropbox/Joseph Nashed/Coding_NC/Sample_data/Yeo_2_NMT_CC_atlas.csv'
Yeo_atlas = pd.read_csv(path_to_yeo)

Salvaged_areas = '/Users/joe/Cook Share Dropbox/Joseph Nashed/Coding_NC/Sample_data/Salvaged.csv'
# Salvaged_areas = '/Users/joe/Downloads/Regression_files/Salvaged.csv'
Salvaged_atlas = pd.read_csv(Salvaged_areas)
# Step 1: Identify columns with at least one 1
cols_with_ones = (Salvaged_atlas == 1).any(axis=0)

#Load Sample Data
Directory_data = '/Users/joe/Cook Share Dropbox/Joseph Nashed/Coding_NC/Sample_data/'
pre_files = ['Pre_0','Pre_1','Pre_2','Pre_3']
post_files = ['Post_0','Post_1','Post_2','Post_3']

Pre_data = [pd.read_csv(os.path.join(Directory_data,i+'.csv')) for i in pre_files]
Post_data = [pd.read_csv(os.path.join(Directory_data,i+'.csv')) for i in post_files]

m_pre_new = [get_cov(i.values,num) for num,i in enumerate(Pre_data)]
m_post_new = [get_cov(i.values,num) for num,i in enumerate(Post_data)]

Corr_matrix_pre =np.asarray(m_pre_new)
Corr_matrix_post = np.asarray(m_post_new)

# Array_to_centre - Take Z-transform of the 
overall_mean_pre = np.tanh(np.mean(np.asarray([np.arctanh(i) for i in m_pre_new]),0))

################################## DO GRADIENTS #################################
################################## DO GRADIENTS #################################
threshold = 0.9
rad=1
appr = 'pca'
compon = 10

######### DO GRADIENTS ##########
# Overall alignement gradients
gm_mean = GradientMaps(n_components=compon, approach=appr, random_state=0,kernel= 'cosine')
gm_mean_fit = gm_mean.fit(overall_mean_pre,sparsity=threshold,n_iter=20)

# Pre and post gradients
gm_post = GradientMaps(n_components=compon, approach=appr, random_state=0,kernel= 'cosine', alignment='procrustes')
gm_pre = GradientMaps(n_components=compon, approach=appr, random_state=0,kernel= 'cosine', alignment='procrustes')
gm_post_fit = gm_post.fit(m_post_new,sparsity=threshold,n_iter=20,reference=gm_mean_fit.gradients_)
gm_pre_fit = gm_pre.fit(m_pre_new,sparsity=threshold,n_iter=20,reference=gm_mean_fit.gradients_)
#### aligned #### 
post_gradients = gm_post_fit.aligned_ 
pre_gradients = gm_pre_fit.aligned_

#### Get Variance ####
tot_variance = np.sum(gm_mean_fit.lambdas_)
component_variance = [i/tot_variance for i in gm_mean_fit.lambdas_]

#################### COMPONENT VARIANCE ##################
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(np.asarray([1,2,3,4,5,6,7,8,9,10],dtype=np.int32),(np.asarray(component_variance)*100).astype(np.int32),marker='o',color='k')
ax1.set_xticks([2,4,6,8,10])
plt.show()
#################### COMPONENT VARIANCE ##################


##### Converter ######
Atlas = '/Users/joe/Cook Share Dropbox/Joseph Nashed/Coding_NC/Atlas/NMT_full.nii.gz'
Atlas_sub_stroke = '/Users/joe/Cook Share Dropbox/Joseph Nashed/Coding_NC/Atlas/NMT_Stroke_Atlas_roi.nii.gz'
converter = NiftiLabelsMasker(Atlas)
voxels = converter.fit_transform(Atlas_sub_stroke) 


##### get surface profections for data ####
post_output = [get_surface_comp(j,k,converter, ni.load_img(Atlas_sub_stroke),3,1) for j,k in zip(pre_gradients,post_gradients)]

gm_pre_all,gm_pre_all_vol,gm_post_all,gm_post_all_vol = zip(*post_output)

f_pre = np.mean(np.asarray(gm_pre_all).squeeze(),0)
f_post = np.mean(np.asarray(gm_post_all).squeeze(),0)

compon = 3
stacked_mat = np.vstack((f_pre[0:3],f_post[0:3])).squeeze()
new_order = [0, 3, 1, 4, 2, 5]  
stacked_mat1 = np.asarray([stacked_mat[i,:] for i in new_order])


maxs = np.min([max(i) for i in stacked_mat1])
mins = np.max([min(i) for i in stacked_mat1])

max_cap = -mins
min_cap = mins

if np.abs(maxs) < np.abs(mins): max_cap = maxs; min_cap = mins;

from brainspace.mesh.mesh_io import read_surface
from brainspace.utils.parcellation import map_to_labels, map_to_mask, relabel_consecutive
import nilearn.image as ni
import nilearn.surface as nis

rh = '/Users/joe/Cook Share Dropbox/Joseph Nashed/Coding_NC/Atlas/Surface/rh.mid_surface.rsl.gii'
lh = '/Users/joe/Cook Share Dropbox/Joseph Nashed/Coding_NC/Atlas/Surface/lh.mid_surface.rsl.gii'

wrh = '/Users/joe/Cook Share Dropbox/Joseph Nashed/Coding_NC/Atlas/Surface/rh.white_surface.rsl.gii'
wlh = '/Users/joe/Cook Share Dropbox/Joseph Nashed/Coding_NC/Atlas/Surface/lh.white_surface.rsl.gii'

RH = nis.load_surf_mesh(rh)
LH = nis.load_surf_mesh(lh)
WRH = nis.load_surf_mesh(wrh)
WLH = nis.load_surf_mesh(wlh)

coordinates = np.vstack((LH[0],RH[0]))
faces = np.vstack((LH[1],RH[1]))
wcoordinates = np.vstack((WLH[0],WRH[0]))
wfaces = np.vstack((WLH[1],WRH[1]))

whole_brain = nis.load_surf_mesh((coordinates,faces))
white_brain = nis.load_surf_mesh((wcoordinates,wfaces))

lh_vtk = read_surface(lh)
rh_vtk = read_surface(rh)

label_text = ['Pre-1','Post-1','Pre-2','Post-2','Pre-3','Post-3']
# save_path_fig = '/Users/joe/Cook Share Dropbox/Joseph Nashed/Coding_NC/Figs/'
save_path_fig = '/Users/josephnashed/Documents/Github_repository/Stroke_Gradients/Figs_new/'
plot_hemispheres(lh_vtk, rh_vtk, array_name=stacked_mat1, size=(800, 800),
                cmap='jet',zoom = 1.25,layout_style = 'row',color_bar=True, color_range=(-1.25, 1.25), screenshot=True,background=(1,1,1), label_text=label_text, transparent_bg =False,filename=save_path_fig+'_'+appr+'_pre_post_jet.png')

################################## DO GRADIENTS #################################
################################## DO GRADIENTS #################################


################################## DO ECCENTRICITY #################################
################################## DO ECCENTRICITY #################################
atlas_converted = converter.transform(Atlas_sub_stroke).squeeze()
atlas_converted_bin = np.where(atlas_converted,1,0)

post_gradients_stroke = [(atlas_converted_bin * np.asarray(i).T).T for i in gm_post_fit.aligned_]
pre_gradients_stroke = [(atlas_converted_bin * np.asarray(i).T).T for i in gm_pre_fit.aligned_] 

post_gradients_stroke_rois = [np.asarray(i)[atlas_converted_bin>0,:] for i in gm_post_fit.aligned_]
pre_gradients_stroke_rois = [np.asarray(i)[atlas_converted_bin>0,:] for i in gm_pre_fit.aligned_] 

### Get distances and centroids ###
distance_pre_full = [get_dists(pd.DataFrame(i[:,0:3],columns=[1,2,3])) for i in pre_gradients_stroke_rois]
distance_post_full = [get_dists(pd.DataFrame(i[:,0:3],columns=[1,2,3])) for i in post_gradients_stroke_rois]

distance_pre_dist = np.asarray([get_dists(pd.DataFrame(i[:,0:3],columns=[1,2,3]))['distance'].values for i in pre_gradients_stroke_rois]).squeeze()
distance_post_dist = np.asarray([get_dists(pd.DataFrame(i[:,0:3],columns=[1,2,3]))['distance'].values for i in post_gradients_stroke_rois]).squeeze()

centroid_pre_dist = np.asarray([get_centroid(pd.DataFrame(i[:,0:3],columns=[1,2,3]))[1] for i in pre_gradients_stroke_rois]).squeeze()
centroid_post_dist = np.asarray([get_centroid(pd.DataFrame(i[:,0:3],columns=[1,2,3]))[1] for i in post_gradients_stroke_rois]).squeeze()

############################################## DO TTESTS ##############################################
############################################## ECCENTRICITY ##############################################
from scipy.stats import ttest_rel,ttest_ind

def do_ttest(x,y):
    T,p = zip(*[ttest_rel(i,j) for i,j in zip(x.T,y.T)])
    return T,p

T_pre_post,p_val_pre_post = do_ttest(distance_post_dist,distance_pre_dist)

from statsmodels.stats.multitest import multipletests
p_adjusted = multipletests(p_val_pre_post, method='fdr_bh')
############################################## DO TTESTS ##############################################
############################################## ECCENTRICITY ##############################################


### get mean distances and centrois ####
mean_pre_gm = np.mean(distance_pre_full,0)
mean_post_gm = np.mean(distance_post_full,0)

mean_pre_gm_centroid = np.mean(centroid_pre_dist,0)
mean_post_gm_centroid = np.mean(centroid_post_dist,0)

################################## DO ECCENTRICITY #################################
################################## DO ECCENTRICITY #################################

marker_size = 100
edge_color = 'black'  # Outline color
edge_linewidth = 2   # Outline linewidth
sx_axis=1.25

#### Do Pre Eccentricity ####
lx = np.argmax(mean_pre_gm[:,0])
ly = np.where((mean_pre_gm[:,0]==np.min(mean_pre_gm[:,0])) & (mean_pre_gm[:,3]>0) & (mean_pre_gm[:,2]<0))[0]
lz = np.argmax(mean_pre_gm[:,2])
l1= [lx, ly, lz]
fig = plt.figure()
ax1 = fig.add_subplot(projection='3d')
###### Main Scatter ######
# plot_3d(x=mean_pre_gm[:,0],y=mean_pre_gm[:,1],z=mean_pre_gm[:,2],color = mean_pre_gm[:,3],ax=ax1, alpha = 0.3) # colors based on eccentricity
plot_3d(x=mean_pre_gm[:,0],y=mean_pre_gm[:,1],z=mean_pre_gm[:,2],color = Yeo_atlas.values[0,1::],ax=ax1, alpha = 0.3) # colors based on Yeo regions
plot_3d(x=mean_pre_gm_centroid[0],y=mean_pre_gm_centroid[1],z=mean_pre_gm_centroid[2],color = 'white',ax=ax1, s= marker_size, edgecolor=edge_color, linewidth=edge_linewidth, marker='o', zorder=100)
ax1.plot(mean_pre_gm_centroid[0],mean_pre_gm_centroid[1],mean_pre_gm_centroid[2],color = 'white',  marker='o',markersize=10)
ax1.set_xlim(-sx_axis, sx_axis)
ax1.set_ylim(-sx_axis, sx_axis)
ax1.set_zlim(-sx_axis, sx_axis)
ax1.set_xticks([-sx_axis, 0, sx_axis])
ax1.set_yticks([-sx_axis, 0, sx_axis])
ax1.set_zticks([-sx_axis, 0, sx_axis])
plt.show()
# plt.close('all')

#### Do Post Eccentricity ####
lx = np.argmax(mean_post_gm[:,0])
ly = np.where((mean_post_gm[:,0]==np.min(mean_post_gm[:,0])) & (mean_post_gm[:,3]>0) & (mean_post_gm[:,2]<0))[0]
lz = np.argmax(mean_post_gm[:,2])
l1= [lx, ly, lz]
fig = plt.figure()
ax1 = fig.add_subplot(projection='3d')
###### Main Scatter ######
plot_3d(x=mean_post_gm[:,0],y=mean_post_gm[:,1],z=mean_post_gm[:,2],color = mean_post_gm[:,3],ax=ax1, alpha = 0.3) ### colors based on eccentricity
# plot_3d(x=mean_post_gm[:,0],y=mean_post_gm[:,1],z=mean_post_gm[:,2],color = Yeo_atlas.values[0,1::],ax=ax1, alpha = 0.3) # colors based on Yeo regions
###### Centroid with outline
plot_3d(x=mean_post_gm_centroid[0],y=mean_post_gm_centroid[1],z=mean_post_gm_centroid[2],color = 'white',ax=ax1, s= marker_size, edgecolor=edge_color, linewidth=edge_linewidth, marker='o', zorder=100)
ax1.plot(mean_post_gm_centroid[0],mean_post_gm_centroid[1],mean_post_gm_centroid[2],color = 'white',  marker='o',markersize=10)
ax1.set_xlim(-sx_axis, sx_axis)
ax1.set_ylim(-sx_axis, sx_axis)
ax1.set_zlim(-sx_axis, sx_axis)
ax1.set_xticks([-sx_axis, 0, sx_axis])
ax1.set_yticks([-sx_axis, 0, sx_axis])
ax1.set_zticks([-sx_axis, 0, sx_axis])
plt.show()


################### Plot eccentricity on surface ####################
##### New converter ######
stroke_converter =  NiftiLabelsMasker(Atlas_sub_stroke)
voxels2 = stroke_converter.fit_transform(Atlas_sub_stroke) 
##### New converter ######


### pre surface eccentricity ####
dist_vol_pre = ni.index_img(stroke_converter.inverse_transform(np.mean(distance_pre_dist,0).reshape(1,-1)),0)
dist_surf_pre = nis.vol_to_surf(dist_vol_pre ,whole_brain, interpolation='nearest', kind='ball', radius=rad, inner_mesh=  white_brain,mask_img = ni.binarize_img(Atlas_sub_stroke,0))
### post surface eccentricity ####
dist_vol_post = ni.index_img(stroke_converter.inverse_transform(np.mean(distance_post_dist,0).reshape(1,-1)),0)
dist_surf_post = nis.vol_to_surf(dist_vol_post ,whole_brain, interpolation='nearest', kind='ball', radius=rad, inner_mesh=  white_brain,mask_img = ni.binarize_img(Atlas_sub_stroke,0))
### T test between pre and post projected ####
# dist_vol = ni.index_img(stroke_converter.inverse_transform(np.asarray(np.asarray(T_pre_post)*np.where(np.asarray(p_adjusted[1])<0.05,1,0)).reshape(1,-1)),0) #only significant ROIs
dist_vol = ni.index_img(stroke_converter.inverse_transform(np.asarray(np.asarray(T_pre_post)).reshape(1,-1)),0)
dist_surf = nis.vol_to_surf(dist_vol ,whole_brain, interpolation='nearest', kind='ball', radius=0.25, inner_mesh=  white_brain,mask_img = ni.binarize_img(Atlas_sub_stroke,0))

from surfplot import Plot
p = Plot(surf_rh=rh,surf_lh=lh,layout='grid',zoom=1.25,size=(800,600))
##### Pre Eccentricity ####
# p.add_layer(dist_surf_pre, cmap='viridis',color_range=(0,1.25)) #stroke_surf_percent
##### Post Eccentricity ####
# p.add_layer(dist_surf_post, cmap='viridis',color_range=(0,1.25)) #stroke_surf_percent
##### T-Test between pre and post ####
p.add_layer(dist_surf, cmap='viridis',color_range=(-3,3)) #stroke_surf_percent
fig = p.build(cbar_kws={'location': 'right', 'label_direction': 90})
fig.savefig(save_path_fig+'T_ecc_viridis_'+appr+'.png')




# ###################################################### Behavioural Analysis #############################################
# ###################################################### Behavioural Analysis #############################################
# ###################################################### Behavioural Analysis #############################################
# ###################################################### Behavioural Analysis #############################################
# Behaviour_path = '/Users/joe/Cook Share Dropbox/Joseph Nashed/Coding_NC/Sample_data/NHPSS_test.csv' ###### new behav path
Behaviour_path = '/Users/josephnashed/Documents/Github_repository/Stroke_Gradients/Sample_data/FINAL_NHPSS.csv'
behav = pd.read_csv(Behaviour_path)
subs = behav['Subj'].values
nhpss = behav['NHPSS-Final'].values


import skfda
from skfda.preprocessing.dim_reduction import FPCA
from skfda.exploratory.visualization import FPCAPlot
from skfda.representation.basis import (
    BSplineBasis,
    FourierBasis,
    MonomialBasis,
)
from skfda.misc.regularization import L2Regularization
from skfda.misc.operators import LinearDifferentialOperator

fd = skfda.FDataGrid(
    data_matrix=behav['NHPSS-Final'].values,
    grid_points=behav['NHPSS-Final'].columns.values,
)

fpca_discretized = FPCA(n_components=1)
fpca_discretized.fit(fd)


basis1 = 8
order1 = 4

basis = skfda.representation.basis.BSplineBasis(n_basis=basis1, order=order1)
basis_fd = fd.to_basis(basis)

fpca = FPCA(n_components=1,
            components_basis=skfda.representation.basis.BSplineBasis(n_basis=basis1, order=order1), 
            regularization=L2Regularization(LinearDifferentialOperator(2)))
scores = fpca.fit_transform(basis_fd)

new_fpca = fpca.fit(basis_fd)


####################### FPCAPLOT ########################
####################### FPCAPLOT ########################
####################### FPCAPLOT ########################
####################### FPCAPLOT ########################
fig, ax = plt.subplots(figsize=(10, 5))
for fact in range(20,200,20):
    basis_fd.dataset_name = None #['30','40']
    fpca_plot = FPCAPlot(
        basis_fd.mean(),
        fpca.components_,
        factor=fact,
        axes = ax,
    ).plot()
ax.set_ylim([0,35])
fig.savefig(save_path_fig+'ALL_NHPSS_DATA_FPCA'+'.eps')
fig.savefig(save_path_fig+'ALL_NHPSS_DATA_FPCA'+'.png')
plt.close('all')


#### median split
split_val = np.median(scipy.stats.zscore(scores))
# group positions
good_group = np.where(scipy.stats.zscore(scores)<split_val_fpca)[0]
poor_group = np.where(scipy.stats.zscore(scores)>=split_val_fpca)[0]
# group scores
# good_scores = nhpss[good_group]
# poor_scores = nhpss[poor_group]
good_scores = scores[good_group]
poor_scores = scores[poor_group]

# ###################################################### Behavioural Analysis #############################################
# ###################################################### Behavioural Analysis #############################################
# ###################################################### Behavioural Analysis #############################################
# ###################################################### Behavioural Analysis #############################################



##### Create dataframes for pre/post good/poor animals' eccentricity
pre_poor_df = make_dataframe(distance_pre_dist[poor_group,:].T,'Pre','Poor')
post_poor_df = make_dataframe(distance_post_dist[poor_group,:].T,'Post','Poor')
pre_good_df = make_dataframe(distance_pre_dist[good_group,:].T,'Pre','Good')
post_good_df = make_dataframe(distance_post_dist[good_group,:].T,'Post','Good')

df_final = pd.concat([pre_good_df, post_good_df, pre_poor_df,post_poor_df], ignore_index=True)



import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.formula.api import mixedlm
import fnmatch

pattern = 'ROI_*'
matching_headers = [header for header in df_final.columns if fnmatch.fnmatch(header, pattern)]

##### change in eccentricity ####
diff_poor = distance_post_dist[poor_group,:] - distance_pre_dist[poor_group,:]
diff_good = distance_post_dist[good_group,:] - distance_pre_dist[good_group,:]

df_difference_poor = pd.DataFrame(((diff_poor)),columns = matching_headers)
df_difference_good = pd.DataFrame(((diff_good)),columns = matching_headers)

print(ttest_ind(np.mean(diff_good,1),np.mean(diff_poor,1)))


##### ROI by ROI ###########
ind_ttest =[]
roi_headers = np.asarray(matching_headers) ### Check all rois
# roi_headers = np.asarray(matching_headers)[np.where(np.asarray(p_adjusted[1])<0.05)[0]] ### only check the significant rois from pre to post
for roi in roi_headers:
    # Set up the ttest for post-pre between good group vs. poor group
    ind_ttest.append(ttest_ind(df_difference_good[roi].values,df_difference_poor[roi].values))

# Set up the ttest for post-pre between good group vs. poor group
df_ttest_ind = pd.DataFrame(ind_ttest,columns = ['T_val','P_val'])

print(np.where(df_ttest_ind['P_val'].values<0.05))


from statsmodels.stats import multitest
corrected_p_values_ttest = multitest.multipletests(df_ttest_ind['P_val'].values, method='fdr_bh')[1] ### pick out the significant ROIs after corrections
ind_sig = np.where(corrected_p_values_ttest<0.05)[1] ### find the fdr corrected locations 
new_ind = np.where(np.asarray(p_adjusted[1])<0.05)[0][ind_sig] ### apply locations to the previous significant regions found
# locations_in_matrix = new_ind ## index to use for whole brain T-tests

# rois_salvaged = np.where(Salvaged_atlas.values==1)[new_ind]  #### only look at these salvaged rois that are significant
rois_salvaged = np.where(Salvaged_atlas.values==1)[1]  #### look at all ROIS

####### do ttest along 
total = []
totalP = []
for i in rois_salvaged:
    print(i)
    tot_temp = []
    tot_ptemp = []
    for j in range(m_pre_new[0].shape[-1]):
        good_post_corr = np.arctanh(np.asarray(m_post_new)[good_group,i,j])
        good_pre_corr =np.arctanh(np.asarray(m_pre_new)[good_group,i,j])
        poor_post_corr = np.arctanh(np.asarray(m_post_new)[poor_group,i,j])
        poor_pre_corr =np.arctanh(np.asarray(m_pre_new)[poor_group,i,j])

        good_diff_cr = np.tanh(good_post_corr-good_pre_corr)
        poor_diff_cr = np.tanh(poor_post_corr-poor_pre_corr)

        Ttest_temp = ttest_ind(good_diff_cr,poor_diff_cr)[0]
        Ttest_Ptemp = ttest_ind(good_diff_cr,poor_diff_cr)[1]
        print(Ttest_temp)
        tot_temp.append(Ttest_temp)
        tot_ptemp.append(Ttest_Ptemp)
    total.append(tot_temp)
    totalP.append(tot_ptemp)


tot_new = np.asarray(total)
tot_newp = np.asarray(totalP)

#### find all the connections that are significant
sig_total = [np.where(tot_newp[num]<0.05,i,0) for num,i in enumerate(tot_new)]

atlas_GM = ni.load_img(Atlas)
# volumes_surface_T = [make_volume_surface(np.asarray(i),converter,atlas_GM) for i in total] ###### unthresholded
volumes_surface_T = [make_volume_surface(np.asarray(i),converter,atlas_GM) for i in sig_total] ###### thresholded


volumes_surface_T_roi = []
# for i in np.where(np.asarray(p_adjusted[1])<0.05)[0]: ### significant rois
for i in np.where(np.asarray(Salvaged_atlas==1)==True)[0]: #### all rois
    # roi_zeros = np.zeros(np.asarray(p_adjusted[1]).size)
    roi_zeros = np.where(np.asarray(Salvaged_atlas==1)==True)[0]
    roi_zeros[i] = 1
    volumes_surface_T_roi.append(make_volume_surface(roi_zeros,stroke_converter,ni.load_img(Atlas_sub_stroke)))


########### PLot the connectivity of each region #################
########### PLot the connectivity of each region #################
########### PLot the connectivity of each region #################
########### PLot the connectivity of each region #################
            
for num,i in enumerate(volumes_surface_T_roi):
    from surfplot import Plot
    max_val = np.max(np.nan_to_num(volumes_surface_T[num][1]))
    min_val = np.min(np.nan_to_num(volumes_surface_T[num][1]))
    
    p = Plot(surf_rh=rh,surf_lh=lh,layout='grid',zoom=1.25,size=(800,600))
    if min_val != 0:
        p.add_layer(volumes_surface_T[num][1], cmap='bwr',color_range=(min_val,-min_val), zero_transparent=True) #stroke_surf_percent
    else:
        p.add_layer(volumes_surface_T[num][1], cmap='bwr',color_range=(-max_val,max_val), zero_transparent=True) #stroke_surf_percent

    p.add_layer(volumes_surface_T_roi[num][1]*10, cmap='viridis',color_range=(0,1)) #stroke_surf_percent
    
    fig = p.build(cbar_kws={'location': 'right', 'label_direction': 90})
    fig.savefig(save_path_fig+'/All_rois/Sym_new_T_value_ROI_'+str(num)+'.png')

    plt.close('all')