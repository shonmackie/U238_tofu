import tofu as tf
import numpy as np
import matplotlib.pyplot as plt
import get_2d_sparc_profile_function as g2sp  #Alex Tinguely's script to get emmission profile
inch    = 2.54e-2; # m
foot    = 12*inch; # m
#load simple SPARC geometry
config_SPARC = tf.load_config('SPARC-V0')
#load emissivity profiles from TRANSP simulation
r_grid, z_grid, emissivity, flux = g2sp.get_2d_sparc_profile(5.973, 'data/10000.CDF', 'data/SPARC_V1E_transp_3.geq')

#print(r_grid[0,:])

emiss2d=emissivity.T
emiss2d*=(5/3.95)/(4*np.pi)#normalize emissivity to correct units and correct total fusion power

#define some parameters
col_r = 0.015 #m, collimator radius
det_r = 0.015 #m, detector sensitive volume radius
d_cf = 0.30 #m, collimator length
sh_R = 3.894 #m major radial position of shield hole
pos_det = np.r_[14.15, 0.409, 0.0875]#Updated 6/18/2024 based on email from X. Wang [14.14428527845,0.40211187714, 0.1875] #detector center position
nin_det = np.r_[-0.999422, -0.0332328, -0.00710971]#[-0.999238741155704, -0.034893416841157, -0.017446708420579] #detector LOS unit vector
nin_det = nin_det/np.linalg.norm(nin_det)
e0 = np.r_[-nin_det[1], nin_det[0], 0]
e0 = e0/np.linalg.norm(e0)
e1 = np.cross(nin_det, e0)
param = np.linspace(0,2*np.pi,100) #parameter for defining geometries

#Tofu uses collection objects for just about everything
U238=tf.data.Collection()
U238.add_mesh_2d_rect(key='m0', domain=None, knots0=r_grid[0,:], knots1=z_grid[:, 0], crop_poly=config_SPARC, units =['m','m'])
U238.add_bsplines(key='m0',deg=1)
U238.add_data(key='emiss2d', data=emiss2d, ref='m0',units='ph/(m3.sr)')

#detector geometry
det = {
    'cents_x': pos_det[0],
    'cents_y': pos_det[1],
    'cents_z': pos_det[2],
    'nin': nin_det,
    'e0': e0,
    'e1': e1,
    'outline_x0': det_r*np.cos(param),
    'outline_x1': det_r*np.sin(param)
}

U238.add_camera_1d(key='det', dgeom=det)
#First aperture geometry dictionary
#FC collimator end
dap1 = {
    'outline_x0': col_r*np.cos(param),
    'outline_x1': col_r*np.sin(param),
    'cent': pos_det+d_cf*nin_det,
    'nin': nin_det,
    'e0': e0,
    'e1': e1
}

U238.add_aperture(key='ap1', **dict(dap1))

#outline of port
#np.r_[-0.29, -0.29, 0.29, 0.29] #x
#np.r_[-0.6, 0.6, 0.6, -0.6] #y

#outline of shield hole
#np.r_[-0.315/2, -0.315/2, 0.315/2, 0.315/2] #x
#np.r_[-0.315/2, 0.315/2, 0.315/2, -0.315/2] #y

#front of port aperture
dap2 = {
    'outline_x0': np.r_[-0.315/2, -0.315/2, 0.315/2, 0.315/2], #sh_r*np.cos(param),
    'outline_x1': np.r_[-0.315/2, 0.315/2, 0.315/2, -0.315/2], #sh_r*np.sin(param),
    'cent': np.r_[3.92, 0, 0],
    'nin': [-1,0,0],
    'e0': [0,-1,0],
    'e1': np.cross([-1,0,0], [0,-1,0])
}

U238.add_aperture(key='ap2', **dict(dap2))

#back end of port
#outline of port
#np.r_[-0.29, -0.29, 0.29, 0.29] #x
#np.r_[-0.6, 0.6, 0.6, -0.6] #y

#back shielding outline
#np.r_[-0.12/2, -0.12/2, -0.315/2, -0.315/2, -0.12/2, -0.12/2, 0.12/2, 0.12/2, 0.315/2, 0.315/2, 0.12/2, 0.12/2] #x
#np.r_[-0.96/2, -0.315/2, -0.315/2, 0.315/2, 0.315/2, 0.96/2, 0.96/2, 0.315/2, 0.315/2, -0.315/2, -0.315/2, -0.96/2] #y
 
dap3 = {
    'outline_x0': np.r_[-0.12/2, -0.12/2, -0.315/2, -0.315/2, -0.12/2, -0.12/2, 0.12/2, 0.12/2, 0.315/2, 0.315/2, 0.12/2, 0.12/2],#np.r_[-0.983/2, -0.983/2, 0.983/2, 0.983/2],# sh_r*np.cos(param),#
    'outline_x1': np.r_[-0.96/2, -0.315/2, -0.315/2, 0.315/2, 0.315/2, 0.96/2, 0.96/2, 0.315/2, 0.315/2, -0.315/2, -0.315/2, -0.96/2],#np.r_[-0.332/2, 0.332/2, 0.332/2, -0.332/2],#sh_r*np.sin(param),#
    'cent': np.r_[2.66, 0, 0],
    'nin': [-1,0,0],
    'e0': [0,-1,0],
    'e1': np.cross([-1,0,0], [0,-1,0])
}

U238.add_aperture(key='ap3', **dict(dap3))

optics ={'det':['ap1', 'ap2', 'ap3']}

U238.add_diagnostic(
    key='U238',
    doptics=optics,
    compute=True,
    config=config_SPARC
)

#dax = U238.plot_as_profile2d(key='emiss2d', dres=0.01, plot_config=config_SPARC, val_out=np.nan, nan0=True)
dax = U238.plot_diagnostic(key='U238', elements='o', plot_config=config_SPARC)
plt.savefig('GeometryTest.PNG')
#plt.show()

dvos, dref = U238.compute_diagnostic_vos(
    # which diag and mesh to use
    key_diag='U238',
    key_cam='det',
    key_mesh='m0',
    # sampling resolution (to be tuned by multiple tests until you find reasonable convergence)
    res_RZ=0.025,
    res_phi=0.025,
    # extra flags
    visibility=False,
    verb=None,
    store=False,
)

# store the dvos
U238.store_diagnostic_vos(
    key_diag='U238',                                                  
    dvos=dvos,                                                          
    dref=dref,                                                          
    spectro=False,                                                    
    overwrite=True,                                                
    replace_poly=True,    
)

U238.compute_diagnostic_signal(
    key='synth',
    key_diag='U238',
    key_integrand='emiss2d',
    brightness=False,
    res=0.025,
    method="vos",
    dvos=dvos,
)

directSignal = U238.get_diagnostic_data(key='U238', data='synth')[0]["det"][0] #n/s
directFlux = directSignal/(np.pi*(det_r*100)**2) #n/cm2/s

print('Direct signal [n/s]: %1.3E' % directSignal)
print('Direct Flux [n/cm2/s]: %1.3E' % directFlux)



