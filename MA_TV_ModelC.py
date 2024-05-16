# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 10:10:12 2024

@author: Timo
"""



#!/usr/bin/env python
import tensorflow as tf
import numpy as np
tf.keras.backend.set_floatx('float64')
import scipy.io
from pyDOE import lhs
#import h5py
#import smt
#import csv
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from matplotlib import colors
from matplotlib import patches
#import time
import LBFGS_function as LBFGS
import meshio
#import inspect
import pyvista as pv
from datetime import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable

np.bool = np.bool_

#%%

np.random.seed(1)
tf.random.set_seed(1)

SAVE = True


# use hyperthreading / increase speed
# do not use more then 4 threads, 1 is usually enough if executed over night
tf.config.threading.set_intra_op_parallelism_threads(3)
tf.config.threading.set_inter_op_parallelism_threads(3)

#'''
#GPU Settings
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=pow(2,12))]) # max 
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
#'''

#%% PINN class
class PINN:
    #in,out can be N dimensional 
    def __init__(self,data, config ,MAX_dict):
    
        '''
            X: coordinates of training data - [x,y]
            F: training data [ux,p] if not equally sized, under each other
            layers: array with network configuration
            adam_config: adam optimizer configuration [batches, learning rate, epochs] 
            weights: loss weights 
        '''
        self.config = config
        self.layers = config['layer']
        self.adam_config = [config['adam_batch'],config['adam_learningrate'],config['adam_epochs'],config['adam_cyclesize']]
        self.X = data['X_train']  # Input training (unnormiert)
        self.F =  data['F_train']  # Output training (unnormiert)
        self.d = data['d']
        self.MAX_x = MAX_dict['x']
        self.MAX_y = MAX_dict['y']
        
        x_norm = self.X[:,0]/self.MAX_x
        y_norm = self.X[:,1]/self.MAX_y
        
        self.MAX_ux = MAX_dict['ux']
        self.MAX_uy = MAX_dict['uy']
        self.MAX_p = MAX_dict['p']
        self.MAX_nuTilde = MAX_dict['nuTilde']
        self.MAX_umag = np.max(abs(data['umag_bl']))
        
        ux_norm = self.F[:,0]/self.MAX_ux
        uy_norm = self.F[:,1]/self.MAX_uy
        
        self.X_norm = np.concatenate((x_norm[:,np.newaxis],y_norm[:,np.newaxis]), axis=1)
        self.F_norm = np.concatenate((ux_norm[:,np.newaxis], uy_norm[:,np.newaxis]),axis=1)
        
        self.f_colloc_points =  np.concatenate((data['colloc'][:,0][:,np.newaxis]/self.MAX_x,data['colloc'][:,1][:,np.newaxis]/self.MAX_y),axis=1)
        
        self.foil = np.vstack((data['foil'][:,0]/self.MAX_x, data['foil'][:,1]/self.MAX_y)).T
        self.zeros_foil  = np.zeros(np.shape(data['foil'][:,0]))
        
        self.cp_foil_coords = np.vstack((data['cp_foil_coords'][:,0]/self.MAX_x, data['cp_foil_coords'][:,1]/self.MAX_y)).T
        self.cp_foil  = data['cp_foil']

        self.coord_bl = np.vstack((data['BC_bl'][:,0]/self.MAX_x, data['BC_bl'][:,1]/self.MAX_y)).T
        self.umag_bl = data['umag_bl']
        
        self.border =  np.concatenate((data['border'][:,0][:,np.newaxis]/self.MAX_x, data['border'][:,1][:,np.newaxis]/self.MAX_y),axis=1)
        self.nuTilde_border = data['nuTilde_border']/self.MAX_nuTilde

        self.losses = []
        self.log_weights = []

        
   #@tf.function
   def init_NN(self):  
   
       # Define Network architecture
       if self.config['useFF'] == True:
           # Using the Fourier-Feature
           N2 = self.layers[3]
           inp = tf.keras.Input(shape=(2,))
           
           initializer1 = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=self.config['FF_stdev'])

           P_1 = tf.keras.layers.Dense(N2, use_bias=False, trainable=False, kernel_initializer=initializer1)

           x_proj = P_1(2*np.pi*inp)
             
           P_sin = tf.math.sin(x_proj)
           P_cos = tf.math.cos(x_proj)
           IN = tf.concat([P_sin,P_cos],axis=-1,name = 'IN')
           HL  = tf.keras.layers.Dense(N2,activation='linear',name = 'HL_0')(IN)
          
       else:
           # starting layer of normal DNN
           inp = tf.keras.Input(shape=(2,), name= 'IN')
           N2 = self.layers[3]
           HL  = tf.keras.layers.Dense(N2,activation='linear',name = 'HL2_0')(inp)
           
       
       # Hidden layers of the DNN
       HL2 = [HL]
       for ii in range(1,self.layers[2]):
           HL2.append(tf.keras.layers.Dense(N2,activation='tanh', name = 'HL2_'+str(ii))(HL2[ii-1]))

       
       OUT1  = tf.keras.layers.Dense(2,activation='linear',name = 'OUT1')(HL2[-1]) # ux,uy
       OUT2 = tf.keras.layers.Dense(2,activation='linear', name = 'OUT2')(HL2[-1]) # p
       
       # Output
       output_layer = tf.keras.layers.concatenate([OUT1,OUT2],name = 'OUT')
       self.NN  = tf.keras.Model(inputs=inp, outputs=output_layer)
       self.NN.summary()
       #'''
           
           
   def net_f(self,colloc,d): 
       # Calculation of the governing equations residual
       @tf.function
       def net_grad(colloc):
           # Calculate all gradients
           up = self.NN(colloc)
           
           ux    = up[:,0]*self.MAX_ux
           uy    = up[:,1]*self.MAX_uy
           p     = up[:,2]*self.MAX_p
           
           nuTilde = abs(up[:,3])*self.MAX_nuTilde
           
           nu = tf.constant(0.00015, dtype='float64')/(100*0.3)
           c_v1 = tf.constant(7.1, dtype='float64')
           xi = nuTilde/nu
               
           f_v1 = tf.pow(xi,3)/(tf.pow(xi,3)+tf.pow(c_v1,3))
           
           nut = f_v1*nuTilde
           
           # ux gradient
           dux = tf.gradients(ux, colloc)[0]
           ux_x = dux[:,0]/self.MAX_x
           ux_y = dux[:,1]/self.MAX_y
           
           # and second derivative
           ux_xx = tf.gradients(ux_x, colloc)[0][:,0]/self.MAX_x 
           ux_yy = tf.gradients(ux_y, colloc)[0][:,1]/self.MAX_y
           
           # uy gradient
           duy = tf.gradients(uy, colloc)[0]
           uy_x = duy[:,0]/self.MAX_x
           uy_y = duy[:,1]/self.MAX_y
           
           # and second derivative
           uy_xx = tf.gradients(uy_x, colloc)[0][:,0]/self.MAX_x
           uy_yy = tf.gradients(uy_y, colloc)[0][:,1]/self.MAX_y
              
           dp = tf.gradients(p, colloc)[0]
           p_x = dp[:,0]/self.MAX_x
           p_y = dp[:,1]/self.MAX_y
           
           # gradient nut
           dnut = tf.gradients(nut, colloc)[0]
           nut_x = dnut[:,0]/self.MAX_x
           nut_y = dnut[:,1]/self.MAX_y
           
           nuTilde_x = tf.gradients(nuTilde,colloc)[0][:,0]/self.MAX_x  
           nuTilde_y = tf.gradients(nuTilde,colloc)[0][:,1]/self.MAX_y
           
           D = (nuTilde+nu)
           
           Sdiff_x = tf.gradients(D*nuTilde_x, colloc)[0][:,0] /self.MAX_x 
           Sdiff_y = tf.gradients(D*nuTilde_y, colloc)[0][:,1] /self.MAX_y 
           
           return ux,uy,p,nuTilde,nut, ux_x,ux_y,ux_xx,ux_yy, uy_x,uy_y, uy_xx, uy_yy, p_x,p_y,nut_x,nut_y, nuTilde_x, nuTilde_y, Sdiff_x, Sdiff_y
       
       ux,uy,p,nuTilde,nut, ux_x,ux_y,ux_xx,ux_yy, uy_x,uy_y, uy_xx, uy_yy, p_x,p_y,nut_x,nut_y, nuTilde_x, nuTilde_y, Sdiff_x, Sdiff_y = net_grad(colloc)
       
       # SA Constants
       c_v1 = tf.constant(7.1, dtype='float64')
       c_b1 = tf.constant(0.1355, dtype='float64')
       sigma = tf.constant(2/3, dtype='float64')
       kappa = tf.constant(0.41, dtype='float64')
       c_b2 = tf.constant(0.622, dtype='float64')
       c_w1 = c_b1/tf.pow(kappa,2) + ((1+c_b2)/sigma) 
       c_w2 = tf.constant(0.3, dtype='float64')
       c_w3 = tf.constant(2, dtype='float64')
       c_s = tf.constant(0.3, dtype='float64')
       nu = tf.constant(0.00015, dtype='float64')/(100*0.3)
        
       # Spalart-Allmaras turbulence model
       xi = nuTilde/nu
           
       f_v1 = tf.pow(xi,3)/(tf.pow(xi,3)+tf.pow(c_v1,3))
       f_v2 = 1 - xi/(1+xi*f_v1)
        
       Omega = tf.sqrt(0.5*(tf.pow(uy_x-ux_y,2) + tf.pow(ux_y-uy_x,2)))
        
       S_a = Omega + f_v2*nuTilde/(tf.pow(kappa*d,2))
       S_b = c_s * Omega     
                          
       Stilde = tf.reduce_max(tf.stack((S_a,S_b),axis = 1),axis = 1)
       #Stilde = S_a
       
       Stilde_r = tf.reduce_max(tf.stack((Stilde,tf.constant(1e-6,dtype='float64')*tf.ones(tf.shape(Stilde),dtype='float64')),axis = 1),axis = 1)
       
       r_a = nuTilde/(Stilde_r*tf.pow(kappa*d,2))
       r_b = tf.constant(10.,dtype='float64')*tf.ones(tf.shape(r_a),dtype='float64')
        
       r = tf.reduce_min(tf.stack((r_a,r_b), axis = 1),axis = 1)
        
       g = r + c_w2*(tf.pow(r,6)-r)
        
       fw = g * tf.pow((1+tf.pow(c_w3,6))/(tf.pow(g,6)+tf.pow(c_w3,6)),1/6)
        
       Sdiff = 1/sigma*(Sdiff_x + Sdiff_y)
       Sc = c_b2/sigma*(tf.pow(nuTilde_x,2) + tf.pow(nuTilde_y,2))
       Sprod = c_b1*Stilde*nuTilde
       Sdes = c_w1*fw*tf.pow(nuTilde,2)/tf.pow(d,2)
        
       SA = ux * nuTilde_x + uy * nuTilde_y - Sdiff - Sc - Sprod + Sdes
       
       # RANS equations
       f_x = (ux*ux_x + uy*ux_y) + p_x - (nu + nut)*(ux_xx+ux_yy) - 2*nut_x*ux_x - nut_y*(ux_y+uy_x) 
       f_y = (ux*uy_x + uy*uy_y) + p_y - (nu + nut)*(uy_xx+uy_yy) - 2*nut_y*uy_y - nut_x*(ux_y+uy_x) 
       f_mass = ux_x + uy_y
   
       return f_mass, f_x, f_y, SA
   
   @tf.function
   def grad_velo(self,colloc):
       
       # Calculation of velocity derivatives for evaluation
       up = self.NN(colloc)
       
       ux    = up[:,0]*self.MAX_ux
       uy    = up[:,1]*self.MAX_uy
       
       nuTilde = abs(up[:,3])*self.MAX_nuTilde
       
       nu = tf.constant(0.00015, dtype='float64')/(100*0.3)
       c_v1 = tf.constant(7.1, dtype='float64')
       xi = nuTilde/nu
           
       f_v1 = tf.pow(xi,3)/(tf.pow(xi,3)+tf.pow(c_v1,3))
       
       nut = f_v1*nuTilde
       
       # ux gradient
       dux = tf.gradients(ux, colloc)[0]
       ux_x = dux[:,0]/self.MAX_x
       ux_y = dux[:,1]/self.MAX_y
       
       # and second derivative
       ux_xx = tf.gradients(ux_x, colloc)[0][:,0]/self.MAX_x 
       ux_yy = tf.gradients(ux_y, colloc)[0][:,1]/self.MAX_y
       
       # uy gradient
       duy = tf.gradients(uy, colloc)[0]
       uy_x = duy[:,0]/self.MAX_x
       uy_y = duy[:,1]/self.MAX_y
       
       # and second derivative
       uy_xx = tf.gradients(uy_x, colloc)[0][:,0]/self.MAX_x
       uy_yy = tf.gradients(uy_y, colloc)[0][:,1]/self.MAX_y
       
       return ux,uy,ux_x,ux_y,ux_xx,ux_yy,uy_x,uy_y,uy_xx,uy_yy
   
   
   def BC_func(self,colloc_BC,BC_value,var_number):
       # Helper function for boundary condition loss
       up1 = self.NN(colloc_BC)
       if var_number == 3:# nutilde
           f1  = tf.keras.losses.mean_squared_error(abs(up1[:,var_number]), np.squeeze(BC_value))
       if var_number == 2: # pressure coefficient
           f1  = tf.keras.losses.mean_squared_error(up1[:,var_number]/0.5, np.squeeze(BC_value))#/0.5
       else: # pressure
           f1  = tf.keras.losses.mean_squared_error(up1[:,var_number], np.squeeze(BC_value))
       return f1
   
   #@tf.function
   def loss_wrapper(self):
       # loss function
       def total_loss(y_true, y_pred):
               
               LW = self.lossweights
               up_bl = self.NN(self.coord_bl)
               umag_NN = tf.sqrt(tf.pow(up_bl[:,0]*self.MAX_ux,2)+tf.pow(up_bl[:,1]*self.MAX_uy,2))
               
               # Calculate data loss
               data_loss1 = LW[0]*(tf.keras.losses.mean_squared_error(y_true[:,0], y_pred[:,0])) #ux
               data_loss2 = LW[1]*tf.keras.losses.mean_squared_error(y_true[:,1], y_pred[:,1]) #uy
               data_loss_p = LW[2]*(self.BC_func(self.cp_foil_coords,self.cp_foil,2) )#+ tf.keras.losses.mean_squared_error(umag_NN,np.squeeze(self.umag_bl))) # p = p_foil 
               # BL profiles only umag

               
               # calculate boundary condition loss
               BC_foil2 = self.BC_func(self.foil,self.zeros_foil,3)
               BC_foil3 = self.BC_func(self.foil,self.zeros_foil,0)
               BC_foil4 = self.BC_func(self.foil,self.zeros_foil,1)
               BC_border = self.BC_func(self.border,self.nuTilde_border,3) # nuTilde = 0
               
               BC_loss = LW[3]*(BC_border + BC_foil2 + BC_foil3+ BC_foil4) 
               
               # calculate physics loss
               f_mass, f_impx, f_impy, SA  = self.net_f(self.f_colloc_points,self.d)
               
               physical_loss1 = LW[4]*tf.reduce_mean(tf.square(f_mass))
               physical_loss2 = LW[5]*tf.reduce_mean(tf.square(f_impx))
               physical_loss3 = LW[6]*tf.reduce_mean(tf.square(f_impy))
               physical_loss4 = LW[7]*tf.reduce_mean(tf.square(SA)) 
               

               
               total = (data_loss1 + data_loss2 + data_loss_p + BC_loss) + (physical_loss1 + physical_loss2+physical_loss3 +physical_loss4)
               
               # save the loss
               tf.py_function(self.losses.append, inp=[[total, data_loss1,data_loss2,data_loss_p, physical_loss1, physical_loss2, physical_loss3, physical_loss4, BC_loss]], Tout=[])
               tf.py_function(self.log_weights.append, inp=[[self.lossweights]], Tout=[])                  
               return total
       
       # store losses for later use
       total_loss.losses = []
       total_loss.log_weights = []

       return total_loss
   
   #@tf.function
   def train_NN(self):
       # Defining the training routine
       
       nIt = self.config['LBFGS_iter']
       self.lossweights = tf.Variable(config['weights'], dtype = 'float64')
       
       self.loss_func = self.loss_wrapper()
       
       func = LBFGS.function_factory(self.NN, self.loss_func, self.X_norm, self.F_norm)
       # convert initial model parameters to a 1D tf.Tensor
       self.idx = func.idx
       
       print('')    
       print('Starting with Adam optimizing ...')
       print('')
       
       self.NN.compile(optimizer = tf.keras.optimizers.legacy.Adam(learning_rate = self.adam_config[1]), loss = self.loss_func)

       if self.config['useLB']== True:
       # Option to use loss balancing
           for ii in range(1,int(np.ceil(self.adam_config[2]/self.adam_config[3]))):
               
               print('')
               print('Cycle : '+str(ii)+' / '+str(int(np.ceil(self.adam_config[2]/self.adam_config[3])))) 
               print('')
               self.NN.fit(self.X_norm, self.F_norm, batch_size = self.adam_config[0], epochs = self.adam_config[3])
               print('')
               print('Adjusting weights ...') 
               print('')
               self.adaptive_loss()
               print('')
               print('New Weights: '+str(self.lossweights.numpy())) 
               print('')
               #tf.py_function(self.log_weights.append, inp=[self.lossweights], Tout=[])
               
                 
       else:
           self.NN.fit(self.X_norm, self.F_norm, batch_size = self.adam_config[0], epochs = self.adam_config[2])
           #'''   
           
       if self.config['LBFGS_iter'] > 0:
           print('-----------------------------------------------------------------')
           print('Starting with L-BFGS optimizing ...')
           print('')
           
           
           if self.config['useLB']==True:
               cyclesize = self.config['LBFGS_cyclesize']
               for ii in range(1,int(np.ceil(nIt/cyclesize))):
                   print('')
                   print('Cycle : '+str(ii)+' / '+str(int(np.ceil(nIt/cyclesize)))) 
                   print('')
                   init_params = tf.dynamic_stitch(func.idx, self.NN.trainable_variables)
                   results = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=func, initial_position=init_params, max_iterations = int(np.ceil(cyclesize/3)))
                   func.assign_new_model_parameters(results.position)  
                   print('')
                   print('Adjusting weights ...') 
                   print('')
                   self.adaptive_loss()
                   print('')
                   print('New Weights: '+str(self.lossweights.numpy())) 
                   print('')
                   
       
           else:
               Iter = int(np.ceil(nIt/3))
               init_params = tf.dynamic_stitch(func.idx, self.NN.trainable_variables)
               results = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=func, initial_position=init_params, max_iterations = Iter)
               func.assign_new_model_parameters(results.position)  

       self.log_weights = [i.numpy() for i in self.log_weights]
       
   def continue_training(self,nIt):
       # function to continue training of a finished model
       self.lossweights = tf.Variable(config['weights'], dtype = 'float64')
       
       self.loss_func = self.loss_wrapper()
       
       func = LBFGS.function_factory(self.NN, self.loss_func, self.X_norm, self.F_norm)
       self.idx = func.idx
       # convert initial model parameters to a 1D tf.Tensor
       init_params = tf.dynamic_stitch(func.idx, self.NN.trainable_variables)
       # train the model with L-BFGS solver
       results = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=func, initial_position=init_params, max_iterations = int(np.ceil(nIt/3)))
       func.assign_new_model_parameters(results.position)   

#%% Read numerical data
print('')
print('---------- Reading Data ----------')
print('')
clen = 100
u_inf = 0.3

try:
    filename = "/local/tvogel/NACA0012_zigzag_Re2e5_M0d3_AoA3/VTK/aoa3_2d_rans_10000/internal.vtu"
    solutFile = meshio.read(filename)
except:
    filename = r'C:\Users\Timo\tubCloud\Shared\Masterarbeit_Timo_shared\01_data\airfoil\RANS\NACA0012_zigzag_Re2e5_M0d3_AoA3\VTK\aoa3_2d_rans_10000\internal.vtu'
    solutFile = meshio.read(filename)
else:
    filename = r"/local/tvogel/NACA0012_zigzag_Re2e5_M0d3_AoA3/VTK/aoa3_2d_rans_10000/internal.vtu"
    solutFile = meshio.read(filename)

ux_raw=np.asarray(solutFile.point_data['U'],dtype=np.float64)[:,0]/u_inf
uy_raw=np.asarray(solutFile.point_data['U'],dtype=np.float64)[:,1]/u_inf
p_raw=np.asarray(solutFile.point_data['p'],dtype=np.float64)/pow(u_inf,2)
x_raw=np.asarray(solutFile.points,dtype=np.float64)[:,0]/clen
y_raw=np.asarray(solutFile.points,dtype=np.float64)[:,1]/clen
z_raw=np.asarray(solutFile.points,dtype=np.float64)[:,2]/clen
nut_raw=np.asarray(solutFile.point_data['nut'],dtype=np.float64)/(u_inf*clen)
nuTilde_raw=np.asarray(solutFile.point_data['nuTilda'],dtype=np.float64) /(u_inf*clen)
nu = 0.00015/(u_inf*clen)




#%% Read numerical data on the airfoil
try:
    filename ='/local/tvogel/NACA0012_zigzag_Re2e5_M0d3_AoA3/VTK/aoa3_2d_rans_10000/boundary/walls.vtp'
    data = pv.PolyData(filename)
except:
    filename = r'C:\Users\Timo\tubCloud\Shared\Masterarbeit_Timo_shared\01_data\airfoil\RANS\NACA0012_zigzag_Re2e5_M0d3_AoA3\VTK\aoa3_2d_rans_10000\boundary\walls.vtp'
    data = pv.PolyData(filename)
else:
    filename ='/local/tvogel/NACA0012_zigzag_Re2e5_M0d3_AoA3/VTK/aoa3_2d_rans_10000/boundary/walls.vtp'
    data = pv.PolyData(filename)

data = pv.PolyData(filename)

xfoil_raw = np.asarray(data.points[:,0],dtype=np.float64)/clen
yfoil_raw = np.asarray(data.points[:,1],dtype=np.float64)/clen
zfoil_raw = np.asarray(data.points[:,2],dtype=np.float64)/clen
uxfoil_raw = np.asarray(data.point_data['U'],dtype=np.float64)[:,0]/u_inf
uyfoil_raw = np.asarray(data.point_data['U'],dtype=np.float64)[:,1]/u_inf
pfoil_raw = np.asarray(data.point_data['p'],dtype=np.float64)/pow(u_inf,2)
nuTildefoil_raw=np.asarray(data.point_data['nuTilda'],dtype=np.float64) /(u_inf*clen)




#%% Define region of interest (ROI)
print('')
print('---------- Set Region of Interest ----------')
print('')

# big domain including the airfoil
ROI = np.where((x_raw >= 0.32) & (x_raw <= 2.2) & (y_raw >= -0.26) & (y_raw <= 0.18) & (z_raw == 0) & (np.in1d(x_raw,xfoil_raw) == False)& (np.in1d(y_raw,yfoil_raw) == False))[0]

# small domain only in the wake
#ROI = np.where((x_raw >= 1.5) & (x_raw <= 3) & (y_raw >= -0.2) & (y_raw <= 0.1) & (z_raw == 0) & (np.in1d(x_raw,xfoil_raw) == False)& (np.in1d(y_raw,yfoil_raw) == False))

yshift_roi = 0 #np.min(y_raw[ROI]) 
xshift_roi = np.min(x_raw[ROI])

x_roi = x_raw[ROI] -  xshift_roi 
y_roi = y_raw[ROI] -  yshift_roi 
ux_roi = ux_raw[ROI]
uy_roi = uy_raw[ROI]
p_roi = p_raw[ROI]
nut_roi = nut_raw[ROI]

nuTilde_roi = nuTilde_raw[ROI]
#rho = 1

x_foil = xfoil_raw[np.where((zfoil_raw == 0) )] -  xshift_roi 
y_foil = yfoil_raw[np.where((zfoil_raw == 0))] -  yshift_roi 
ux_foil = uxfoil_raw[np.where((zfoil_raw == 0))]
uy_foil = uyfoil_raw[np.where((zfoil_raw == 0))]
p_foil = pfoil_raw[np.where((zfoil_raw == 0))]
nuTilde_foil = nuTildefoil_raw[np.where((zfoil_raw == 0))]

xx_field, xx_foil = np.meshgrid(x_roi, x_foil[x_foil >= 0])
yy_field, yy_foil = np.meshgrid(y_roi, y_foil[x_foil >= 0]) #[x_foil >= 0]

dist = np.sqrt(pow(xx_field-xx_foil,2)+pow(yy_field-yy_foil,2))
d_roi = np.amin(dist, axis = 0)

#%% Read experimental data
try:
    filename ='/local/tvogel/Experiments/Wake_profiles/HW55P63_Wakes_u30_aoaEff3_zigzagTrip.mat'
    exp_wake = scipy.io.loadmat(filename)
    exp_BL = scipy.io.loadmat('/local/tvogel/Experiments/Boundary_layer_profiles/HW55P15_BL_u30_aoaEff3_zigzagTrip')
    exp_cp = scipy.io.loadmat('/local/tvogel/Experiments/Surface_pressure/Cp_ExpLowNoise_NACA0012_u30mds_AoA0eff_zigzag')
    xfoil_aoa3 = scipy.io.loadmat(r'/local/tvogel/Experiments/Surface_pressure/Cp_XFOIL_aoa3_tripxdc0d05.mat')

except:
    filename = r'C:\Users\Timo\tubCloud\Shared\Masterarbeit_Timo_shared\01_data\Experiments\Wake_profiles\HW55P63_Wakes_u30_aoaEff3_zigzagTrip.mat'
    exp_wake = scipy.io.loadmat(filename)
    exp_BL = scipy.io.loadmat(r'C:\Users\Timo\tubCloud\Shared\Masterarbeit_Timo_shared\01_data\Experiments\Boundary_layer_profiles\HW55P15_BL_u30_aoaEff3_zigzagTrip')
    exp_cp = scipy.io.loadmat(r'C:\Users\Timo\tubCloud\Shared\Masterarbeit_Timo_shared\01_data\Experiments\Surface_pressure\Cp_ExpLowNoise_NACA0012_u30mds_AoA0eff_zigzag')
    xfoil_aoa3 = scipy.io.loadmat(r'C:\Users\Timo\tubCloud\Shared\Masterarbeit_Timo_shared\01_data\Experiments\Surface_pressure\Cp_XFOIL_aoa3_tripxdc0d05.mat')

else:
    filename ='/local/tvogel/Experiments/Wake_profiles/HW55P63_Wakes_u30_aoaEff3_zigzagTrip.mat'
    exp_wake = scipy.io.loadmat(filename)
    exp_BL = scipy.io.loadmat('/local/tvogel/Experiments/Boundary_layer_profiles/HW55P15_BL_u30_aoaEff3_zigzagTrip')
    exp_cp = scipy.io.loadmat('/local/tvogel/Experiments/Surface_pressure/Cp_ExpLowNoise_NACA0012_u30mds_AoA0eff_zigzag')
    xfoil_aoa3 = scipy.io.loadmat(r'/local/tvogel/Experiments/Surface_pressure/Cp_XFOIL_aoa3_tripxdc0d05.mat')

 
# shift experimental data to match the numerical coordinates
x_srf = np.squeeze(exp_wake['x_srf']+50)/(100) -  xshift_roi 
y_srf = np.squeeze(exp_wake['y_srf']/(100)) -  yshift_roi


yshift_wake = abs( y_foil[x_foil.argmin()]-y_srf[x_srf.argmin()])-0.00055
xshift_wake = abs( x_foil[x_foil.argmin()]-x_srf[x_srf.argmin()])

x_exp = np.round((exp_wake['x']+50)/(100),2)-  xshift_roi - xshift_wake
y_exp = exp_wake['y']/(100) -  yshift_roi - yshift_wake

x_srf = x_srf - xshift_wake
y_srf = y_srf - yshift_wake

#%% define training points

def find_nearest(array,value): 
    idx = (np.abs(array-value)).argmin()
    idx = np.where((np.abs(array-array[idx])) <= 0.0001)[0]
    return idx 

u_0 = np.max(exp_wake['u'])

ux_exp = exp_wake['u']/u_0
uy_exp = exp_wake['v']/u_0

POS = np.array([1.1,1.5,2]) -  xshift_roi
PROFILE = np.hstack((find_nearest(x_exp,POS[0]),find_nearest(x_exp,POS[1]),find_nearest(x_exp,POS[2])))#,find_nearest(x_exp,POS[3])))

x_train = x_exp[PROFILE] 
y_train = y_exp[PROFILE] 

ux_train = ux_exp[PROFILE] 
uy_train = uy_exp[PROFILE] 

X_train = np.concatenate((x_train,y_train), axis = 1)
F_train = np.concatenate((ux_train,uy_train), axis = 1)


#%% Define collocation point

#'''
x_colloc = np.arange(0,np.max(x_train),0.005)
y_colloc = np.arange(np.min(y_train),np.max(y_train),0.005)

xx_colloc,yy_colloc = np.meshgrid(x_colloc,y_colloc)
xx_colloc = xx_colloc.flatten()
yy_colloc = yy_colloc.flatten()


AoA = -3 * np.pi/180
foil = np.concatenate((x_srf[:,np.newaxis],y_srf[:,np.newaxis]), axis = 1) # angled profil
foil_y_corr = np.sin(-1*AoA)*(x_srf +  xshift_roi) + np.cos(-1*AoA)*(y_srf +  yshift_roi) # helper to devide in upper and lower side
foil_x_corr = np.cos(-1*AoA)*(x_srf +  xshift_roi) - np.sin(-1*AoA)*(y_srf +  yshift_roi) -  xshift_roi # helper to devide in upper and lower side

y_upper = y_srf[np.where(foil_y_corr >= 0)]
y_lower = y_srf[np.where(foil_y_corr <= 0)]
x_upper = x_srf[np.where(foil_y_corr >= 0)]
x_lower = x_srf[np.where(foil_y_corr <= 0)]

y_sep = (y_foil[x_foil == np.max(x_foil)]-y_foil[x_foil == np.min(x_foil)])/(np.max(x_foil)-np.min(x_foil)) *x_foil
y_sep = y_sep + (y_foil[np.argmin(x_foil)]-y_sep[np.argmin(x_foil)])

# calculate the distance to the airfoil surface
xx_field, xx_srf = np.meshgrid(xx_colloc, x_srf[x_srf >= 0] )
yy_field, yy_srf = np.meshgrid(yy_colloc, y_srf[x_srf >= 0] ) #[x_foil >= 0]

dist = np.sqrt(pow(xx_field-xx_srf,2)+pow(yy_field-yy_srf,2))
d = np.amin(dist, axis = 0)


AIRFOIL = []
for i in x_colloc[np.where(x_colloc <= np.max(x_srf))]:
    AIRFOIL = np.append(AIRFOIL,np.asarray(np.where((xx_colloc == i) & (yy_colloc <= y_upper[np.argmin(abs( (x_upper)-i ))]) & (yy_colloc >= y_lower[np.argmin(abs(x_lower-i))]))[0], dtype=np.int64))

AIRFOIL = AIRFOIL.astype(int)
xx_colloc = np.delete(xx_colloc,AIRFOIL)
yy_colloc = np.delete(yy_colloc,AIRFOIL)
d = np.delete(d,AIRFOIL)

ALL = np.where(xx_colloc >= 0)[0]

# section the domain
tresh = 0.06    # treshhold for distance from airfoil -> where the point density reduces

FOIL = np.where(d <= tresh)[0]  # dense point region
FAR = np.where(d > tresh)[0]    # lower density region

colloc_foil =  np.random.choice(FOIL, int(np.ceil(len(FOIL)/1)), replace=False)
#colloc_far =  np.random.choice(FAR, int(np.ceil(len(FAR)/4)),replace=False)
colloc_far =  np.random.choice(FAR, int(np.ceil(len(FAR)/4)),replace=False)

COLLOC = np.hstack((colloc_foil,colloc_far))

#COLLOC =  np.random.choice(ALL, int(np.ceil(len(ALL)/2)),replace=False)

x_colloc = xx_colloc.flatten()[COLLOC]
y_colloc = yy_colloc.flatten()[COLLOC]
d_colloc = d[COLLOC]

colloc = np.vstack((x_colloc,y_colloc)).T

#%% Deine boundary conditions and pressure training data

# BC outside wake
FAR_REST = np.hstack((np.where((yy_colloc <= -0.15 - yshift_roi)&(d > tresh))[0], np.where((d > tresh)&(yy_colloc >= 0.05- yshift_roi))[0]))
BORDER = np.random.choice(FAR_REST, int(np.ceil(len(FAR_REST)/10)),replace=False)
 
x_border = xx_colloc[BORDER]
y_border = yy_colloc[BORDER]
BC_border = np.concatenate((x_border[:,np.newaxis],y_border[:,np.newaxis]), axis=1)
nuTilde_border = np.zeros(np.shape(x_border))

# airfoil
AoA = -3 * np.pi/180

# pressure tab positions
tab_x = np.cos(AoA)*exp_cp['x'] - np.sin(AoA)*exp_cp['y'] -  xshift_roi 
tab_y = np.sin(AoA)*exp_cp['x'] + np.cos(AoA)*exp_cp['y'] -  yshift_roi 
tab_pos = np.vstack((tab_x[tab_x >= -0.05],tab_y[tab_x >= -0.05])).T

# exp airfoil 
foil = np.concatenate((x_srf[x_srf>=0][:,np.newaxis],y_srf[x_srf>=0][:,np.newaxis]), axis = 1) # angled profil
foil_y_corr = np.sin(-1*AoA)*(x_srf +  xshift_roi) + np.cos(-1*AoA)*(y_srf +  yshift_roi) # helper to devide in upper and lower side
foil_x_corr = np.cos(-1*AoA)*(x_srf +  xshift_roi) - np.sin(-1*AoA)*(y_srf +  yshift_roi) -  xshift_roi # helper to devide in upper and lower side
f_foil_up = scipy.interpolate.interp1d( x_srf[np.where((foil_y_corr >= 0)&(x_srf >= -0.05))[0]], y_srf[np.where((foil_y_corr >= 0)&(x_srf >= -0.05))[0]])
f_foil_down = scipy.interpolate.interp1d( x_srf[np.where((foil_y_corr <= 0)&(x_srf >= -0.05))[0]],  y_srf[np.where((foil_y_corr <= 0)&(x_srf >= -0.05))[0]])

# xfoil position
xfoil_y_orig = xfoil_aoa3['y']
xfoil_x = np.cos(AoA)*xfoil_aoa3['x'] - np.sin(AoA)*xfoil_aoa3['y'] -  xshift_roi 
xfoil_y = np.sin(AoA)*xfoil_aoa3['x'] + np.cos(AoA)*xfoil_aoa3['y'] -  yshift_roi 
cp_xfoil = xfoil_aoa3['cp']

RANGE = np.where((xfoil_x >= np.min(foil[:,0]))&(xfoil_x <= np.max(foil[:,0])))[0]
xfoil_x_corr = np.squeeze(xfoil_x[RANGE])
xfoil_y_corr = np.hstack((f_foil_up(xfoil_x_corr[np.squeeze(xfoil_y_orig[RANGE,np.newaxis]) >= 0]),f_foil_down(xfoil_x_corr[np.squeeze(xfoil_y_orig[RANGE,np.newaxis]) < 0])))

# corrected xfoil data coordinates to fit exactly on foil surface
xfoil_foil = np.vstack((xfoil_x_corr,xfoil_y_corr)).T

# get the coordinates of foil that match tab position the closest
PRESSURE = np.argmin(scipy.spatial.distance.cdist(xfoil_foil,tab_pos,'euclidean'),axis=0)



BC_foil = xfoil_foil[PRESSURE,:]
# get xfoil cp for tab positions
cp_foil_BC = cp_xfoil[RANGE][PRESSURE]




#%% Define training data in the boundary layer

#TAKE MEASURMENT POINTS
x_bl = (exp_BL['x']+50)/(100) -  xshift_roi -xshift_wake
y_bl = exp_BL['y']/(100)-  yshift_roi -yshift_wake

N = int(123/3)
x_bl_train = np.concatenate((x_bl[:N,:],x_bl[2*N:,:]), axis = 0)
y_bl_train = np.concatenate((y_bl[:N,:],y_bl[2*N:,:]), axis = 0)

BC_bl = np.concatenate((x_bl_train,y_bl_train), axis = 1)

umag_bl = exp_BL['u_mag']/u_0
umag_bl_train = np.concatenate((umag_bl[:N,:],umag_bl[2*N:,:]), axis = 0)




#%% plotting all used data points
plt.figure()
axs = plt.gca()
#plt.scatter(x_roi,y_roi, s=0.2 , c = 'grey')
#plt.tricontourf(x_roi,y_roi,nuTilde_roi)
plt.scatter(x_srf,y_srf, s=2 ,c = 'gray', label = 'airfoil')
plt.scatter(x_colloc,y_colloc, s=2 ,c = 'orange', label = 'colloc')
plt.scatter(x_train,y_train, s=2 ,c = 'green', label = 'wake')
plt.scatter(BC_foil[:,0],BC_foil[:,1], s=5 ,c = 'red', label = 'pressure')
plt.scatter(foil[:,0],foil[:,1], s=1 ,c = 'blue')
plt.scatter(BC_bl[:,0],BC_bl[:,1], s=5 ,c = 'orange', label = 'bl')
plt.scatter(x_border,y_border, s=2 ,c = 'blue', label = 'nuTilde')
plt.title('boundary conditions')
axs.set_aspect("equal")

# exponential learning rate decay
#plt.plot(np.arange(1,150000), 1e-3 * pow(0.9, (np.arange(1,150000) / 10000)))

#%% NETWORK SETTINGS

data = {
    # training input
    'X_train': X_train,
    # training output
    'F_train': F_train,
    # boundary conditions
    'foil':foil,
    'cp_foil_coords': BC_foil,
    'cp_foil': cp_foil_BC,
    'border': BC_border, 
    'nuTilde_border' : nuTilde_border,
    'BC_bl': BC_bl,
    'umag_bl': umag_bl_train,
    #collocation points
    'colloc': colloc,
    'd': d_colloc}

config = {
    # config NN
    'layer': [2,4,6,128], #[IN,OUT,HL,NODES]
    # config Adam
    'adam_batch': len(x_train),
    'adam_learningrate': 1e-3,
    'adam_epochs': 50000,
    'adam_cyclesize': 1000,
    'adam_decaysteps': 5000,
    #config LBFGS
    'LBFGS_iter': 300000,
    'LBFGS_cyclesize': 1000,
    # starting weights
    'weights': [100,0,1,1,1,1,1,1], #[ux,uy,p_foil,bc, fmass, fx,fy,SA]
    'useFF': False,
    'FF_stdev': 1,
    'useLB': False,
    'note': ''}   
       
MAX_dict = {
    'x': np.max(np.abs(x_colloc)),
    'y': np.max(np.abs(y_colloc)),
    'ux': np.max(np.abs(ux_train)),
    'uy': np.max(np.abs(uy_train)),
    'p': 1,
    'nuTilde': 100*nu}


#%% TRAIN MODEL
print('')
print('---------- Define Model ----------')
print('')

for ii,S in enumerate (['final_exp']):
    
    n_model = 'airfoil_'+S
        
    file = datetime.today().strftime('%Y-%m-%d') +'_model_'+str(n_model)
    save_loc='./saved_weights/' + datetime.today().strftime('%Y-%m-%d')
    print('Current:' + file)
    print('')
                                     
    model = PINN(data, config, MAX_dict)
    
    
    model.init_NN()
    
    
    model.train_NN()  
    
    #model.NN.load_weights('saved_weights/2024-01-08_model_airfoil_basic_08_02.keras')
    #model.continue_training(200000)

    
    #%% Save model and settings
    
    model.NN.save_weights(save_loc +'_model_'+str(n_model)+'.keras')
    
    with open(save_loc + '_config_'+str(n_model)+'.txt', 'w') as f:
        f.write(str(config))

    print('------Saved------') 
    
    #%% Get and save results

    loss = np.array(model.losses)
    weights = np.squeeze(np.array(model.lossweights))
    
 
    X_all =  np.vstack((x_roi/model.MAX_x,y_roi/model.MAX_y)).T
    
    F_pred = model.NN.predict(X_all) # this takes a while for 11 mio points
    x_pred   = X_all[:,0] *model.MAX_x
    y_pred   = X_all[:,1] *model.MAX_y
    
    ux_pred   = F_pred[:,0]*model.MAX_ux
    uy_pred   = F_pred[:,1]*model.MAX_uy
    p_pred   = F_pred[:,2]*model.MAX_p      
    
    nuTilde_pred = abs(F_pred[:,3])*model.MAX_nuTilde
    
    xi = nuTilde_pred/nu
        
    f_v1 = pow(xi,3)/(pow(xi,3)+pow(7.1,3))
    nut_pred = f_v1*nuTilde_pred
    
    save = {'x': x_pred,
            'y': y_pred,
            'ux': ux_pred,
            'uy': uy_pred,
            'p': p_pred,
            'nut': nut_pred,
            'weights': np.squeeze(np.array(model.lossweights, dtype = 'float32')),
            'losses': np.array(model.losses, dtype = 'float32')}
    
    np.save(save_loc + '_data_'+str(n_model)+'.npy', save)
    
    #%% load results from a saved model
    x_val = x_exp.flatten()
    y_val = y_exp.flatten()
    ux_val = ux_exp.flatten()
    uy_val = uy_exp.flatten()
    
    
    #'''
    #2024-04-07_model_airfoil_final_exp_woSA
    #2024-03-11_model_airfoil_final_exp
    
    case = 'C6_woSA' # name for saving
    model.NN.load_weights('saved_weights/final_v2/2024-04-07_model_airfoil_final_exp_woSA.keras')
    data = np.load('saved_weights/final_v2/2024-04-07_data_airfoil_final_exp_woSA.npy', allow_pickle=True).item()
    ux_pred=data['ux']
    uy_pred=data['uy']
    p_pred=data['p']
    nut_pred=data['nut']
    x_pred=data['x']
    y_pred=data['y']
    loss=data['losses']
    weights = data['weights']
    #'''
    
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage[utf8]{inputenc} \usepackage[T1]{fontenc}'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Computer Modern'
    plt.rcParams['font.size'] = '9'
    plt.rcParams['figure.figsize'] = [5.5, 9.2]
    plt.rcParams['axes.linewidth'] = 0.7
    
    
    #loss[ux,uy,p_foil,bc, fmass, fx,fy,SA]
    tl = loss[:,0]
    d1 = loss[:,1]
    d2 = loss[:,2]
    d3 = loss[:,3]
    p1 = loss[:,4]
    p2 = loss[:,5]
    p3 = loss[:,6]
    p4 = loss[:,7]
    bc = loss[:,8]
    
#%%    

    print('') 
    print('------Plotting------') 
    print('') 
    
    PLOT = True
    
    
    if PLOT == True:
        # plot the loss
        patch = patches.Rectangle([0,1e-14], width = 50000, height = (1e2 - 1e-12),color = 'lightgrey')

        fig,axs = plt.subplots(2,1)
        #fig.suptitle('starting weights:' + str(weights))
        t = np.linspace(1,len(loss),num=len(loss))
        axs[0].semilogy(t, d1, 'orange',linewidth=1, label = 'data $u_x$')
        #axs[0].semilogy(t, d2, 'green',linewidth=1 , label = 'data $u_y$')
        axs[0].semilogy(t, bc, 'cyan',linewidth=1, label = 'BC')
        axs[0].semilogy(t, d3, 'gray', linewidth=1, label = 'data $c_p$ \& BL')
        axs[0].semilogy(t, p1, 'red',linewidth=1, label = '$f_{conti}$')
        axs[0].semilogy(t, p2, 'magenta',linewidth=1, label = '$f_x$')
        axs[0].semilogy(t, p3, 'yellow',linewidth=1, label = '$f_y$')
        axs[0].semilogy(t, p4, 'blue',linewidth=1, label = 'SA')
        axs[0].semilogy(t, tl,'black',linewidth=1, label = 'total')
        axs[0].set_ylabel('MSE loss')
        leg = axs[0].legend(loc = 'upper right', ncols=4,facecolor='white', framealpha=1,edgecolor='white')
        [i.set_linewidth(2) for i in leg.legend_handles]

        axs[0].grid(visible=True)
        #axs[0].set_title('MSE loss')
        axs[0].set_ylim([1e-10, 2e1])
        plt.setp(axs[0].get_xticklabels(), visible=False) 
        #axs[0].set_xlabel('Iteration')
        axs[0].add_patch(patch)
        '''
        axs[1].semilogy(t, d1+d2+bc, 'blue',linewidth=1, label = 'data')
        axs[1].semilogy(t, p1+p2+p3+p4, 'red',linewidth=1 , label = 'physics')
        axs[1].semilogy(t, tl, 'black' , linewidth=1, label = 'total')
        axs[1].grid(visible=True)
        axs[1].legend(loc = 'upper right', ncols=3)
        axs[1].set_xlabel('iteration')
        # axs[1].set_ylim([1e-6, 1e1])
        '''
        patch = patches.Rectangle([0,1e-14], width = 50000, height = (1e2 - 1e-12),color = 'lightgrey')

        loss_grad = np.diff(tl)
        axs[1].plot([0,len(loss_grad)],[1e-8,1e-8],'-.', c='grey')
        axs[1].semilogy(t[:-1],loss_grad,'-o',linewidth=0,markersize = 1)
        axs[1].grid(visible=True)
        axs[1].set_ylabel('Change in MSE loss')
        axs[1].set_xlabel('Iteration')
        axs[1].add_patch(patch)
        axs[1].set_ylim([1e-12, 2e1])
        
        plt.savefig('fig/' +file+'_loss.png', bbox_inches="tight")
            
#%% Plot ux profiles   

        def find_nearest(array,value,tresh): 
            idx = (np.abs(array-value)).argmin()
            pos = array[idx]
            idx = np.array(np.where((np.abs(array-array[idx])) <= tresh))
            return idx, pos
        
        position = np.array([0.4,1.2,2]) -  xshift_roi 
        tresh = [0.0009,0.004,0]
        
        plt.rcParams['font.size'] = '18'
        plt.rcParams['figure.figsize'] = [5.5, 0.7*5.5]
        plt.rcParams['axes.linewidth'] = 0.3
        
        #position = np.array([0.4,0.7,0.95]) -  xshift_roi 
        #tresh = [0.0009,0.0009,0.0009]
        #case = 'C5_BL'
        #position = np.array([1,1.05,1.1]) -  xshift_roi 
        #tresh = [0.0002,0.004,0.004]
        
        fig, axs = plt.subplots(1,len(position), sharex=True, sharey=True)
        #fig.suptitle('ux')
        
        for i,pos in enumerate(position):
            idx, posi = find_nearest(x_val,pos,tresh[i])
            idx2, posi2 = find_nearest(x_pred,pos,tresh[i])
            idx3, posi3 = find_nearest(x_roi,pos,tresh[i])
            idx_sep, posi_sep = find_nearest(x_foil,pos,0)
            
            y_plt = y_val[idx]
            ux_plt = ux_val[idx]
            y_plt_pred = y_pred[idx2]
            ux_plt_pred = ux_pred[idx2]
            
            y_plt_roi = y_roi[idx3]
            ux_plt_roi= ux_roi[idx3]
            
            ind_sort = y_plt.argsort()
            ind_sort2 = y_plt_pred.argsort()
            ind_sort3 = y_plt_roi.argsort()
            
            y_plt = y_plt[0,ind_sort]
            ux_plt = ux_plt[0,ind_sort]
            y_plt_pred = y_plt_pred[0,ind_sort2]
            ux_plt_pred = ux_plt_pred[0,ind_sort2]
            y_plt_roi = y_plt_roi[0,ind_sort3]
            ux_plt_roi = ux_plt_roi[0,ind_sort3]
 
            if posi2+  xshift_roi <= 1:
                axs[i].plot(ux_plt_roi[y_plt_roi <= y_sep[idx_sep]].T,y_plt_roi[y_plt_roi <= y_sep[idx_sep]].T +   yshift_roi,'-',c = 'steelblue',markersize=2.5, linewidth = 2.5)
                axs[i].plot(ux_plt_roi[y_plt_roi >= y_sep[idx_sep]].T,y_plt_roi[y_plt_roi >= y_sep[idx_sep]].T +   yshift_roi,linestyle = '-',c = 'steelblue', linewidth =2.5)
    
                axs[i].plot(ux_plt_pred[y_plt_pred >= y_sep[idx_sep]].T,y_plt_pred[y_plt_pred >= y_sep[idx_sep]].T +   yshift_roi,linestyle = '-.',c = 'r', linewidth = 2.5, label = 'PINN')
                axs[i].plot(ux_plt_pred[y_plt_pred <= y_sep[idx_sep]].T,y_plt_pred[y_plt_pred <= y_sep[idx_sep]].T +   yshift_roi,linestyle = '-.',c = 'r', linewidth =2.5)

            else:
                axs[i].plot(ux_plt_roi.T,y_plt_roi.T,'-',c = 'steelblue', linewidth = 2.5)
                axs[i].plot(ux_plt.T,y_plt.T,'k-',markersize=2.5, linewidth = 2.5)
                axs[i].plot(ux_plt_pred.T,y_plt_pred.T,'-.',c = 'r', linewidth = 2.5)
            
            axs[i].set_title(r'x/c  ='+str(round(posi+  xshift_roi,2)))
            
            axs[i].grid(True)
            axs[i].set_ylim([np.min(y_roi)+   yshift_roi,np.max(y_roi)+0.01 +   yshift_roi])
            #axs[i].set_ylim([-0.12,0.02]) # TE
            #axs[i].set_ylim([-0.11,0.08]) # BL
            
            axs[i].set_xlim([np.min(ux_roi)-0.02,np.max(ux_roi)+0.02])
            #axs[i].set_ylabel('y/c')
            #axs[i].set_xlabel('$u_x/u_{\infty}$')
            
            if i > 0:
                plt.setp(axs[i].get_yticklabels(), visible=False) 
        #axs[-1].legend(['RANS','PINN'], loc = 'lower right',ncol = 2,bbox_to_anchor=(1.15,-0.17),frameon=False)
        
        axs[1].set_xlabel('$\overline{u}_x/u_{\infty}$')
        axs[0].set_ylabel('$y/c$')
        plt.savefig('C:/Users/Timo/tubCloud/Shared/Masterarbeit_Timo_shared/04_pictures/'+case+'_ux.pdf', bbox_inches="tight", format='pdf')
    
    
        
 #%%   Plot uy profiles      
        #plt.savefig('fig/'+file+'_ux.png', bbox_inches="tight")
        
        fig, axs = plt.subplots(1,len(position))
        #fig.suptitle('uy')
        for i,pos in enumerate(position):
            idx, posi = find_nearest(x_val,pos,tresh[i])
            idx2, posi2 = find_nearest(x_pred,pos,tresh[i])
            idx3, posi3 = find_nearest(x_roi,pos,tresh[i])
            idx_sep, posi_sep = find_nearest(x_foil,pos,0)
            
            y_plt = y_val[idx]
            uy_plt = uy_val[idx]
            y_plt_pred = y_pred[idx2]
            uy_plt_pred = uy_pred[idx2]
            
            y_plt_roi = y_roi[idx3]
            uy_plt_roi= uy_roi[idx3]
            
            ind_sort = y_plt.argsort()
            ind_sort2 = y_plt_pred.argsort()
            ind_sort3 = y_plt_roi.argsort()
            
            y_plt = y_plt[0,ind_sort]
            uy_plt = uy_plt[0,ind_sort]
            y_plt_pred = y_plt_pred[0,ind_sort2]
            uy_plt_pred = uy_plt_pred[0,ind_sort2]
            y_plt_roi = y_plt_roi[0,ind_sort3]
            uy_plt_roi = uy_plt_roi[0,ind_sort3]
    
            if posi2+  xshift_roi <= 1:
                axs[i].plot(uy_plt_roi[y_plt_roi <= y_sep[idx_sep]].T,y_plt_roi[y_plt_roi <= y_sep[idx_sep]].T +   yshift_roi,'-',c = 'steelblue',markersize=2.5, linewidth = 2.5)
                axs[i].plot(uy_plt_roi[y_plt_roi >= y_sep[idx_sep]].T,y_plt_roi[y_plt_roi >= y_sep[idx_sep]].T +   yshift_roi,linestyle = '-',c = 'steelblue', linewidth =2.5)
    
                axs[i].plot(uy_plt_pred[y_plt_pred >= y_sep[idx_sep]].T,y_plt_pred[y_plt_pred >= y_sep[idx_sep]].T +   yshift_roi,linestyle = '-.',c = 'r', linewidth = 2.5, label = 'PINN')
                axs[i].plot(uy_plt_pred[y_plt_pred <= y_sep[idx_sep]].T,y_plt_pred[y_plt_pred <= y_sep[idx_sep]].T +   yshift_roi,linestyle = '-.',c = 'r', linewidth =2.5)
    
            else:
                axs[i].plot(uy_plt_roi.T,y_plt_roi.T,'-',c = 'steelblue', linewidth = 2.5)
                axs[i].plot(uy_plt.T,y_plt.T,'k-',markersize=2.5, linewidth = 2.5)
                axs[i].plot(uy_plt_pred.T,y_plt_pred.T,'-.',c = 'r', linewidth = 2.5)
            
             
            axs[i].set_title(r'x/c  ='+str(round(posi+  xshift_roi,2)))
            axs[i].grid(True)
            axs[i].set_ylim([np.min(y_roi)+   yshift_roi,np.max(y_roi)+0.01 +   yshift_roi])
            #axs[i].set_ylim([-0.12,0.02]) # TE
            #axs[i].set_ylim([-0.11,0.08]) # BL
            axs[i].set_xlim([np.min(uy_roi)-0.01,np.max(uy_pred)+0.01])
        
            if i > 0:
                plt.setp(axs[i].get_yticklabels(), visible=False) 
        
        #axs[-1].legend(['RANS','PINN'], loc = 'lower right',ncol = 2,bbox_to_anchor=(1.15,-0.17),frameon=False)
        
        #fig.text(0.5, 0.03, '$u_y/u_{\infty}$', va='center', ha='center', fontsize=plt.rcParams['axes.titlesize'])
        #fig.text(0.03, 0.5, 'y/c', va='center', ha='center', rotation='vertical', fontsize=plt.rcParams['axes.titlesize'])
        
        axs[1].set_xlabel('$\overline{u}_y/u_{\infty}$')
        axs[0].set_ylabel('$y/c$')
        plt.savefig('C:/Users/Timo/tubCloud/Shared/Masterarbeit_Timo_shared/04_pictures/'+case+'_uy.pdf', bbox_inches="tight", format='pdf')
    
    #%%  Plot p profiles        
        fig, axs = plt.subplots(1,len(position))
        #fig.suptitle('p')
        for i,pos in enumerate(position):
            idx, posi = find_nearest(x_roi,pos,tresh[i])
            idx2, posi2 = find_nearest(x_pred,pos,tresh[i])
            idx_sep, posi_sep = find_nearest(x_foil,pos,0)
            
            y_plt = y_roi[idx]
            p_plt = p_roi[idx]
            y_plt_pred = y_pred[idx2]
            p_plt_pred = p_pred[idx2]
            
            ind_sort = y_plt.argsort()
            ind_sort2 = y_plt_pred.argsort()
            
            y_plt = y_plt[0,ind_sort]
            p_plt = p_plt[0,ind_sort]
            y_plt_pred = y_plt_pred[0,ind_sort2]
            p_plt_pred = p_plt_pred[0,ind_sort2]
     
            if posi2+  xshift_roi <= 1:
                axs[i].plot(p_plt[y_plt >= y_sep[idx_sep]].T,y_plt[y_plt >= y_sep[idx_sep]].T +   yshift_roi,'-',c= 'steelblue',markersize=3, linewidth = 2.5, label = 'RANS')
                axs[i].plot(p_plt[y_plt <= y_sep[idx_sep]].T,y_plt[y_plt <= y_sep[idx_sep]].T +   yshift_roi,'-',c= 'steelblue',markersize=3, linewidth = 2.5)
     
                axs[i].plot(p_plt_pred[y_plt_pred >= y_sep[idx_sep]].T,y_plt_pred[y_plt_pred >= y_sep[idx_sep]].T +   yshift_roi,'-.',c = 'red', linewidth = 2.5, label = 'PINN')
                axs[i].plot(p_plt_pred[y_plt_pred <= y_sep[idx_sep]].T,y_plt_pred[y_plt_pred <= y_sep[idx_sep]].T +   yshift_roi,'-.',c = 'red', linewidth = 2.5)
     
               
            else:
                axs[i].plot(p_plt.T,y_plt.T,'-',c='steelblue',markersize=3, linewidth = 2.5)
                axs[i].plot(p_plt_pred.T,y_plt_pred.T,'-.',c = 'red', linewidth = 2.5)
            
             
            axs[i].set_title(r'x/c  ='+str(round(posi+  xshift_roi,2)))
            axs[i].grid(True)
            axs[i].set_ylim([np.min(y_roi)+   yshift_roi,np.max(y_roi)+0.01 +   yshift_roi])
            #axs[i].set_ylim([-0.12,0.02]) # TE
            #axs[i].set_ylim([-0.11,0.08]) # BL
            p_diff = abs(np.min(p_roi) - np.max(p_roi))
            p_margin = abs(np.min(p_plt_pred) - np.max(p_plt_pred))
            
            axs[i].set_xlim([np.min(p_plt_pred) - abs(p_diff-p_margin)/2,np.max(p_plt_pred) + abs(p_diff-p_margin)/2])
            #axs[i].set_xlim([np.max(p_plt_pred) - 0.03,np.max(p_plt_pred) + 0.01]) #TE
            if i > 0:
                plt.setp(axs[i].get_yticklabels(), visible=False) 
        
        axs[1].set_xlabel(r'$\dfrac{\Tilde{\overline{p}}-p_{\infty}}{\rho u_{\infty}^2}$')
        axs[0].set_ylabel('$y/c$')
        plt.savefig('C:/Users/Timo/tubCloud/Shared/Masterarbeit_Timo_shared/04_pictures/'+case+'_p.pdf', bbox_inches="tight", format='pdf')
    
#%% Plot nut profiles  

        fig, axs = plt.subplots(1,len(position))
        #fig.suptitle('nut')
        for i,pos in enumerate(position):
            idx, posi = find_nearest(x_roi,pos,tresh[i])
            idx2, posi2 = find_nearest(x_pred,pos,tresh[i])
            idx_sep, posi_sep = find_nearest(x_foil,pos,0)
            
            y_plt = y_roi[idx]
            nut_plt = nut_roi[idx]
            y_plt_pred = y_pred[idx2]
            nut_plt_pred = nut_pred[idx2]
            
            ind_sort = y_plt.argsort()
            ind_sort2 = y_plt_pred.argsort()
            
            y_plt = y_plt[0,ind_sort]
            nut_plt = nut_plt[0,ind_sort]*1e4
            y_plt_pred = y_plt_pred[0,ind_sort2]
            nut_plt_pred = nut_plt_pred[0,ind_sort2]*1e4
    
            if posi+  xshift_roi <= 1:
                axs[i].plot(nut_plt[y_plt >= y_sep[idx_sep]].T,y_plt[y_plt >= y_sep[idx_sep]].T +   yshift_roi,'-',c='steelblue',markersize=2.5, linewidth = 2.5, label = 'RANS')
                axs[i].plot(nut_plt[y_plt <= y_sep[idx_sep]].T,y_plt[y_plt <= y_sep[idx_sep]].T +   yshift_roi,'-',c='steelblue',markersize=2.5, linewidth = 2.5)
    
                axs[i].plot(nut_plt_pred[y_plt_pred >= y_sep[idx_sep]].T,y_plt_pred[y_plt_pred >= y_sep[idx_sep]].T +   yshift_roi,'-.',c = 'r', linewidth = 2.5, label = 'PINN')
                axs[i].plot(nut_plt_pred[y_plt_pred <= y_sep[idx_sep]].T,y_plt_pred[y_plt_pred <= y_sep[idx_sep]].T +   yshift_roi,'-.',c = 'r', linewidth = 2.5)
    
               
            else:
                axs[i].plot(nut_plt.T,y_plt.T,'-',c='steelblue',markersize=2.5, linewidth = 2.5)
                axs[i].plot(nut_plt_pred.T,y_plt_pred.T,'-.',c = 'r', linewidth = 2.5)
            
            axs[i].set_title(r'x/c  ='+str(round(posi+  xshift_roi,2)))
            axs[i].grid(True)
            
            axs[i].set_ylim([np.min(y_roi)+   yshift_roi,np.max(y_roi)+0.01 +   yshift_roi])
            #axs[i].set_ylim([-0.12,0.02]) # TE
            #axs[i].set_ylim([-0.11,0.08]) # BL
            axs[i].set_xlim([np.min(nut_roi*1e4)-0.2,np.max(nut_roi*1e4)+3])
            axs[i].ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
            if i > 0:
                plt.setp(axs[i].get_yticklabels(), visible=False) 
        
        axs[1].set_xlabel(r'$\dfrac{\nu_t}{c u_{\infty}} \times 10^{-4}$')
    
        axs[0].set_ylabel('$y/c$')
        plt.savefig('C:/Users/Timo/tubCloud/Shared/Masterarbeit_Timo_shared/04_pictures/'+case+'_nut.pdf', bbox_inches="tight", format='pdf')

#%%   Plot residual fields  
        plt.rcParams['font.size'] = '9'
        plt.rcParams['figure.figsize'] = [5.5, 9.2]
        
        X_roi = np.concatenate([x_roi[:,np.newaxis]/model.MAX_x,y_roi[:,np.newaxis]/model.MAX_y],axis = 1)

        f_mass, f_impx, f_impy, SA  = model.net_f(X_roi,d_roi)
        fig, axs = plt.subplots(2,2)
        axs = axs.flatten()
        #fig.suptitle('residuals')
        name = ['Momentum x', 'Momentum y','Spalart-Allmaras','Continuity']
        
        for i,f in enumerate([f_impx, f_impy, SA,f_mass]):
            f = np.sqrt(abs(f.numpy()))
            #f[np.where(np.isnan(f))] = 0
            #
            plot = axs[i].tricontourf(X_roi[:,0]*model.MAX_x+   xshift_roi,X_roi[:,1]*model.MAX_y+   yshift_roi,f,100,norm=colors.LogNorm(vmin=f.min(),vmax=f.max()))
            
            plot.set_clim([1e-6,1e1])
            axs[i].set_title(name[i], fontsize = '9')
            #axs[i].set_ylabel('y/c')
            axs[i].set_aspect('equal')
                
            
            axs[i].fill(x_srf+   xshift_roi,y_srf+   yshift_roi,c = 'black')
            
        plt.setp(axs[0].get_xticklabels(), visible=False) 
        plt.setp(axs[1].get_xticklabels(), visible=False)
        plt.setp(axs[1].get_yticklabels(), visible=False) 
        plt.setp(axs[-1].get_yticklabels(), visible=False)
        axs[-1].set_xlabel('x/c')
        axs[-2].set_xlabel('x/c')
        axs[0].set_ylabel('y/c')
        axs[2].set_ylabel('y/c')
            
        fig.subplots_adjust(wspace=0.1, hspace=-0.85) 
        
        cbar = fig.colorbar(plot, ax=axs, location = 'bottom',orientation='horizontal', pad = 0.07, aspect = 30)
        #cbar.ax.locator_params(nbins=5)
        
            
        plt.savefig('C:/Users/Timo/tubCloud/Shared/Masterarbeit_Timo_shared/04_pictures/'+case+'_residuals.png', dpi = 1000, bbox_inches="tight", format='png')
        

        
#%%  Plot flow fields

        plt.rcParams['font.size'] = '9'
        plt.rcParams['figure.figsize'] = [5.5, 9.2]
        plt.rcParams['axes.linewidth'] = 0.7
        
        fig, axs = plt.subplots(4,2)
        #fig.suptitle('fields')
        
        name = ['$\overline{u}_x/u_\infty$', '$\overline{u}_y/u_\infty$', r'$\dfrac{\Tilde{\overline{p}}-p_{\infty}}{\rho u_{\infty}^2}$',r'$\dfrac{\nu_t}{c u_{\infty}}$']
        val = [ux_roi,uy_roi,p_roi,nut_roi]

        for i,U in enumerate([ux_pred,uy_pred,p_pred,nut_pred]):
            axs[i,0].tricontourf(x_roi+   xshift_roi,y_roi+   yshift_roi, val[i],200, cmap = 'coolwarm',norm=colors.TwoSlopeNorm(0))
            plot = axs[i,1].tricontourf(x_pred+   xshift_roi,y_pred+   yshift_roi, U, 200, cmap = 'coolwarm',norm=colors.TwoSlopeNorm(0))
            
            plot.set_clim([np.min(val[i]),np.max(val[i])])
            
            divider = make_axes_locatable(axs[i,1])
            cax = divider.append_axes('right', size="4%", pad=0.1)
            cbar = fig.colorbar(plot, cax=cax)
            cbar.ax.locator_params(nbins=3)
            cbar.formatter.set_powerlimits((0, 0))
            if i <2:
                axs[i,1].set_ylabel(name[i],rotation=0, labelpad = 21,y = 0.4)
            else:
                axs[i,1].set_ylabel(name[i],rotation=0, labelpad = 23,y = 0.1)
    
            plt.setp(axs[i,1].get_yticklabels(), visible=False) 
            if i < 3:
                plt.setp(axs[i,0].get_xticklabels(), visible=False) 
                plt.setp(axs[i,1].get_xticklabels(), visible=False) 
            if i == 0:
                axs[i,1].set_title('PINN', fontsize = '9')
                axs[i,0].set_title('RANS', fontsize = '9')
                
            axs[i,0].fill(x_srf +   xshift_roi,y_srf+   yshift_roi,c = 'black')
            axs[i,1].fill(x_srf +   xshift_roi,y_srf+   yshift_roi,c = 'black')
            axs[i,0].set_aspect('equal')
            axs[i,1].set_aspect('equal')
            
            axs[-1,1].set_xlabel('x/c')
            axs[-1,0].set_xlabel('x/c')
            
        fig.subplots_adjust(wspace=0.4, hspace=-0.9)   
        
        
        #fig.text(0.5, -0.5, 'x/c', va='center', ha='center')
        fig.text(0.03, 0.5, 'y/c', va='center', ha='center', rotation='vertical')
                
        plt.savefig('C:/Users/Timo/tubCloud/Shared/Masterarbeit_Timo_shared/04_pictures/'+case+'_fields.png', dpi = 1000, bbox_inches="tight", format='png')
        
        
        
#%% plot cp

        plt.rcParams['font.size'] = '22'
        
        p_foil_val = pfoil_raw[np.where((zfoil_raw == 0))][x_foil>=0]
        
        up = model.NN(model.foil)
        p_foil_pred = up[:,2]/0.5
        
        plt.figure()
        #plt.fill(x_srf,y_srf-yshift)
        plt.plot(BC_foil[:,0]+   xshift_roi,cp_foil_BC,'rd',markersize = 5, linewidth = 1.5)
        plt.plot(xfoil_x+  xshift_roi,cp_xfoil,'r-',label = r'XFOIL')
        plt.plot(x_foil[x_foil>=0]+  xshift_roi,p_foil_val/0.5, c = 'grey',marker = 'o',linestyle = '-',markersize = 1,label = r'RANS')
        plt.plot(foil[:,0][foil[:,0]>= 0]+  xshift_roi,p_foil_pred,'b-',label = r'PINN')

        
        #plt.xlim([0.25,1.1])
        plt.grid(True)
        leg = plt.legend( loc = 'lower right',ncol = 2,bbox_to_anchor=(1.05,-0.23),frameon=False)
        plt.ylabel('$c_p$')
        plt.xlabel('x/c')
        plt.savefig('fig/'+file+'_pfoil.png', bbox_inches="tight")        
        plt.close('all')
