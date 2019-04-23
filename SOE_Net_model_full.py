###################################################
# SOE_NET building blocks
# Coded by Isma Hadji (hadjisma@cse.yorku.ca)
###################################################
"""Building the SOE_Net network.
Implements the functions for model building.
"""
import tensorflow as tf
import numpy as np
import configure as cfg

 
def power(v,p):
    vp = tf.pow(v, p, name='None')
    return vp

def S3D(G_a, G_b, G_c, G_d, G_e, G_f, G_g, G_h, G_i, G_j, ox,oy,ot):
    SteeredConv = power(ox,3)*G_a + 3*power(ox,2)*oy*G_b + 3*ox*power(oy,2)*G_c + power(oy,3)*G_d + 3*power(ox,2)*ot*G_e + 6*ox*oy*ot*G_f + 3*power(oy,2)*ot*G_g + 3*ox*power(ot,2)*G_h + 3*oy*power(ot,2)*G_i + power(ot,3)*G_j   
    return SteeredConv

def S2D(G_a, G_b, G_c, G_d, theta_in):
    theta = np.float32(np.divide(np.pi, 180.0) * theta_in)
    
    G_ka = power(np.cos(theta),3)
    G_kb = -3*(power(np.cos(theta),2))*np.sin(theta)
    G_kc =  3*(np.cos(theta))*(power(np.sin(theta),2))
    G_kd =  power(np.sin(theta),3)
    
    G = G_ka*G_a + G_kb*G_b + G_kc*G_c + G_kd*G_d
    
    return G

def MSR(G_a, G_b, G_c, G_d, G_e, G_f, G_g, G_h, G_i, G_j, ox,oy,ot):
    
    n = np.array([ox, oy, ot])
    n  = np.divide(n, np.linalg.norm(n))
    
    e1 = np.array([1.0,0.0,0.0], dtype = np.float32)
    e2 = np.array([0.0,1.0,0.0], dtype = np.float32)
    
    if ( np.absolute(np.arccos(np.dot(n, e1)/np.linalg.norm(n))) > np.absolute(np.arccos(np.dot(n, e2)/np.linalg.norm(n))) ):
        ua = np.cross(n,e1)
    else:
        ua = np.cross(n,e2)
    
    ua = ua / np.linalg.norm(ua)

    ub = np.cross(n,ua)
    ub = ub / np.linalg.norm(ub);
    
    SteeredConv = []
    M_or = np.array([np.cos(0) * ua + np.sin(0) * ub ])
    SteeredConv += [tf.square(S3D(G_a, G_b, G_c, G_d, G_e, G_f, G_g, G_h, G_i, G_j, M_or[0,0],M_or[0,1],M_or[0,2]))]
    M_or = np.array([np.cos((np.pi/4)) * ua + np.sin((np.pi/4)) * ub ])
    SteeredConv += [tf.square(S3D(G_a, G_b, G_c, G_d, G_e, G_f, G_g, G_h, G_i, G_j, M_or[0,0],M_or[0,1],M_or[0,2]))]
    M_or = np.array([np.cos(2*(np.pi/4)) * ua + np.sin(2*(np.pi/4)) * ub ])
    SteeredConv += [tf.square(S3D(G_a, G_b, G_c, G_d, G_e, G_f, G_g, G_h, G_i, G_j, M_or[0,0],M_or[0,1],M_or[0,2]))]
    M_or = np.array([np.cos(3*(np.pi/4)) * ua + np.sin(3*(np.pi/4)) * ub ])
    SteeredConv += [tf.square(S3D(G_a, G_b, G_c, G_d, G_e, G_f, G_g, G_h, G_i, G_j, M_or[0,0],M_or[0,1],M_or[0,2]))]
    
    SteeredConv = tf.concat(axis=4, values=SteeredConv, name='concatSOE')
    SteeredConv = tf.reduce_sum(SteeredConv, axis=4, keepdims=True)
    return SteeredConv


def conv3d(name, in_data, x, y, t):
    #fx = tf.Variable(fx, dtype=tf.float32, name="fx")
    #fy = tf.Variable(fy, dtype=tf.float32, name="fy")
    #ft = tf.Variable(ft, dtype=tf.float32, name="ft")
    
    fx = x[np.newaxis,np.newaxis,:,np.newaxis,np.newaxis]
    fy = y[np.newaxis,:,np.newaxis,np.newaxis,np.newaxis]
    ft = t[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]
    SepConv3d = []
    for ch in range(in_data.shape[4]):
        vol = in_data[:,:,:,:,ch]
        TmpConv3d = tf.nn.conv3d(vol[:,:,:,:,None], fx, strides=[1, 1, 1, 1, 1], padding='SAME')
        TmpConv3d = tf.nn.conv3d(TmpConv3d, fy, strides=[1, 1, 1, 1, 1], padding='SAME')
        TmpConv3d = tf.nn.conv3d(TmpConv3d, ft, strides=[1, 1, 1, 1, 1], padding='SAME')
        SepConv3d += [TmpConv3d]
    SepConv3d = tf.concat(axis=4, values=SepConv3d, name='concatSOE')            
    
    return SepConv3d

def conv2d(name, in_data, x, y):
    
    #fx = tf.Variable(fx, trainable=cfg.FILTER_TRAINABLE, dtype=tf.float32, name="fx")
    #fy = tf.Variable(fy, trainable=cfg.FILTER_TRAINABLE, dtype=tf.float32, name="fy")
    #ft = tf.Variable(ft, trainable=cfg.FILTER_TRAINABLE, dtype=tf.float32, name="ft")
    
    fx = x[np.newaxis,:,np.newaxis,np.newaxis]
    fy = y[:,np.newaxis,np.newaxis,np.newaxis]
    SepConv3d = []
    for ch in range(in_data.shape[4]):
        SepConv2d = []
        for fr in range(in_data.shape[1]):
            vol = in_data[:,fr,:,:,ch]
            TmpConv2d = tf.nn.conv2d(vol[:,:,:,None], fx, strides=[1, 1, 1, 1], padding='SAME')
            TmpConv2d = tf.nn.conv2d(TmpConv2d, fy, strides=[1, 1, 1, 1], padding='SAME')
            
            SepConv2d += [TmpConv2d[:,None,:,:,:]]
        SepConv2d = tf.concat(axis=1, values=SepConv2d, name='concatSOE')
        SepConv3d += [SepConv2d]
    SepConv3d = tf.concat(axis=4, values=SepConv3d, name='concatSOE')
        
    return SepConv3d
   
def SC3D(name, in_data, basis, orientations, bias):
    
    
    soe_net_conv = []

    G_a = conv3d('G_a', in_data, basis[0,:], basis[3,:], basis[3,:])
    G_b = conv3d('G_b', in_data, basis[2,:], basis[1,:], basis[3,:])
    G_c = conv3d('G_c', in_data, basis[1,:], basis[2,:], basis[3,:])
    G_d = conv3d('G_d', in_data, basis[3,:], basis[0,:], basis[3,:])
    G_e = conv3d('G_e', in_data, basis[2,:], basis[3,:], basis[1,:])
    G_f = conv3d('G_f', in_data, basis[4,:], basis[1,:], basis[1,:])
    G_g = conv3d('G_g', in_data, basis[3,:], basis[2,:], basis[1,:])
    G_h = conv3d('G_h', in_data, basis[1,:], basis[3,:], basis[2,:])
    G_i = conv3d('G_j', in_data, basis[3,:], basis[1,:], basis[2,:])
    G_j = conv3d('G_i', in_data, basis[3,:], basis[3,:], basis[0,:])
    conv = []
    for ch in range(orientations.shape[0]):
        #ox = tf.Variable(orientations[ch,0], dtype=tf.float32, name="ox")
        #oy = tf.Variable(orientations[ch,1], dtype=tf.float32, name="oy")
        #ot = tf.Variable(orientations[ch,2], dtype=tf.float32, name="ot")
        ox = orientations[ch,0]
        oy = orientations[ch,1]
        ot = orientations[ch,2]
        
        conv += [S3D(G_a, G_b, G_c, G_d, G_e, G_f, G_g, G_h, G_i, G_j, ox,oy,ot)]
    conv = tf.concat(axis=4, values=conv, name='concatG')
    
    soe_net_conv = conv
    #soe_net_conv = tf.nn.bias_add(soe_net_conv, bias)
    
        
    return soe_net_conv

def MSC3D(name, in_data, basis, orientations, bias):
    
    soe_net_conv = []
    
    G_a = conv3d('G_a', in_data, basis[0,:], basis[3,:], basis[3,:])
    G_b = conv3d('G_b', in_data, basis[2,:], basis[1,:], basis[3,:])
    G_c = conv3d('G_c', in_data, basis[1,:], basis[2,:], basis[3,:])
    G_d = conv3d('G_d', in_data, basis[3,:], basis[0,:], basis[3,:])
    G_e = conv3d('G_e', in_data, basis[2,:], basis[3,:], basis[1,:])
    G_f = conv3d('G_f', in_data, basis[4,:], basis[1,:], basis[1,:])
    G_g = conv3d('G_g', in_data, basis[3,:], basis[2,:], basis[1,:])
    G_h = conv3d('G_h', in_data, basis[1,:], basis[3,:], basis[2,:])
    G_i = conv3d('G_j', in_data, basis[3,:], basis[1,:], basis[2,:])
    G_j = conv3d('G_i', in_data, basis[3,:], basis[3,:], basis[0,:])
    
    conv = []
    for ch in range(orientations.shape[0]):
        #ox = tf.Variable(orientations[ch,0], trainable=cfg.ORIENTATION_TRAINABLE, dtype=tf.float32, name="ox")
        #oy = tf.Variable(orientations[ch,1], trainable=cfg.ORIENTATION_TRAINABLE, dtype=tf.float32, name="oy")
        #ot = tf.Variable(orientations[ch,2], trainable=cfg.ORIENTATION_TRAINABLE, dtype=tf.float32, name="ot")
        ox = orientations[ch,0]
        oy = orientations[ch,1]
        ot = orientations[ch,2]
            
        conv += [MSR(G_a, G_b, G_c, G_d, G_e, G_f, G_g, G_h, G_i, G_j, ox,oy,ot)]
    conv = tf.concat(axis=4, values=conv, name='concatG')
    
    soe_net_conv = conv
    #soe_net_conv = tf.nn.bias_add(soe_net_conv, bias)
    
        
    return soe_net_conv


def TPR(name, C):
    C_plus_minus = []
    
    C_plus_minus += [tf.square(tf.clip_by_value(C, 0, tf.reduce_max(C)), name="branch_pos")]
    C_plus_minus += [tf.square(tf.clip_by_value(C, tf.reduce_min(C), 0), name="branch_neg")]
    C_plus_minus = tf.concat(axis=4, values=C_plus_minus, name='ConcatRec')
    return C_plus_minus

def FWR(name, C):
    C_plus = tf.square(C, name)
    return C_plus

def DivNorm3d(name, in_data, eps):
    
    if eps is "max_based":
        epsilon  = np.finfo(np.float32).eps + tf.multiply(0.01,tf.reduce_max(in_data, axis=[1,2,3,4], keepdims= True))
        epsilon = tf.tile(epsilon, (1,in_data.shape[1],in_data.shape[2],in_data.shape[3],in_data.shape[4])) 
    elif eps is "std_based":
        var = tf.reduce_sum(in_data, axis=4, keepdims= True)
        epsilon = tf.reduce_mean(var, axis=[2,3], keepdims= True)
        epsilon = tf.tile(epsilon, (1,1,in_data.shape[2],in_data.shape[3],in_data.shape[4])) 

    sumE = tf.reduce_sum(in_data, axis=4, keepdims=True)
    normalizer = epsilon + tf.tile(sumE, (1,1,1,1,in_data.shape[4]))
    norm = tf.divide(in_data, normalizer)

    return norm

def AAP(wc, a):
    eta = 2*wc
    wl = eta*a
    sigma = np.divide(3,wl)
    ws = 2.5*wl
    T = np.ceil(np.divide((2*np.pi),ws))
    
    return sigma, wl, T
    
def Gaussian3d(sigma, taps):
    flen = (cfg.FILTER_TAPS*2)+1
    shape = (1, flen)
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[h< np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh!=0:
        h /=sumh
    g_filter = np.reshape(h, (flen,))
    
    return np.float32(g_filter)

def STPP(numL):
    sig = 1
    a = 0.35
    wc = 3* np.divide(np.sqrt(3), sig)
    sigma = np.zeros((numL,),dtype=np.float32)
    wl = np.zeros((numL,),dtype=np.float32)
    T = np.zeros((numL,), dtype=np.int)
    for L in range(numL):
        sigma[L], wl[L], T[L] = AAP(wc, a)
        wc = wl[L]   
    return T, sigma

def STPF(name, numL, L):
    _, sigma = STPP(numL)
    g_filter = Gaussian3d(sigma[L], cfg.FILTER_TAPS)
    
    return g_filter

def STPS(numL, L):
    strides, _ = STPP(numL)
    return strides[L]

def CPP(name, num_dir, L,rec_style):
    if rec_style is 'two_path':
        in_idx = np.arange(num_dir*num_dir*np.power(2,L))
        #mods = np.mod(in_idx,num_dir)
        divs = np.divide(in_idx, num_dir)
        cc_filter = []   
        for i in range(num_dir*np.power(2,L)):
            x = np.zeros((num_dir*num_dir*np.power(2,L),))
            out_idx = np.nonzero(divs==i)            
            x[out_idx] = 1.
            cc_filter.append(x)
    else:
        in_idx = np.arange(num_dir*num_dir)
        #mods = np.mod(in_idx,num_dir)
        divs = np.divide(in_idx, num_dir)
        cc_filter = []   
        for i in range(num_dir):
            x = np.zeros((num_dir*num_dir,))
            out_idx = np.nonzero(divs==i)            
            x[out_idx] = 1.
            cc_filter.append(x)

    cc_filter = np.divide(np.array(cc_filter), num_dir)
    cc_filter = np.float32(np.transpose(cc_filter))
    
    return cc_filter

def SP3D(name, in_data, numL, L):
    
    T = STPS(numL, L)
    g_filter = STPF(name, numL, L)
    st_pool = conv3d('STPool', in_data, g_filter, g_filter, g_filter)
    st_pool = st_pool[:,::T,::T,::T,:]
    return st_pool

def CP3D(name, in_data, L):
        
    cc_filter = CPP(name, cfg.NUM_DIRECTIONS,L+1, cfg.REC_STYLE)
    
    if cfg.REC_STYLE is 'two_path':
        shape_L1 = 2*cfg.NUM_DIRECTIONS
    else:
        shape_L1 = cfg.NUM_DIRECTIONS
    if in_data.shape[-1] == shape_L1:
        cc_pool = in_data
        
    else:
        cc = cc_filter[None, None, None, :,:]
        cc_pool = tf.nn.conv3d(in_data, cc, strides=[1, 1, 1, 1, 1], padding='VALID')
    return cc_pool
          
def GAP(name, in_data):
    g_pool = tf.reduce_mean(in_data, axis=[1,2,3])
    return g_pool

def GSP(name, in_data):
    flen = (cfg.FILTER_TAPS*2)+1
    if in_data.shape[1] >= flen:
	g_pool = tf.reduce_sum(in_data[:,cfg.FILTER_TAPS:-cfg.FILTER_TAPS,:,:,:], axis=[1,2,3])
    else:
        g_pool = tf.reduce_sum(in_data[:,:,:,:,:], axis=[1,2,3])
    return g_pool

def FCL(name, in_data, weight, bias, dropout):
    fc = tf.reshape(in_data, [in_data.shape[0], weight.shape[0]])
    if weight.shape[1] == cfg.NUM_CLASSES:
        fc = tf.matmul(fc, weight) + bias
    else:
        fc = tf.matmul(fc, weight) + bias
        fc = tf.nn.relu(fc, name=name)
        fc = tf.nn.dropout(fc, dropout)
    return fc
