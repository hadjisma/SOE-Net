###################################################
# EXTRACT SOE_MSOE features
# Coded by Isma Hadji (hadjisma@cse.yorku.ca)
###################################################
''' Full fledged test code to see SOEs at work'''

import time
import tensorflow as tf
import numpy as np
import sys
import input_data, os
import configure as cfg
import init_SOE_NET as init_net
import SOE_Net_model_full as model

#path to the root folder where all datasets can be found
root_dir = sys.argv[1] 
# name of the dataset where videos, features, frames are located
dataset = sys.argv[2] 
# name of the video for which you want to extract features
vid_name = sys.argv[3] 
 
    
sourcepath = root_dir + "/" + dataset + "/"
savepath = root_dir + "/" + dataset + "/" + "/features/"

if not os.path.exists(savepath):
    os.makedirs(savepath)

video = vid_name
print video

################################################################################################
#""" INITIALIZE SOE-NET PARAMETERS """
################################################################################################
tf.reset_default_graph()

orientations = init_net.initOrientations (cfg.ORIENTATIONS,cfg.SPEEDS,cfg.NUM_DIRECTIONS)
                
print"[INFO] INIT", orientations,"STANDARD ORIENTATIONS..."
basis = init_net.initSeparableFilters ('basis', cfg.FILTER_TAPS, filter_type="G2")

basis_2d = init_net.initSeparableFilters_SO('basis2d', cfg.FILTER_TAPS, filter_type="G2")
print"[INFO] INIT", basis,"3D SEPARABLE FILTERS..."
print"[INFO] INIT", basis_2d,"2D SEPARABLE FILTERS..."

biases_soe = init_net.initBiases('bias', 0)
              
print"[INFO] INIT", biases_soe,"SEPARABLE FILTERS..."
raw_input('Press enter to continue ... ')


################################################################################################
#""" READ THE INPUT DATA """
################################################################################################

# TEST DATA
tiny_batch_size = 1 # to extract features for one video at a time for this example code purposes
vid_path = sourcepath + video + "_clips.txt"
print "[INFO] READING IN TEST DATA FROM :", vid_path
test_clips, test_start_indices, test_labels = input_data.load_clips_labels(vid_path)
num_test_clips = len(test_clips)
print "[INFO] TOTAL NUMBER OF TESTING VIDEO CLIPS :", num_test_clips
#raw_input('Press enter to continue ... ')
TEST_ITERS = int(float(num_test_clips)/float(tiny_batch_size))
print "[INFO] NUMPBER OF ITERATIONS TO GO THROUGH TEST SET IS: ", TEST_ITERS
raw_input('Press enter to continue')

################################################################################################
#""" PREPARE DATA FOR TENSORFLOW """
################################################################################################
if cfg.CROP:
    input_shape   = [tiny_batch_size, cfg.TIME_S, cfg.IMG_S, cfg.IMG_S, 1]
else:
    input_shape   = [tiny_batch_size, cfg.TIME_S, cfg.IMG_RAW_H, cfg.IMG_RAW_W, 1]
 
batch_videos_ph  = tf.placeholder(tf.float32, shape=input_shape, name="batch_videos")
print batch_videos_ph
raw_input('press enter to continue')

################################################################################################   
#""" BUILD THE TENSORFLOW GRAPH """
################################################################################################
# example 1: Extract SOE features
soe = model.get_SOE(batch_videos_ph, basis, orientations, biases_soe)

# example 2: Extract MSOE features
msoe = model.get_MSOE(batch_videos_ph, basis, orientations, biases_soe)

# example 3: Extract SO features
so = model.get_SO(video, basis_2d, cfg.NUM_DIRECTIONS)

# example 4: Extract SOE_Net features
soenet = model.SOE_Net(batch_videos_ph, basis, orientations, biases_soe)

# TODO: add hybrid soe-net variant here where you save the fc layer output

# example 5: Extract MSOE_Net features (saving all features for potential VOS usage)
msoenet1,msoenet2,msoenet3,msoenet4 = model.MSOE_Net(batch_videos_ph, basis, orientations, biases_soe)

# example 6: Extract SO_Net features (saving all features for potential VOS usage)
sonet1,sonet2,sonet3,sonet4 = model.SO_Net(batch_videos_ph, basis_2d, cfg.NUM_DIRECTIONS, biases_soe)

init = tf.global_variables_initializer()
sess = tf.Session()
# initialize the non-trained variables
sess.run(init)
raw_input('Press enter to continue ... ')

test_indices = np.arange(num_test_clips)
test_start_idx = 0
SOE_FEAT  = []
MSOE_FEAT = []
SO_FEAT = [] 

SOE_NET_FEAT = []

SO1 = []
SO2 = []
SO3 = []
SO4 = []

MSOE1 = []
MSOE2 = []
MSOE3 = []
MSOE4 = []

for iter in range(TEST_ITERS):
    """ LOAD VIDEOS ONE BATCH AT A TIME """
    test_batch_clips, test_batch_indices, test_batch_labels, test_stop_idx = input_data.select_batch (test_clips,test_start_indices, test_labels, test_start_idx, test_indices, tiny_batch_size)
    print test_batch_indices
    batch_videos=input_data.load_frames(test_batch_clips,test_batch_indices,crop=False)
    print "size of clips in batch is : ", np.shape(batch_videos)
    print "stop_idx is : ", test_stop_idx
    test_start_idx = test_stop_idx
    
    """ RUN THE TENSORFLOW GRAPH """               
    tic = time.time()
    soe_feat = soe.eval(session=sess, feed_dict={batch_videos_ph: batch_videos}) 
    print "time to run through one mini-batch is: ", time.time()-tic
    
    tic = time.time()
    msoe_feat = sess.run(msoe, feed_dict={batch_videos_ph: batch_videos}) 
    print "time to run through one mini-batch is: ", time.time()-tic
    
    tic = time.time()
    so_feat = sess.run(so, feed_dict={batch_videos_ph: batch_videos}) 
    print "time to run through one mini-batch is: ", time.time()-tic

    tic = time.time()
    soenet_feat = sess.run(soenet, feed_dict={batch_videos_ph: batch_videos}) 
    print "time to run through one mini-batch is: ", time.time()-tic
    
    tic = time.time()
    msoe1,msoe2,msoe3,msoe4 = sess.run([msoenet1,msoenet2,msoenet3,msoenet4], feed_dict={batch_videos_ph: batch_videos}) 
    print "time to run through one mini-batch is: ", time.time()-tic
    
    tic = time.time()
    so1,so2,so3,so4 = sess.run([sonet1,sonet2,sonet3,sonet4], feed_dict={batch_videos_ph: batch_videos}) 
    print "time to run through one mini-batch is: ", time.time()-tic
    
    """ EXAMPLE: SAVE THE RESULTS FOR FURTHER USE (IN ANOTHER APPLICATION FOR EXAMPLE)"""
    for b in range(msoe_feat.shape[0]):
        # TODO: INSTEAD OF APPENDING ACCUMULATE RESULTS OBTAINED BY APPLYING GSP
        SOE_NET_FEAT.append(soenet_feat[b,:])
        if iter == 0:
            SOE_FEAT.append(soe_feat[b,0:-cfg.FILTER_TAPS,:,:,:])
            MSOE_FEAT.append(msoe_feat[b,0:-cfg.FILTER_TAPS,:,:,:])
            SO_FEAT.append(so_feat[b,0:-cfg.FILTER_TAPS,:,:,:])
            
            SO1.append(so1[b,0:-cfg.FILTER_TAPS,:,:,:])
            SO2.append(so2[b,0:-cfg.FILTER_TAPS,:,:,:])
            SO3.append(so3[b,0:-cfg.FILTER_TAPS,:,:,:])
            SO4.append(so4[b,0:-cfg.FILTER_TAPS,:,:,:])
            
            MSOE1.append(msoe1[b,0:-cfg.FILTER_TAPS,:,:,:])
            MSOE2.append(msoe2[b,0:-cfg.FILTER_TAPS,:,:,:])
            MSOE3.append(msoe3[b,0:-cfg.FILTER_TAPS,:,:,:])
            MSOE4.append(msoe4[b,0:-cfg.FILTER_TAPS,:,:,:])
            
        elif iter == TEST_ITERS-1:
            SOE_FEAT.append(soe_feat[b,cfg.FILTER_TAPS:,:,:,:])
            MSOE_FEAT.append(msoe_feat[b,cfg.FILTER_TAPS:,:,:,:])
            SO_FEAT.append(so_feat[b,cfg.FILTER_TAPS:,:,:,:])
            
            SO1.append(so1[b,cfg.FILTER_TAPS:,:,:,:])
            SO2.append(so2[b,cfg.FILTER_TAPS:,:,:,:])
            SO3.append(so3[b,cfg.FILTER_TAPS:,:,:,:])
            SO4.append(so4[b,cfg.FILTER_TAPS:,:,:,:])
            
            MSOE1.append(msoe1[b,cfg.FILTER_TAPS:,:,:,:])
            MSOE2.append(msoe2[b,cfg.FILTER_TAPS:,:,:,:])
            MSOE3.append(msoe3[b,cfg.FILTER_TAPS:,:,:,:])
            MSOE4.append(msoe4[b,cfg.FILTER_TAPS:,:,:,:])
        else:
            SOE_FEAT.append(soe_feat[b,cfg.FILTER_TAPS:-cfg.FILTER_TAPS,:,:,:])
            MSOE_FEAT.append(msoe_feat[b,cfg.FILTER_TAPS:-cfg.FILTER_TAPS,:,:,:])
            SO_FEAT.append(SO_FEAT[b,cfg.FILTER_TAPS:-cfg.FILTER_TAPS,:,:,:])
            
            SO1.append(so1[b,cfg.FILTER_TAPS:-cfg.FILTER_TAPS,:,:,:])
            SO2.append(so2[b,cfg.FILTER_TAPS:-cfg.FILTER_TAPS,:,:,:])
            SO3.append(so3[b,cfg.FILTER_TAPS:-cfg.FILTER_TAPS,:,:,:])
            SO4.append(so4[b,cfg.FILTER_TAPS:-cfg.FILTER_TAPS,:,:,:])
            
            MSOE1.append(msoe1[b,cfg.FILTER_TAPS:-cfg.FILTER_TAPS,:,:,:])
            MSOE2.append(msoe2[b,cfg.FILTER_TAPS:-cfg.FILTER_TAPS,:,:,:])
            MSOE3.append(msoe3[b,cfg.FILTER_TAPS:-cfg.FILTER_TAPS,:,:,:])
            MSOE4.append(msoe4[b,cfg.FILTER_TAPS:-cfg.FILTER_TAPS,:,:,:])
        
        #raw_input('Press enter to continue ... ')

SOE_NET_FEAT = np.concatenate(SOE_NET_FEAT, axis=0)   
SOE_FEAT = np.concatenate(SOE_FEAT, axis=0)   
MSOE_FEAT = np.concatenate(MSOE_FEAT, axis=0)   
SO_FEAT = np.concatenate(SO_FEAT, axis=0)  

SO1 = np.concatenate(SO1, axis=0)
SO2 = np.concatenate(SO2, axis=0)
SO3 = np.concatenate(SO3, axis=0)
SO4 = np.concatenate(SO4, axis=0)


MSOE1 = np.concatenate(MSOE1, axis=0)
MSOE2 = np.concatenate(MSOE2, axis=0)
MSOE3 = np.concatenate(MSOE3, axis=0)
MSOE4 = np.concatenate(MSOE4, axis=0)

#save results to npy file

feat_path = savepath + vid_name + ".npy"
extracted_features = {'soe': SOE_FEAT, 'msoe': MSOE_FEAT, 'SO': SO_FEAT, 'soe_net':SOE_NET_FEAT,\
                      'msoe1': MSOE1,'msoe2': MSOE2,'msoe3': MSOE3,'msoe4': MSOE4,\
                      'so1':SO1,'so2':SO2,'so3':SO3,'so4':SO4}
np.save(feat_path, extracted_features)
    
print 'Done'
           
       
        
        
        
        
        
    
