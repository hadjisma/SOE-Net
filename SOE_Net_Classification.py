###################################################
# Using a classifier with the SOE_Net features
# Coded by Isma Hadji (hadjisma@cse.yorku.ca)
###################################################
''' code for SOE_Net features classification'''
import time
import os
import input_data
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

eps = np.finfo(np.float32).eps
def Bhattachraya(p,q):
    dist = np.sqrt(1.0 - np.sum(np.sqrt(np.multiply(p,q))))
    return dist

PTH_TEST_LST = '/home/hadjisma/Dropbox/SOE_Net-tensorflow/data/Gamma_frm_TEST.txt'
RES_PTH = '/home/hadjisma/Dropbox/Gamma_frm/Results/'
print "[INFO] READING IN TEST DATA FROM :", PTH_TEST_LST
clips, start_indices, labels = input_data.load_clips_labels(PTH_TEST_LST)
all_labels = []
all_features = []
count = 0
classifier = 'NN'

for i in range(len(clips)):
    filename = clips[i].split('/')[-2] #[6]
    
    label = labels[i]
    filepath = RES_PTH + filename + '_' + str(start_indices[i]) + '.npy'
#    print filepath 
#    print filename
#    print label
#    raw_input('press')
    if i==0:
        old_filename = filename
        feat_size = np.shape(np.load(filepath).item().get('conv5'))
        feat = np.zeros(feat_size, dtype=np.float32)
    else:
        old_filename = clips[i-1].split('/')[-2] #[6]
    print filename, old_filename, label
        
    if os.path.isfile(filepath):    
        feat = feat + np.load(filepath).item().get('conv5')        
    
    if (filename != old_filename) or (i==len(clips)-1):
        if np.isnan(np.load(filepath).item().get('conv5')).any():
            print i, filepath
            print('detecetd NAN')
            continue
        else:
            count = count + 1
            all_features.append(feat)
            all_labels.append(labels[i-1])
            feat = np.zeros(feat_size, dtype=np.float32)

print ('Extracted %d features!' % count)
    
    
all_features = np.array(all_features)
all_labels = np.array(all_labels)
print all_features.shape, all_labels.shape
raw_input('press enter to continue')

if classifier == 'NN':
    
    #STEP 2: find Nearest Neighbors
    feat_sum = np.tile(np.sum(all_features, axis=1, keepdims=True), (1, all_features.shape[1]))
    feat_norm = np.divide(all_features, feat_sum)
    testNAN = np.nonzero(np.isnan(feat_norm[:,0]) == True)[0]
    print feat_norm.shape
    print testNAN
    print testNAN.shape
    raw_input('press enter')
    feat_norm = np.delete(feat_norm, tuple(testNAN), axis=0)
    all_labels = np.delete(all_labels, tuple(testNAN), axis=0)
    #print feat_norm.shape, all_labels.shape
    #raw_input('press enter')
    dist = pairwise_distances(feat_norm, metric=Bhattachraya)
    
    sorted_dist = np.argsort(dist, axis=1)
    nearest_feat = sorted_dist[:,1]
    #STEP3: do NN classification
    classified = all_labels[nearest_feat]
    diff = np.shape(np.array(np.where(classified != all_labels)))[1]
    
    accuracy = 1 - (float(diff) / float(len(all_labels)))
    print 'accuracy is :', accuracy
else:
    
    classified = []
    correct = 0.0
    total = 0.0
    test_data = np.zeros((1,all_features.shape[1]))
    test_targets = np.zeros((1,1))
    for i in range(all_features.shape[0]):
        total = total + 1
        # seperate data for one versus rest classification
        test_data[0,:] = np.array(all_features[i])
        test_targets[0,:] = np.array(all_labels[i])
        #print test_data
	    #print test_targets.shape
	    
        train_data = np.delete(all_features, i, axis=0)
        train_targets = np.delete(all_labels, i, axis=0)
        
        #print (train_data.shape)
        #print (train_targets.shape)
        #print (test_data.shape)
        #raw_input('press enter to continue')
        #train_data = scaler1.fit_transform(train_data)

        logreg = LogisticRegression(C=1.0,
                         multi_class='ovr',
                         penalty='l2', solver='liblinear', tol=0.001, max_iter=1000)

        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_data)
	    #train_data =  np.random.permutation(train_data)
        # TRAIN
        t0 = time.time()
        logreg.fit(train_data, train_targets)
        run_time = time.time() - t0
        print('Example %d run in %.3f s' % (i,run_time))
        
        test_data = scaler.transform(test_data)
        # TEST
        score = logreg.score(test_data, test_targets)
        prediction = logreg.predict(test_data)
        classified.append(prediction)
	   
        if test_targets == prediction:
            correct = correct + 1
            print("Test score: %.4f and predicted label is %d" % (score, prediction))
            print("correct %f | total %f | accuracy = %f" % (correct, total, (correct/total)))
            #raw_input('press enter to continue')



