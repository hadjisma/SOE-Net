###################################################
# data loading functions
# Coded by Isma Hadji (hadjisma@cse.yorku.ca)
###################################################
"""
Functions to read batches of video frames to use as an input
"""
import os, time
import random
import cv2
import numpy as np
import configure as cfg


def load_meanfile(meanfile):
    f_mean = np.load(meanfile)   
    return f_mean

def load_clips_labels(path_lst):
    print "[Info] loading lists..."

    with open(path_lst, 'r') as f:
        data_lst = f.read().splitlines()

    N = len(data_lst)
    clips = []
    start_indices=[]
    labels = []
    for i in range(N):
        line = data_lst[i]
        filename, frame_start, cat_id = line.split(' ')
        clips.append( filename)
        start_indices.append(int(frame_start))
        labels.append(int(cat_id))
    return clips, start_indices, labels


def read_clip_and_label(clips, start_indices, labels, video_indices, meanfile, batch_size, start_pos=-1, shuffle=False):
    
    read_dirnames = []
    read_clips = []
    read_indices = []
    data = []
    label = []
    batch_index = 0
    next_batch_start = -1
    
    if start_pos < 0:
        shuffle = True
    if shuffle:
        random.seed(time.time())
        random.shuffle(video_indices)
    else:
        video_indices = video_indices[start_pos: len(clips)]
  
    for index in video_indices:
        if(batch_index>=batch_size):
            next_batch_start = index
            break
    
        dirname = clips[index]
        start_idx = start_indices[index]
        tmp_label = labels[index]
        tmp_data = get_clip(dirname, start_idx, cfg.CROP, meanfile)
        
        if(len(tmp_data)!=0):
            data.append(tmp_data)
            label.append(int(tmp_label))
            batch_index = batch_index + 1
            read_dirnames.append(dirname)
            read_clips.append(start_idx)
            read_indices.append(index)

    valid_len = len(data)
    pad_len = batch_size - valid_len
    if pad_len:
        for i in range(pad_len):
            data.append(tmp_data)
            label.append(int(tmp_label))

    np_arr_data = np.array(data).astype(np.float32)
    np_arr_label = np.array(label).astype(np.int64)

    return np_arr_data, np_arr_label, next_batch_start, read_dirnames, read_clips,read_indices, valid_len


def next_batch(num_samples, start_idx, batch_size):
    stop_idx = min(start_idx + batch_size, num_samples)
    return stop_idx

def load_frames(batch_clips, batch_indices, crop):
    img_size = (cfg.IMG_RAW_H, cfg.IMG_RAW_W)
    N = len(batch_clips)

    if crop:
        data = np.zeros((N, cfg.TIME_S) + (cfg.IMG_S , cfg.IMG_S) + (1,) ,dtype=np.uint8)
    else:
        data = np.zeros((N, cfg.TIME_S) + (cfg.IMG_RAW_H , cfg.IMG_RAW_W) + (1,) ,dtype=np.uint8)
    
    for i in range(N):
        video_path=batch_clips[i]
        start_frame=batch_indices[i]
        for frame_count in range(cfg.TIME_S):
            filename = cfg.IMAGE_FORMAT.format(start_frame)
            #print "filename is:", os.path.join(video_path, filename)
            img = cv2.imread(os.path.join(video_path, filename))
            #print img.shape
            #raw_input('check image size')
            img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
            # in case image was not resized at extraction time
            if img.shape != img_size:
                img = cv2.resize(
                    img,
                    img_size[::-1],
                    interpolation=cv2.INTER_LINEAR)
            # to crop or not to crop that is the question
            if crop:
                crop_x = int((cfg.IMG_RAW_H - cfg.IMG_S)/2)
                crop_y = int((cfg.IMG_RAW_W - cfg.IMG_S)/2)
                data[i, frame_count, :,:,:] = img[crop_x:crop_x+cfg.IMG_S,crop_y:crop_y+cfg.IMG_S,np.newaxis]
            else:
                data[i, frame_count, :,:,:] = img[:,:,np.newaxis]

            start_frame += 1
    return data

def load_color_frames(batch_clips, batch_indices, crop):
    img_size = (cfg.IMG_RAW_H, cfg.IMG_RAW_W)
    N = len(batch_clips)

    if crop:
        data = np.zeros((N, cfg.TIME_S) + (cfg.IMG_S , cfg.IMG_S) + (3,) ,dtype=np.uint8)
    else:
        data = np.zeros((N, cfg.TIME_S) + (cfg.IMG_RAW_H , cfg.IMG_RAW_W) + (3,) ,dtype=np.uint8)
    
    for i in range(N):
        video_path=batch_clips[i]
        start_frame=batch_indices[i]
        for frame_count in range(cfg.TIME_S):
            filename = cfg.IMAGE_FORMAT.format(start_frame)
            #print "filename is:", os.path.join(video_path, filename)
            img = cv2.imread(os.path.join(video_path, filename))
            #img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
            # in case image was not resized at extraction time
            if (img.shape[0] != img_size[0]) or (img.shape[1] != img_size[1]):
                img = cv2.resize(
                    img,
                    img_size[::-1],
                    interpolation=cv2.INTER_LINEAR)
            # to crop or not to crop that is the question
            if crop:
                crop_x = int((cfg.IMG_RAW_H - cfg.IMG_S)/2)
                crop_y = int((cfg.IMG_RAW_W - cfg.IMG_S)/2)
                data[i, frame_count, :,:,:] = img[crop_x:crop_x+cfg.IMG_S,crop_y:crop_y+cfg.IMG_S,:]
            else:
                data[i, frame_count, :,:,:] = img[:,:,:]

            start_frame += 1
    return data

def select_batch (clips,start_indices,labels,start_idx,indices, batch_size):
    stop_idx=next_batch(len(clips), start_idx, batch_size)

    batch_clips=[]
    batch_indices=[]
    batch_labels=[]
    batch_idx=indices[start_idx:stop_idx]
    
    for i in batch_idx:
        batch_clips.append(clips[i])
        batch_indices.append(int(start_indices[i]))
        batch_labels.append(int(labels[i]))
    return batch_clips, batch_indices, batch_labels, stop_idx

