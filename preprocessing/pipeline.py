import numpy as np
import argparse
import multiprocessing
import h5py
from progressbar import ProgressBar
import os
import time
import traceback
import PyTorch
import PyTorchAug

from get_data_info import get_labels, VideoGenerator
from get_video_info import get_numframe_duration, convert_video

def extract_videos(invideos_path, outlabels_file, outvideos_file, length, dimension):
    #queue_size = 20
    #out_video_filename = os.path.join(outvideos_path,'extract_videos.hdf5')
    #out_video_file = os.path.join(outvideos_path, 'extract_videos.t7')
    #mode = 'r+' if os.path.exists(out_video_filename) else 'w'

    #out_video_file = h5py.File(out_video_filename, mode)
    #extracted_videos = out_video_file.keys()
    #out_video_file.close()

    videos_list = [v[:-4] for v in os.listdir(invideos_path) if v[-4:] == '.mov']

    #videos_remaining = list(set(videos_list) - set(extracted_videos))

    #num_videos = len(videos_remaining)
    num_videos = len(videos_list)

    generator = VideoGenerator(videos_list, invideos_path, 'mov', length, dimension)

    video_labels = []
    video_tensors = []

    dl = {}
    dt = {}

    for v in range(num_videos):
        try:
            next_video = next(generator)
        except Exception:
            print 'Error in getting the next video: ', videos_list[v]

        video_name, video_tensor = next_video
        video_class = video_name.split('_')[0]
        print video_tensor.dtype
        print video_tensor.shape
        if video_tensor is None:
            print 'Could not read the video ', video_name
            continue
        video_labels.append(video_class)
        video_tensors.append(video_tensor)
        dl[v] =

    video_labels = np.asarray(video_labels).astype(np.float32)
    video_labels = video_labels.reshape(video_labels.shape[0],1)
    video_tensors = np.asarray(video_tensors)
    #video_tensors = np.asarray(video_tensors).astype(np.object)
    #video_tensors = video_tensors.reshape(video_tensors.shape[0],1)

    #video_out = np.concatenate((video_labels,video_tensors),axis=1)

    #labels = PyTorch.asFloatTensor(video_labels)
    #data = PyTorch.asFloatTensor(video_tensors)
    #PyTorchAug.save(outlabels_file, labels)
    #print 'Saved the labels to file ', outlabels_file
    #PyTorchAug.save(outvideos_file,data)
    #print 'Saved the videos to file ', outvideos_file



extract_videos('../data_temp', '../data_temp/extract_labels.t7', '../data_temp/extract_videos.t7', 16, (112,112))


'''    print 'Creating ', num_threads, ' processes to extract the videos.'
    get_video_queue = multiprocessing.Queue(maxsize=queue_size)
    stop_get = multiprocessing.Event()
    stop_extract = multiprocessing.Event()

    def get_videos(idx):
        generator = VideoGenerator(videos_remaining[idx:num_videos:num_threads],invideos_path,'mov',length,dimension)
        flag = True
        while flag:
            try:
                if get_video_queue.qsize() < queue_size:
                    try:
                        next_video = next(generator)
                    except ValueError:
                        continue
                    get_video_queue.put(next_video)
                else:
                    time.sleep(0.1)
            except Exception:
                flag = False
                print('Error in getting the next video.')
                print(traceback.print_exc())

    get_next_videos = [multiprocessing.Process(target=get_videos, args=[i]) for i in range(num_threads)]

    for nv in get_next_videos:
        nv.daemon = True
        nv.start()

    store_video_queue = multiprocessing.Queue()
    def store_videos():
        while not (stop_get.is_set() and get_video_queue.empty()):
            next_video = None
            while True:
                if not get_video_queue.empty():
                    next_video = get_video_queue.get()
                    if not next_video:
                        continue
                    break
                else
                    time.sleep(0.1)

            video_name, video_tensor = next_video
            video_class = video_name.split('_')[0]
            if video_tensor is None:
                print 'Could not read the video ', video_name
                continue
            store_video_queue.put((video_class,video_tensor))

    store_next_videos = [multiprocessing.Process(target=store_videos)]

'''