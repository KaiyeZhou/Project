# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
import IPython

import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    # parser.add_argument(
    #     '--image-ext',
    #     dest='image_ext',
    #     help='image file name extension (default: jpg)',
    #     default='jpg',
    #     type=str
    # )
    # parser.add_argument(
    #     'im_or_folder', help='image or folder of images', default=None
    # )
    parser.add_argument(
        'video', help='input video', default=None
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main(args):
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    # if os.path.isdir(args.video) #

    im_list = video2imgs(args.video)

    # if os.path.isdir(args.im_or_folder):
    #     im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    # else:
    #     im_list = [args.im_or_folder]


    # vis_imgshape = vis_img.shape
    # size = (vis_imgshape[1], vis_imgshape[0])
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # size = (1600,900)
    # videoWriter = cv2.VideoWriter(args.output_dir + os.path.basename(args.video), fourcc, 20, size)
    time_start = time.time()
    
    for i, im in enumerate(im_list):
        # out_name = os.path.join(
        #     args.output_dir, '{}'.format(os.path.basename(im_name) + '.pdf')
        # )
        # logger.info('Processing {} -> {}'.format(im_name, out_name))
        # im = cv2.imread(im_name)

        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps, cls_bodys = infer_engine.im_detect_all(
                model, im, None, timers=timers
            )
        # IPython.embed()
        logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        for k, v in timers.items():
            logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
        if i == 0:
            logger.info(
                ' \ Note: inference on the first image will be slower than the '
                'rest (caches and auto-tuning need to warm up)'
            )
        # if i % 5 == 0:
        All_Coords = vis_utils.vis_one_image(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            cls_boxes,
            None,            # im_name,           
            None,            # args.output_dir,      
            
            cls_segms,
            cls_keyps,
            cls_bodys,
            dataset=dummy_coco_dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=0.7,
            kp_thresh=2
        )
        if cls_keyps is None:
            vis_img = visualize(im, All_Coords)
        else:
            print('keypoint')
            vis_img = All_Coords
        
        if i ==0:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v'.encode('utf-8'))
            # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            #fourcc = cv2.VideoWriter_fourcc(*'XVID')
            vis_imgshape = vis_img.shape
            size = (vis_imgshape[1], vis_imgshape[0])
            videoWriter = cv2.VideoWriter(args.output_dir + os.path.basename(args.video).split('.')[0] + '.mp4', fourcc, 25, size)
            # videoWriter.write(vis_img)
        # IUV_list.append(vis_img)
        #imgs2video

        # videoWriter = cv2.VideoWriter(args.output_dir + '/zuoqianceyang5.mp4', fourcc, 20, size)
        ## 5 qu 1
        # if i % 5 == 0:
        videoWriter.write(vis_img)
    videoWriter.release()
    time_end = time.time()
    print('totally cost: {:.3f}s', time_end - time_start)

def video2imgs(videopath):
    # savedir = '/home/server010/zhoukaiye/jianshen/jianshenshuju/728datasetpic/youcefeng/'
    img_list = []
    # if not os.path.exists(savedir):
    #     os.makedirs(savedir)
    vc = cv2.VideoCapture(videopath)
    c=1
    if vc.isOpened():
        print ('successfully open')
        # rval, frame = vc.read()
        # img_list.append(frame)
        # cv2.imwrite(savedir + 'youcefeng' + str(c) + '.jpg', frame)
    else:
        exit()
    while True:
        c += 1
        rval, frame = vc.read()
        if rval == False: break

        # cv2.imwrite(savedir + 'youcefeng' + str(c) + '.jpg', frame)
        img_list.append(frame)
        # print (c)

    vc.release()
    return img_list

def visualize(im, IUV):
    fig = plt.figure(figsize=[16,9])
    plt.imshow( im[:,:,::-1] )
    plt.contour( IUV[:,:,1]/256.,10, linewidths = 1 )   #IUV
    plt.contour( IUV[:,:,2]/256.,10, linewidths = 1 )   #IUV
    plt.axis('off') ; 
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    
    # plt.savefig('/home/server010/zhoukaiye/DensePose/densepose/DensePoseData/jianshenout/jianshenfigure1.jpg') 
    plt.savefig('/home/server010/zhoukaiye/jianshen/jianshenshuju/shuangchaungvideos/shuangchaungzhanshi10.jpg', dpi = 100)
    plt.close()
    vis_img = cv2.imread('/home/server010/zhoukaiye/jianshen/jianshenshuju/shuangchaungvideos/shuangchaungzhanshi10.jpg')
    return vis_img
    # i = 0
    # folder = '/home/server010/zhoukaiye/DensePose/densepose/DensePoseData/jianshenout/youcefeng/'
    # for root, dirs, files in os.walk(folder):
    #     for file in files:
    #         i += 1
    #         fn = 'youcefeng' + str(i) + '.jpg'
    #         pn = 'youcefeng' + str(i) + '_IUV.png'
    #         qn = 'youcefeng' + str(i) + '_INDS.png'
    #         im = cv2.imread(root + '/' + fn)
    #         IUV = cv2.imread(root + '/' + pn)
    #         INDS = cv2.imread(root + '/' + qn,  0)
    #         fig = plt.figure(figsize=[16,9])
    #         plt.imshow( im[:,:,::-1] )
    #         plt.contour( IUV[:,:,1]/256.,10, linewidths = 1 )
    #         plt.contour( IUV[:,:,2]/256.,10, linewidths = 1 )
    #         plt.axis('off') ; 
    #         plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    #         plt.margins(0,0)
    #         plt.savefig('/home/server010/zhoukaiye/DensePose/densepose/DensePoseData/jianshenout/results/youcefeng1/youcefeng' + str(i) + '.jpg') 
    #         plt.close()
    #         #plt.show()

# def imgs2video()
#     folder = '/media/data2/zky/DensePose-master/pytorch/build/densepose/DensePoseData/results/zuoqianceyang5crop'
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     img_shape = cv2.imread('/media/data2/zky/DensePose-master/pytorch/build/densepose/DensePoseData/results/zuoqianceyang5crop/zuoqianceyang1.jpg').shape
#     size = (img_shape[1], img_shape[0])
#     # videoWriter = cv2.VideoWriter('/media/data2/zky/DensePose-master/pytorch/build/densepose/DensePoseData/results/zuoqianceyang5.mp4', fourcc, 20, size)
#     videoWriter = cv2.VideoWriter(args.output_dir + '/zuoqianceyang5.mp4', fourcc, 20, size)

#     i = 0
#     for root, dirs, files in os.walk(folder):
#         for file in files:
#             i += 1
#             fn = 'zuoqianceyang' + str(i) + '.jpg'
#             image = cv2.imread(root + '/' + fn)
#             videoWriter.write(image)
#     videoWriter.release()


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
