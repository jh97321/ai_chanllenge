#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
@author: Xiangru Yang
@last update: 2019/10/29
@function: re-id game submit
'''

import os
import cv2
import json
import time
import timeit
import shutil
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.autograd import Variable
from scipy.spatial.distance import cdist
from model import pda
import random
import argparse
from re_ranking import re_ranking


def Generatematfile(MODEL_PATH, query_img_path, gallery_img_path, query_img_mat, gallery_img_mat):
    extractor = pfextractor(MODEL_PATH)

    if not os.path.exists(query_img_mat):
        os.makedirs(query_img_mat)
    if not os.path.exists(gallery_img_mat):
        os.makedirs(gallery_img_mat)

#==============Generating query features mat

    query_img_list = os.listdir(query_img_path)
    # query_img_list.sort()
    query_img_list_num = len(query_img_list)

    query_img_mat_path = query_img_mat + '/' + 'query_img_mat.json'
    query_feature_mat_path = query_img_mat + '/' + 'query_feature_mat.json'

    with open(query_img_mat_path, "w") as f_query1:
        json.dump(query_img_list, f_query1)


    i = 0
    query_img_features = []
    for query_single_img in query_img_list:
        if '.png' not in query_single_img:
            continue
        query_single_img_path = query_img_path + '/'+ query_single_img
        query_img = cv2.imread(query_single_img_path)
        query_single_img_feature = extractor.extract(query_img)
        query_img_features.append(query_single_img_feature)
        i += 1
        print("query_single_img feature: %d/%d, %s is done!" % (i, query_img_list_num, query_single_img))

    with open(query_feature_mat_path, "w") as f_query2:
        json.dump(query_img_features, f_query2)

    print("Generating query matrix file completed!")




    # ==============Generating query features mat
    gallery_img_list = os.listdir(gallery_img_path)
    # gallery_img_list.sort()
    gallery_img_list_num = len(gallery_img_list)

    gallery_img_mat_path = gallery_img_mat + '/' +  'gallery_img_mat.json'
    gallery_feature_mat_path = gallery_img_mat + '/' + 'gallery_feature_mat.json'

    with open(gallery_img_mat_path, "w") as f_gallery1:
        json.dump(gallery_img_list, f_gallery1)

    j = 0
    gallery_img_features = []
    for gallery_single_img in gallery_img_list:
        if '.png' not in query_single_img:
            continue
        gallery_single_img_path = gallery_img_path + '/' + gallery_single_img
        gallery_img = cv2.imread(gallery_single_img_path)
        gallery_single_img_feature = extractor.extract(gallery_img)
        gallery_img_features.append(gallery_single_img_feature)
        j += 1
        print("gallery_single_img feature: %d/%d, %s is done!" % (j, gallery_img_list_num, gallery_single_img))

    with open(gallery_feature_mat_path, "w") as f_gallery2:
        json.dump(gallery_img_features, f_gallery2)
    print("Generating gallery matrix file completed!")

    return 0


def Matching(query_img_mat, gallery_img_mat):
    query_img_mat_path = query_img_mat + '/' + 'query_img_mat.json'
    query_feature_mat_path = query_img_mat + '/' + 'query_feature_mat.json'

    gallery_img_mat_path = gallery_img_mat + '/' + 'gallery_img_mat.json'
    gallery_feature_mat_path = gallery_img_mat + '/' + 'gallery_feature_mat.json'

    with open(query_img_mat_path, "r") as f_query11:
        query_name = json.load(f_query11)

    with open(query_feature_mat_path, "r") as f_query12:
        q_feature = json.load(f_query12)

    with open(gallery_img_mat_path, "r") as f_gallery11:
        g_name = json.load(f_gallery11)

    with open(gallery_feature_mat_path, "r") as f_gallery12:
        g_feature = json.load(f_gallery12)

    
    ##***********without rerank(syl)
    result_all = {}
    for i in range(len(query_name)):
        q_img = query_name[i]
        q1_feature = np.array(q_feature[i])
        q1_feature = np.reshape(q1_feature, (1, 4096))

        g_feature = np.array(g_feature)
        g_feature = np.reshape(g_feature,(len(g_feature),4096))
        distmat = (cdist(q1_feature, g_feature, metric='cosine'))[0]
        distmat_np = np.array(distmat)
        indices = np.argsort(distmat_np)

        this_result = [g_name[indices[j]] for j in range(200)]
        result_all[q_img] = this_result


    with open("./reid_yxr_160_11_22.json", 'w+') as f8:
        json.dump(result_all, f8)
    

    print('Matching over!')








class pfextractor():
    def __init__(self, model_path):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU index
        self.model = pda.PDA().cuda()
        if type(model_path) == str:
            print('the choice is 1')
            self.model.load_state_dict(torch.load(model_path))
        elif type(model_path) == dict:
            print('the choice is 2')
            self.model.load_state_dict(model_path) 
        else:
            print('the choice is 3')
            self.model.load_state_dict(model_path, strict=False)

        self.transform = transforms.Compose([
            transforms.Resize((384, 128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])



    def fliphor(self, inputs):
        inv_idx = torch.arange(inputs.size(3)-1,-1,-1).long()  # N x C x H x W
        return inputs.index_select(3,inv_idx)


    def extract(self, image):
        self.model.eval()
        # ff = torch.FloatTensor(1, 4096).zero_()
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = self.transform(image)
        image = image.unsqueeze_(0).float()
        image = Variable(image)
        output = self.model(image.cuda())
        f = output[0].data.cpu()


        flip_image = self.fliphor(image)
        flip_output = self.model(flip_image.cuda())
        flip_f = flip_output[0].data.cpu()
        ff = f + flip_f



        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        return ff.tolist()




if __name__ == '__main__':

    #THRESHOLD = 0.35

    parser = argparse.ArgumentParser()

    parser.add_argument('--query_img_path', type=str, default="/media/h/d911f32b-d7e0-4a5a-9689-2e835f62147c/aichallenge/query_a", help='query_img_path')
    parser.add_argument('--gallery_img_path', default="/media/h/d911f32b-d7e0-4a5a-9689-2e835f62147c/aichallenge/gallery_a", type=str, help='gallery_img_path')
    parser.add_argument('--model', type=str, default="/media/syl/BIG_ONE/REID/A_compete/match/439.pt", help='model path')
    parser.add_argument('--query_img_mat', type=str, default="./mat/query_mat", help='query_img mat path')
    parser.add_argument('--gallery_img_mat', type=str, default="./mat/gallery_mat", help='gallery_img mat path')
    parser.add_argument('--save_path', type=str, help='result save path')
    parser.add_argument('--generate_mat_file', default=True,action="store_true", help='generate_gallerymat_file function')
    parser.add_argument('--generate_gallery_mat_file', action="store_true", help='generate_querymat_file function')
    parser.add_argument('--matching', default=True ,action="store_true", help='matching function')
    args = parser.parse_args()


    query_img_path = args.query_img_path
    gallery_img_path = args.gallery_img_path
    MODEL_PATH = "/media/h/d911f32b-d7e0-4a5a-9689-2e835f62147c/aichallenge_match/model_160.pt" #args.model
    flag_keys = False


    query_img_mat = args.query_img_mat
    gallery_img_mat = args.gallery_img_mat
    print(args.gallery_img_mat)
    if args.generate_mat_file:
        print("start Generatematfile==========================")
        if flag_keys:
            Generatematfile(new_model, query_img_path, gallery_img_path, query_img_mat, gallery_img_mat) #MODEL_PATH
        else:
            Generatematfile(MODEL_PATH, query_img_path, gallery_img_path, query_img_mat, gallery_img_mat) 

    if args.matching:
        print("start calculate the average distance between camerasytleA and B==========================")
        Matching(query_img_mat, gallery_img_mat)

