
from einops.layers.torch import Rearrange, Reduce
#from typing import Union, List, Tuple, Literal
from torch.nn import functional as F
#from torch.utils.data import DataLoader
import torchvision.transforms as transforms
#from PIL import Image
#from os.path import join, exists, isfile
#from os import makedirs
#from tqdm.auto import tqdm
#from torchvision.utils import save_image
from skimage.transform import resize
#import matplotlib.pyplot as plt
#import vision_transformer as vits
#import configparser
#import argparse
import torchvision
import numpy as np
import torch
#import cv2
import os
#import faiss
import lpips
#import utils

import vision_transformer as vits
#from scipy.spatial.distance import cdist


def Model(args):
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    for name, p in model.named_parameters():
        p.requires_grad = False
    model.eval()
    model.to(args.device)
    url = None 
    #elif args.arch == "vit_base" and args.patch_size == 8:
    url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
    args.split_factor = 28

    if url is not None:
        print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
        model.load_state_dict(state_dict, strict=True)
    else:
        print("There is no reference weights available for this model => We use random weights.")

    return model


class Cosine_Similarity(torch.nn.Module):
    def __init__ (self,
                  similarity_measure = torch.nn.CosineSimilarity()):

        super().__init__()
        self.similarity_measure = similarity_measure
        
    def forward(self, input_image):
        self.input_image_orig = input_image
        H = self.input_image_orig.shape[0]
        W = self.input_image_orig.shape[1]
        self.input_image = torch.zeros([H + 1, W + 1, 1, self.input_image_orig.shape[-1]])
        self.input_image[:H, :W] = self.input_image_orig
        self.similarity_map = np.empty((H, W), dtype=np.float32)
        for m in range(H):
           for n in range(W):
               
               self.similarity_map[m,n] = np.sum(
                                     [ [ int((n+l)>=0 and (m+k)>=0) * int(m+k<H and n+l<W) * (
                                     self.similarity_measure(self.input_image[m,n], self.input_image[m+k, n+l])
                                     ).detach().cpu().numpy() 
                                     for k in [-1,0,1] ] 
                                     for l in [-1,0,1] ]  
                                      )-1
               
        return self.similarity_map

def UpScale(image_encoding, att_scale, opt):
    
    if att_scale>1: 
        image_encoding = image_encoding.permute(2, 0, 1)#@@DINO 
        image_encoding = (torch.nn.functional.interpolate(
                        image_encoding.unsqueeze(0),
                        scale_factor=att_scale,
                        mode="nearest",
                    )[0]#.cpu().numpy()
                )
        image_encoding = image_encoding.permute(1, 2, 0)#@@DINO  
    return image_encoding#.permute(1,2,0)#@@@ features: DINO                 

def N_support(image_encoding, neigh_patch_size, CosineSim):
    Re_arrange_P1 = Rearrange('(h m) (w n) c -> h w (m n c)', m=neigh_patch_size, n=neigh_patch_size)    
    image_encodingP1 = Re_arrange_P1(image_encoding);
    similarity_mapP1 = CosineSim.forward(image_encodingP1.unsqueeze(2))#.flatten()
    return similarity_mapP1


def SNSM(image_encoding, opt):
    CosineSim = Cosine_Similarity()
    Arrange = Rearrange('(h w) c-> h w c', h=opt.split_factor)  
    image_encoding = Arrange(image_encoding)
    image_encoding1 = UpScale(image_encoding, opt.upscale_factor, opt)
    SimMap_P2 = N_support(image_encoding1, opt.window_size, CosineSim)
    return torch.tensor(SimMap_P2).to(image_encoding.device)#.get_device())
                
def snsm_feature_extract(input_data, model, device, opt, f_name):
    #if not exists(opt.result_save_folder):
    #    makedirs(opt.result_save_folder)
    #pool_size_loc = opt.emb_dim
    Re_arrange = Rearrange('h w -> (h w)')

    with torch.no_grad():
        #tqdm.write('====> Extracting Features')
        input_data = input_data.unsqueeze(0).to(device)
             
        dino_image_encoding = model.get_intermediate_layers(input_data, opt.encoder)[0]

        image_encoding = lpips.normalize_tensor(dino_image_encoding)[0]
           
        #assert opt.snsm==True 
        
        #bl_image_encoding = dino_image_encoding.view(28,28,768).mean(dim=-1).detach().cpu().numpy()       
        snsm_encoding = SNSM(image_encoding, opt).detach().cpu().numpy()
        return snsm_encoding
        

def input_transform(resize=(224, 224)):
    if resize[0] > 0 and resize[1] > 0:
        return transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

class Args:
    arch = "vit_base"
    patch_size = 8
    backbone = "DINOv1"
    encoder = 1
    upscale_factor = 4
    window_size = 2
    nocuda = True

opt = Args()
opt.device = torch.device("cuda" if torch.cuda.is_available() and not opt.nocuda else "cpu")

model = Model(opt)

def run_snsm(image):
    # image = PIL image from Gradio
    size = image.size
    img = input_transform()(image)
    snsm_feature_map = snsm_feature_extract(
        img, model, opt.device, opt, "input"
    )
    resize_dims = (size[1], size[0])
    result = resize(snsm_feature_map, resize_dims, anti_aliasing=True)
    result = (result - result.min()) / (result.max() - result.min() + 1e-8)
    return result    

import gradio as gr

demo = gr.Interface(
    fn=run_snsm,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="numpy"),
    title="SNSM Feature Map"
)

demo.launch()


