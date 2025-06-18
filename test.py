import argparse
from pathlib import Path
import os
import sys
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
from function import calc_mean_std, normal, coral
import models.Mambaformer as Mambaformer
import models.MambaformerGLT  as  MambaformerGLT 
import matplotlib.pyplot as plt
from matplotlib import cm
from function import normal
import numpy as np
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "my_modify"))

def test_transform(size, crop):
    transform_list = []
   
    if size != 0: 
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform
def style_transform(h,w):
    k = (h,w)
    size = int(np.max(k))
    print(type(size))
    transform_list = []    
    transform_list.append(transforms.CenterCrop((h,w)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def content_transform():
    
    transform_list = []   
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

  

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str,
                    help='File path to the content image')
parser.add_argument('--content_dir', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style', type=str,
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--style_dir', type=str,
                    help='Directory path to a batch of style images')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')
parser.add_argument('--vgg', type=str, default='./experiments/vgg_normalised.pth')
parser.add_argument('--decoder_path', type=str, default='experiments/decoder_iter_160000.pth') 
parser.add_argument('--mbfr_path', type=str, default='experiments/transformer_iter_160000.pth')
parser.add_argument('--embedding_path', type=str, default='experiments/embedding_iter_160000.pth')


parser.add_argument('--style_interpolation_weights', type=str, default="")
parser.add_argument('--a', type=float, default=1.0)
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
parser.add_argument('--hidden_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
args = parser.parse_args()

# Advanced options
content_size=512
style_size=512
crop='store_true'
save_ext='.jpg'
output_path=args.output
preserve_color='store_true'
alpha=args.a

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Either --content or --content_dir should be given.
if args.content:
    content_paths = [Path(args.content)]
else:
    content_dir = Path(args.content_dir)
    content_paths = [f for f in content_dir.glob('*')]

# Either --style or --style_dir should be given.
if args.style:
    style_paths = [Path(args.style)]    
else:
    style_dir = Path(args.style_dir)
    style_paths = [f for f in style_dir.glob('*')]

if not os.path.exists(output_path):
    os.mkdir(output_path)


vgg = MambaformerGLT.vgg
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:44])

decoder = MambaformerGLT.decoder
mbfr = Mambaformer.mambaformer()  # m11215021

embedding = MambaformerGLT.PatchEmbed()

decoder.eval()
mbfr.eval()
vgg.eval()
from collections import OrderedDict
new_state_dict = OrderedDict()
state_dict = torch.load(args.decoder_path)
for k, v in state_dict.items():
    #namekey = k[7:] # remove `module.`
    namekey = k
    new_state_dict[namekey] = v
decoder.load_state_dict(new_state_dict)

new_state_dict = OrderedDict()
state_dict = torch.load(args.mbfr_path)

for k, v in state_dict.items():
    #namekey = k[7:] # remove `module.`
    namekey = k
    new_state_dict[namekey] = v
mbfr.load_state_dict(new_state_dict)

new_state_dict = OrderedDict()
state_dict = torch.load(args.embedding_path)
for k, v in state_dict.items():
    #namekey = k[7:] # remove `module.`
    namekey = k
    new_state_dict[namekey] = v
embedding.load_state_dict(new_state_dict)

network = MambaformerGLT.mambaformerGLT(vgg,decoder,embedding,mbfr,args)
network.eval()
network.to(device)

# [time]
inference_times = []  # Store individual inference times
total_pairs = len(content_paths) * len(style_paths)  # Total number of content-style pairs

content_tf = test_transform(content_size, crop)
style_tf = test_transform(style_size, crop)

for content_path in content_paths:
    for style_path in style_paths:
        print(content_path)
       
      
        content_tf1 = content_transform()       
        content = content_tf(Image.open(content_path).convert("RGB"))

        h,w,c=np.shape(content)    
        style_tf1 = style_transform(h,w)
        style = style_tf(Image.open(style_path).convert("RGB"))

      
        style = style.to(device).unsqueeze(0)
        content = content.to(device).unsqueeze(0)
        
        with torch.no_grad():
            # [time]
            # start_time = time.time()
            output= network(content,style)[0]

            # [time]
            # end_time = time.time()
            # inference_time = end_time - start_time
            # inference_times.append(inference_time)
            # print(f"Inference time for {basename(content_path)} + {basename(style_path)}: {inference_time:.4f} seconds")
            # ]
        output = output.cpu()
                
        output_name = '{:s}/{:s}_stylized_{:s}{:s}'.format(
            output_path, splitext(basename(content_path))[0],
            splitext(basename(style_path))[0], save_ext
        )
 
        save_image(output, output_name)

# [time] 
if inference_times:
    avg_inference_time = sum(inference_times) / len(inference_times)
    total_inference_time = sum(inference_times)
    
    print("\n" + "="*50)
    print("INFERENCE TIME STATISTICS")
    print("="*50)
    print(f"Total image pairs processed: {len(inference_times)}")
    print(f"Total inference time: {total_inference_time:.4f} seconds")
    print(f"Average inference time: {avg_inference_time:.4f} seconds")
    print(f"Min inference time: {min(inference_times):.4f} seconds")
    print(f"Max inference time: {max(inference_times):.4f} seconds")
