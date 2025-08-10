import argparse
from pathlib import Path
import os
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import os
import sys
import random
from tqdm import tqdm
#from util.utils import load_pretrained
import shutil

# Get 2 layers abs path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
# Import Directly
from models import Mambaformer
from models import MambaformerGLT

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
parser.add_argument('--output_dir', type=str, default='',
                    help='Directory to save the output image(s)')
parser.add_argument('--seed', default=777, type=int,
                    help="Seed for reproducibility")
parser.add_argument('--model_name', type=str, default='mambaformer')
parser.add_argument('--img_size', type=int, default=256)

parser.add_argument('--vgg', type=str, default='./models/pretrained/vgg_normalised.pth')
parser.add_argument('--decoder_path', type=str, default='./models/pretrained/decoder_iter_160000.pth') 
parser.add_argument('--mbfr_path', type=str, default='./models/pretrained/mambaformer_iter_160000.pth')
parser.add_argument('--embedding_path', type=str, default='./models/pretrained/embedding_iter_160000.pth')


parser.add_argument('--style_interpolation_weights', type=str, default="")
parser.add_argument('--a', type=float, default=1.0)
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
parser.add_argument('--hidden_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the mambaformer)")
args = parser.parse_args()

def load_pretrained(args):
    #vgg = models_helper.vgg
    vgg = MambaformerGLT.vgg
    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:44])

    decoder = MambaformerGLT.decoder
    mbfr = Mambaformer.mambaformer()
    embedding = MambaformerGLT.PatchEmbed()

    decoder_path = args.decoder_path
    mbfr_path = args.mbfr_path
    embedding_path = args.embedding_path
        

    decoder.eval()
    mbfr.eval()
    vgg.eval()
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    state_dict = torch.load(decoder_path)
    for k, v in state_dict.items():
        namekey = k
        new_state_dict[namekey] = v
    decoder.load_state_dict(new_state_dict)

    new_state_dict = OrderedDict()
    state_dict = torch.load(mbfr_path)
    for k, v in state_dict.items():
        namekey = k
        new_state_dict[namekey] = v
    mbfr.load_state_dict(new_state_dict)

    new_state_dict = OrderedDict()
    state_dict = torch.load(embedding_path)
    for k, v in state_dict.items():
        namekey = k
        new_state_dict[namekey] = v
    embedding.load_state_dict(new_state_dict)

    with torch.no_grad():
        network = MambaformerGLT.mambaformerGLT(vgg,decoder,embedding,mbfr,args)
    
    print(f"Loaded Embedding checkpoints from {embedding_path}")
    print(f"Loaded Mamba checkpoints from {mbfr_path}")
    print(f"Loaded CNN decoder checkpoints from {decoder_path}")
    return network


def select_random_images(root_dir, num_images, save_dir=None):
    images = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.jpg') or filename.endswith('.png'):
                images.append(os.path.join(dirpath, filename))

    if len(images) < num_images:
        print("Warning: Number of images in folder is less than required.")

    selected_images = random.sample(images, min(num_images, len(images)))
    
    if save_dir is not None:
        # Ensure the destination directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Copy selected images to the destination directory
        for image in selected_images:
            shutil.copy(image, save_dir)
    return selected_images


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



print("Name: ", args.model_name)
# Advanced options
content_size=args.img_size
style_size=args.img_size
crop='store_true'
save_ext='.jpg'
output_dir=args.output_dir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.content:
    content_paths = [Path(args.content)]
else:
    content_dir = Path(args.content_dir)
    content_paths = [f for f in content_dir.glob('*')]

if args.style:
    style_paths = [Path(args.style)]    
else:
    style_dir = Path(args.style_dir)
    style_paths = [f for f in style_dir.glob('*')]

random.seed(args.seed) #TODO: replace to my code
num_images_to_select = 80
style_paths = select_random_images(args.style_dir, num_images_to_select)
num_images_to_select = 40
content_paths = select_random_images(args.content_dir, num_images_to_select)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

network = load_pretrained(args)
network.eval()
network.to(device)

content_tf = test_transform(content_size, crop)
style_tf = test_transform(style_size, crop)
content_loss = 0.0
style_loss = 0.0

print(f"Stylizing {len(content_paths)} content images with {len(style_paths)} style images . . .")

for content_path in tqdm(content_paths):
    for style_path in tqdm(style_paths):
        content = content_tf(Image.open(content_path).convert("RGB"))
        style = style_tf(Image.open(style_path).convert("RGB"))
    
        style = style.to(device).unsqueeze(0)
        content = content.to(device).unsqueeze(0)
        
        with torch.no_grad():
            output, loss_c, loss_s, _, _, _ = network(content,style) 
    
        content_loss += loss_c
        style_loss += loss_s
        
        output_name = '{:s}/{:s}_stylized_{:s}{:s}'.format(
            output_dir, splitext(basename(content_path))[0],
            splitext(basename(style_path))[0], save_ext
        )
        
        if args.output_dir != "":
            save_image(output, output_name)

print("Image size: ", args.img_size)
print(args.model_name)
print(f"Content loss total: {content_loss.item()} - Style loss total: {style_loss.item()}")
print(f"Content loss mean: {content_loss.item()/(len(content_paths)*len(style_paths))} - Style loss mean: {style_loss.item()/(len(content_paths)*len(style_paths))}")