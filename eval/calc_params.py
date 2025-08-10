import os
import sys
import torch
import torch.nn as nn
import argparse
from collections import OrderedDict
# sys.path.append(os.path.dirname(__file__))
# sys.path.append('..')
sys.path.insert(0, '.')
import models.Mambaformer as Mambaformer
import models.MambaformerGLT as MambaformerGLT


parser = argparse.ArgumentParser()
parser.add_argument('--vgg', type=str, default='./models/pretrained/vgg_normalised.pth')
parser.add_argument('--decoder_path', type=str, default='./models/pretrained/decoder_iter_160000.pth')
parser.add_argument('--mbfr_path', type=str, default='./models/pretrained/transformer_iter_160000.pth')
parser.add_argument('--embedding_path', type=str, default='./models/pretrained/embedding_iter_160000.pth')
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                    help="Type of positional embedding to use on top of the image features")
parser.add_argument('--hidden_dim', default=512, type=int,
                    help="Size of the embeddings (dimension of the transformer)")
# for Mamba-ST
parser.add_argument('--d_state', type=int, default=16)
parser.add_argument('--use_pos_embed', action='store_true') 
parser.add_argument('--rnd_style', action='store_true') 
parser.add_argument('--img_size', type=int, default=256)
                
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# load model
vgg = MambaformerGLT.vgg
vgg.load_state_dict(torch.load(args.vgg, map_location=device))
vgg = nn.Sequential(*list(vgg.children())[:44])

decoder = MambaformerGLT.decoder

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "my_modify"))

mbfr = Mambaformer.mambaformer()

embedding = MambaformerGLT.PatchEmbed()

# load pre-trained weight
new_state_dict = OrderedDict()
state_dict = torch.load(args.decoder_path, map_location=device)
for k, v in state_dict.items():
    namekey = k
    new_state_dict[namekey] = v
decoder.load_state_dict(new_state_dict)

new_state_dict = OrderedDict()
state_dict = torch.load(args.mbfr_path, map_location=device)
for k, v in state_dict.items():
    namekey = k
    new_state_dict[namekey] = v
mbfr.load_state_dict(new_state_dict)

new_state_dict = OrderedDict()
state_dict = torch.load(args.embedding_path, map_location=device)
for k, v in state_dict.items():
    namekey = k
    new_state_dict[namekey] = v
embedding.load_state_dict(new_state_dict)

# construct network
network = MambaformerGLT.mambaformerGLT(vgg, decoder, embedding, mbfr, args)

# calculate # of params
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_all_parameters(model):
    return sum(p.numel() for p in model.parameters())

# calculate each structure
vgg_trainable_params = count_parameters(vgg)
vgg_total_params = count_all_parameters(vgg)
decoder_params = count_parameters(decoder)
mambaformer_params = count_parameters(mbfr)
embedding_params = count_parameters(embedding)

# calculate entire model
trainable_params = count_parameters(network)
total_params = count_all_parameters(network)


print("\n--- # of params: ---")
print(f"VGG Encoder trainable : {vgg_trainable_params:,}") 
print(f"VGG Encoder total : {vgg_total_params:,}")
print(f"Decoder : {decoder_params:,}")
print(f"Mambaformer : {mambaformer_params:,}")
print(f"Embedding : {embedding_params:,}")
print("\n--- MambaformerGLT TOTAL ---")
print(f"total params: {total_params:,}")