import os
import sys
import torch
import torch.nn as nn
import argparse
from collections import OrderedDict
import models.transformer as transformer
import models.StyTR as StyTR


parser = argparse.ArgumentParser()
parser.add_argument('--vgg', type=str, default='./experiments/vgg_normalised.pth')
parser.add_argument('--decoder_path', type=str, default='experiments/decoder_iter_160000.pth')
parser.add_argument('--Trans_path', type=str, default='experiments/transformer_iter_160000.pth')
parser.add_argument('--embedding_path', type=str, default='experiments/embedding_iter_160000.pth')
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
vgg = StyTR.vgg
vgg.load_state_dict(torch.load(args.vgg, map_location=device))
vgg = nn.Sequential(*list(vgg.children())[:44])

decoder = StyTR.decoder
# m11215021
#Trans = transformer.Transformer() #origin
import my_modify.cape_mambaformer_Res_smth_struct_05_orgCAPE_m2 as cape_mambaformer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "my_modify"))
# import my_modify.VMambaformer as VMambaformer
#import my_modify.VSSMformer_pure as VSSMformer

# ====== mamba-st ======
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append("../MambaST/")
# import models.mamba as mambast_trans

Trans = cape_mambaformer.cape_mambaformer() # TODO: replace with your model
#

embedding = StyTR.PatchEmbed()

# load pre-trained weight
new_state_dict = OrderedDict()
state_dict = torch.load(args.decoder_path, map_location=device)
for k, v in state_dict.items():
    namekey = k
    new_state_dict[namekey] = v
decoder.load_state_dict(new_state_dict)

new_state_dict = OrderedDict()
state_dict = torch.load(args.Trans_path, map_location=device)
for k, v in state_dict.items():
    namekey = k
    new_state_dict[namekey] = v
Trans.load_state_dict(new_state_dict)

new_state_dict = OrderedDict()
state_dict = torch.load(args.embedding_path, map_location=device)
for k, v in state_dict.items():
    namekey = k
    new_state_dict[namekey] = v
embedding.load_state_dict(new_state_dict)

# construct network
network = StyTR.StyTrans(vgg, decoder, embedding, Trans, args)

# calculate # of params
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_all_parameters(model):
    return sum(p.numel() for p in model.parameters())

# calculate each structure
vgg_trainable_params = count_parameters(vgg)
vgg_total_params = count_all_parameters(vgg)
decoder_params = count_parameters(decoder)
transformer_params = count_parameters(Trans)
embedding_params = count_parameters(embedding)

# calculate entire model
trainable_params = count_parameters(network)
total_params = count_all_parameters(network)


print("\n--- # of params: ---")
print(f"VGG Encoder 可訓練參數量: {vgg_trainable_params:,}") 
print(f"VGG Encoder 總參數量: {vgg_total_params:,}")
print(f"Decoder 參數量: {decoder_params:,}")
print(f"Transformer 參數量: {transformer_params:,}")
print(f"Embedding 參數量: {embedding_params:,}")
print("\n--- 總計 ---")
# print(f"可訓練參數量: {trainable_params:,}")
print(f"total params: {total_params:,}")

# 打印在訓練過程中實際優化的參數量
# training_params = decoder_params + transformer_params + embedding_params
# print(f"\n訓練過程中優化的參數量: {training_params:,}")

# # 查看每個模塊的結構
# print("\n--- 模塊結構 ---")
# print("VGG Encoder:")
# print(vgg)
# print("\nDecoder:")
# print(decoder)
# print("\nTransformer:")
# print(Trans)
# print("\nEmbedding:")
# print(embedding)