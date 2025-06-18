# Mambaformer-GLT
Mambaformer-GL: Global-Local Transfer Framework with Mamba and Transformer 
Decoder for Detail-preserving Style Transfer<br>

Zi-Han Hong<br>
<!---
[![Paper](https://img.shields.io/badge/Paper-link-orange.svg "PAKDD 2023 paper")](https://link.springer.com/book/10.1007/978-3-031-33374-3)
[![arXiv](https://img.shields.io/badge/arXiv-pdf-yellow.svg "arXiv paper link")](https://arxiv.org/abs/2305.08750)
-->
> National Taiwan University of Science and Technology (NTUST)<br>
> Department of Computer Science and Information Engineering (CSIE)

This paper is proposed on style transfer task and based on StyTR^2([Github](https://github.com/diyiiyiii/StyTR-2)).<br>
Our method achieves more stable stylization effects with excellent preservation in detail, particularly in retaining key features such as facial characteristics so that avoids the Uncanny Valley Effect while requiring lower computational resources.<br>

## Result Presentation
![Result presentation of Mamabaformer-GLT](<figure/results_presentation.png>)

## Architecture
![Mambaformer-GLT Architecture](<figure/architecture.png>)

## Environment Setup:
### Clone repository by :
```
$ git clone https://github.com/Galleta20102/Mambaforer-GLT.git
```
### Conda environment
- Create a conda environment by `env.yml` and activate it:
    ```
    $ conda env create -f env.yml
    $ conda activate MambaformerGLT
    ```
- Then install PyTorch (depends on your CUDA) :
    ```
    $ pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
    ```
    Please make sure you have up-to-date NVIDIA drivers supporting CUDA 11.3 at least.
    > [!WARNING] The ERROR about `causal-conv1d`
    > If you get an error msg like: you don't have `nvcc`
    > You can install it by [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)
    > or just use the command directly :
    > ```
    > $ sudo apt install nvidia-cuda-toolkit
    > ```

## Getting Dataset
Download style and content datasets then put into the folder `/datasets` .<br>
Style image dataset is WikiArt collected from [WIKIART](https://www.wikiart.org/) - [download](https://www.kaggle.com/datasets/steubk/wikiart)<br>
Content image dataset is COCO2014 - [download](https://cocodataset.org/#download)<br><br>

Directory structure may be like:
```
Mambaformer-GLT/

|-datasets    # all dataset
    |-coco2014
    |-wikiart

|-eval # code for evaluation
    |- calc_params.py
    |- copy_input.py
    |- eval_artfid.py
    |- ...
    
|-figure    # some figures for document

|-modles    # Mambaformer-GLT model code
    |-Mambaformer.py
    |-moMambaformerGLT.py
    |-Vit_helper.py
    
|- utils
|- .gitignore
|- env.yaml
|- ...
```
## Testing
## Training


```
$ python train.py --style_dir ../styleTransfer/datasets/wikiart/ --content_dir ../styleTransfer/datasets/coco2014/images/ --save_dir experiments/ --batch_size 4
```

Test
```
$ python test.py  --content_dir ../styleTransfer/dataset
s/famouse_paintings/cnt_all/ --style_dir ../styleTransfer/datasets/famouse_paintings/sty   --output out_tmp --decoder_path experiments/tmp/decoder_iter_160000.pth --mbfr_path experiments/tmp/mambaformer_iter_160000.pth --embedding_path experiments/tmp/embedding_iter_160000.pth
```