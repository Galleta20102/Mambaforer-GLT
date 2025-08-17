# Mambaformer-GLT
Mambaformer-GL: Global-Local Transfer Framework with Mamba and Transformer 
Decoder for Detail-preserving Style Transfer<br>

 Authors : [洪子涵 Zi-Han Hong](https://www.linkedin.com/in/galleta-hong),　[吳怡樂 Yi-Leh Wu](http://faculty.csie.ntust.edu.tw/~ywu/)<br>

> National Taiwan University of Science and Technology (NTUST)<br>
> Department of Computer Science and Information Engineering (CSIE)

<!---
[![Paper](https://img.shields.io/badge/Paper-link-orange.svg "PAKDD 2023 paper")](https://link.springer.com/book/10.1007/978-3-031-33374-3)
[![arXiv](https://img.shields.io/badge/arXiv-pdf-yellow.svg "arXiv paper link")](https://arxiv.org/abs/2305.08750)
-->

This paper is proposed on style transfer task and based on [StyTR<sup>2</sup>](https://github.com/diyiiyiii/StyTR-2).<br>
Our method achieves more stable stylization effects with excellent preservation in detail, particularly in retaining key features such as facial characteristics so that avoids the Uncanny Valley Effect while requiring lower computational resources.<br>

## Result Presentation
![Result presentation of Mamabaformer-GLT](<figure/results_prsnt.png>)

## Architecture
![Mambaformer-GLT Architecture](<figure/architecture.png>)

## Environment Setup
### Clone repository by :
```
git clone https://github.com/Galleta20102/Mambaforer-GLT.git
```
### Conda environment
- Create a conda environment by `env.yaml` and activate it:
    ```
    conda env create -f env.yaml
    conda activate MambaformerGLT
    ```
> [!WARNING]
> If you get an error msg like `nvcc was not found` while installing `causal-conv1d` ;<br>
> ![Sample Error Msg while create env by ymal](<figure/error_causal-conv1d.png>)
> You can install it at [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive), or just use the command directly (the version need to ***>= 11.6***) :<br>
> ```
> sudo apt install nvidia-cuda-toolkit
>
> # Checking for Cuda compilation tools version >= 11.6
> nvcc -V
> ```
> Then remove the failded environment by following command, and [create environment](#conda-environment) again :
> ```
> conda remove --name MambaformerGLT --all
> ```
- Then install PyTorch/CUDA dependent packages (depends on your CUDA) :
    ```
    pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
    pip install mamba-ssm==1.2.0
    ```
    Please make sure you have up-to-date NVIDIA drivers supporting CUDA 11.3 at least.
- Don't forget to deactivate the virtual environment after you perform all experiments :
    ```
    deactivate
    ```

## Getting Dataset
Download style and content datasets then put into the folder `datasets/` .<br>
Style image dataset is WikiArt collected from [WIKIART](https://www.wikiart.org/) - [download](https://www.kaggle.com/datasets/steubk/wikiart)<br>
Content image dataset is COCO2014 - [download](https://cocodataset.org/#download)<br><br>

Directory structure may be like:
```
Mambaformer-GLT/
├── datasets/    # PLEASE PUT DATASETS IN HERE
│   ├── coco2014/　　# CONTENT IMAGE DATASET
│   └── wikiart/　　# STYLE IMAGE DATASET
├── eval/    # code for evaluation
│   ├── calc_params.py
│   ├── copy_input.py
│   ├── eval_artfid.py
│   └── ...
├── figure/    # some figures for document
├── models/    # Mambaformer-GLT model code
│   ├── pretrained/
│   ├── Mambaformer.py
│   ├── MambaformerGLT.py
│   └── Vit_helper.py
├── utils/
├── .gitignore
├── env.yaml
├── ...
└── train.py
```
## Testing
You can download pretrained models, and put them into `models/pretrained/` :
- [vgg_normalised.pth](https://drive.google.com/file/d/1Zk4atdvgMJCQhJIhIHS0rNBNzaix-mQZ/view?usp=sharing) 
- [embedding_iter_160000](https://drive.google.com/file/d/1OZQH8Dg6CG-B2V7d0yFgDVMA8bHB6BM1/view?usp=sharing)
- [decoder_iter_160000](https://drive.google.com/file/d/1HmlJHQ11-h-9iYm1TY0I1c-42cbSxCAp/view?usp=sharing)
- [mambaformer_iter_160000](https://drive.google.com/file/d/16FeHGZqg8lTqPNbJhZTZGudZiD9n5ZSy/view?usp=sharing)<br>

To transfer images by content and style images, use command to test our model :
```
$ python test.py \
  --content_dir <dir_path/of/cnt_img> \
  --style_dir <dir_path/of/sty_img> \
  --output <dir_path/of/output_img> \
  --decoder_path <path/of/decoder.pth> \
  --mbfr_path <path/of/mambaformer.pth> \
  --embedding_path experiments/tmp/<path/of/embedding.pth>
```
> [!NOTE]
> Please replace placeholder paths `<dir_path/of/...>` with actual directory/file paths.<br>
> Transfer with **n** content images and **m** style images, output will have **n x m** imgs.

#### Testing Parameter Description
- Input Parameters
    - `--content_dir` : Directory path containing your content images
    - `--style_dir` : Directory path containing your style images
- Output Parameters
    - `--output` : Directory path for output images
- Model Paths
    - `--decoder_path` : Path to decoder model weights file (decoder_iter_xxxxx.pth)
    - `--mbfr_path` : Path to MambaFormer model weights file (mambaformer_iter_xxxxx.pth)
    - `--embedding_path`: Path to embedding layer weights file (embedding_iter_xxxxx.pth)

### Usage Example for Testing
```
python test.py --content_dir datasets/test/cnt_img --style_dir datasets/test/sty_img --output datasets/test/output  --decoder_path models/pretrained/decoder_iter_160000.pth --mbfr_path models/pretrained/mambaformer_iter_160000.pth --embedding_path models/pretrained/embedding_iter_160000.pth
```

## Training
If you want to train your own Mamabformer-GLT, use training command :
```
$ python train.py \
 --content_dir <dir_path/of/cnt_dataset> \
 --style_dir <dir_path/of/sty_dataset> \
 --save_dir <dir_path/of/model_pth> \
 --batch_size <batch_size>
```
#### Training Parameter Description
- Dataset Parameters
    - `--content_dir` : Directory path of content image dataset
    - `--style_dir` : Directory path of your style image dataset
- Output Parameters
    - `--save_dir` : Directory path to save trained model weights (.pth files)
- Training Parameters
    - `--batch_size` : Batch size for training (e.g., 8, 16, 32)<br>
        Use appropriate batch size based on your GPU memory (common values: 8, 16, 32)
        
### Usage Example for Training
```
python train.py  --style_dir datasets/wikiart --content_dir datasets/coco2014/images  --save_dir models/experiments  --batch_size 4
```
> [!TIP]
> Other common parameters you can use for training:
> - `--resume_iter`: Iteration checkpoint to resume training from, specify in increments of 10,000 (e.g., 20000, 50000, 100000)
> - `--max_iter` : Maximum of training iterator (default=160k)
> - `--hidden_dim` : Size of the embeddings, dimension of the mambaformer (default=512)
> - `--log_dir` : Directory to save the log (default=./logs)

## Evaluation
For Mamabformer-GLT evaluation, we need to copy content and style images by all combination of pairs. Then get the loss and all metrics.

### Preparation 
1. **Copy Images** <br><br>
    First step is selecting images from both content and style datasets.
    Here we select 40 images from COCO2014 and 80 from WikiArt :
    ```
    # COCO2014 for content
    python eval/samples_fromDataset.py  --src_dir datasets/coco2014/images/train2014  --tgt_dir datasets/eval/cnt_img  --spl_num 40  --tgt_type dir2dir

    # WikiArt for style
    python eval/samples_fromDataset.py  --src_dir datasets/wikiart  --tgt_dir datasets/eval/sty_img  --spl_num 80  --tgt_type sub2dir
    ```
2. **Generate evaluation images**<br><br>
    Then we generate images (nxm) from both `eval/cnt_img` and `eval/sty_img`.<br>
    > Here is 40 x 80 = 3200 images in `cnt_img_eval` and `sty_img_eval` respectively.
    ```
    python eval/copy_inputs.py --cnt datasets/eval/cnt_img --sty datasets/eval/sty_img
    ```
3. **Generate Stylized Images from Model**<br><br>
    ```
    python eval/eval_loss.py --content_dir datasets/eval/cnt_img  --style_dir datasets/eval/sty_img/  --decoder_path models/pretrained/decoder_iter_160000.pth   --mbfr_path models/pretrained/mambaformer_iter_160000.pth   --embedding_path models/pretrained/embedding_iter_160000.pth --output datasets/eval/Mambaformer-GLT/ --img_size 256 --seed 123456
    ```
    This step will create the target folder `datasets/eval/Mambaformer-GLT/` containing all transfered images.<br>
    > Set `--img_size 256` just for quickly evaluation.<br>
    > In testing, model outputs 512x512 resolution images. 

Now, the evaluation data and output images are all prepared!

### # of Params
Use command to calaulate the number of model parameters : 
```
$ python eval/calc_params.py --embedding_path <path/of/embedding.pth> --mbfr_path <path/of/mambaformer.pth> --decoder_path <path/of/decoder.pth>
```
Usage Example :
```
python eval/calc_params.py --decoder_path models/pretrained/decoder_iter_160000.pth --mbfr_path ./models/pretrained/mambaformer_iter_160000.pth --embedding_path ./models/pretrained/embedding_iter_160000.pth
```
> The output will be like:
> ```
> --- # of params: ---
> VGG Encoder trainable : xxx
> VGG Encoder total : xxx
> Decoder : xxx
> Mambaformer : xxx
> Embedding : xxx
> 
> --- MambaformerGLT TOTAL ---
> total params: xx,xxx,xxx
> ```

###  Quantitative Evaluation
Use command for evaluation metrics :
```
$ python eval/eval_artfid.py --cnt datasets/eval/cnt_img_eval/ --sty datasets/eval/sty_img_eval/ --tar datasets/eval/Mambaformer-GLT/
```
The `--tar` is a folder that you generated stylized images from model (Step 3 in Evaluation Preparation).<br>
After a moment, you will see the output like this :
```
ArtFID: xx.xxxx FID: xx.xxxx LPIPS: x.xxxx LPIPS_gray: x.xxxx
CFSD: x.xxxx
```
