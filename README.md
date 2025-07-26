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
![Result presentation of Mamabaformer-GLT](<figure/results_presentation.png>)

## Architecture
![Mambaformer-GLT Architecture](<figure/architecture.png>)

## Environment Setup
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
    
    > **The ERROR about `causal-conv1d`:**<br>
    > If you get an error msg like `you don't have nvcc`<br>
    > You can install it by [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive), or just use the command directly :<br>
    > ```
    > $ sudo apt install nvidia-cuda-toolkit
    > ```
- After you have performed all experiments, don't forget to close the virtual environment using :
    ```
    $ deactivate
    ```

## Getting Dataset
Download style and content datasets then put into the folder `/datasets` .<br>
Style image dataset is WikiArt collected from [WIKIART](https://www.wikiart.org/) - [download](https://www.kaggle.com/datasets/steubk/wikiart)<br>
Content image dataset is COCO2014 - [download](https://cocodataset.org/#download)<br><br>

Directory structure may be like:
```
Mambaformer-GLT/
├── datasets/    # PLEASE PUT DATASETS IN HERE
│   ├── coco2014/
│   └── wikiart/
├── eval/    # code for evaluation
│   ├── calc_params.py
│   ├── copy_input.py
│   ├── eval_artfid.py
│   └── ...
├── figure/    # some figures for document
├── models/    # Mambaformer-GLT model code
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
<!--Pretrained models: <br>-->
#### To transfer images by content and style images, use command to test our model :
```
$ python test.py \
  --content_dir <dir_path/of/cnt_img> \
  --style_dir <dir_path/of/sty_img> \
  --output <dir_path/of/output_img> \
  --decoder_path <path/of/decoder_iter_160000.pth> \
  --mbfr_path <path/of/mambaformer_iter_160000.pth> \
  --embedding_path experiments/tmp/<path/of/embedding_iter_160000.pth>
```
> [!NOTE]
> Replace placeholder paths `<dir_path/of/...>` with actual directory/file paths.<br>
> Transfer with **n** content images and **m** style images, output will have **n x m** imgs.

#### Parameter Description
- Input Parameters
    - `--content_dir` : Directory path containing your content images
    - `--style_dir` : Directory path containing your style images
- Output Parameters
    - `--output` : Directory path for output images
- Model Paths
    - `--decoder_path` : Path to decoder model weights file (decoder_iter_160000.pth)
    - `--mbfr_path` : Path to MambaFormer model weights file (mambaformer_iter_160000.pth)
    - `--embedding_path`: Path to embedding layer weights file (embedding_iter_160000.pth)

- Usage Example<br>
    ```
    $ python test.py \
    --content_dir datasets/test/cnt_img \
    --style_dir datasets/test/sty_img \
    --output datasets/test/output \
    --decoder_path models/pretrained/decoder_iter_160000.pth \
    --mbfr_path models/pretrained/mambaformer_iter_160000.pth \
    --embedding_path models/pretrained/embedding_iter_160000.pth
    ```

## Training
```
$ python train.py \
 --content_dir <dir_path/of/cnt_dataset> \
 --style_dir <dir_path/of/sty_dataset> \
 --save_dir <dir_path/of/model_pth> 
 --batch_size <batch_size>
```
#### Parameter Description
- Dataset Parameters
    - `--content_dir` : Directory path of content image dataset
    - `--style_dir` : Directory path of your style image dataset
- Output Parameters
    - `--save_dir` : Directory path to save trained model weights (.pth files)
- Training Parameters
    - `--batch_size` : Batch size for training (e.g., 8, 16, 32)<br>
        Use appropriate batch size based on your GPU memory (common values: 8, 16, 32)
- Usage Example
    ```
    $ python train.py \
     --style_dir datasets/wikiart  \
     --content_dir datasets/coco2014/images \
     --save_dir models/experiments
     --batch_size 4
    ```
> [!NOTE]
> Other common parameters you can use for training:
> `--resume_iter`: Iteration checkpoint to resume training from, specify in increments of 10,000 (e.g., 20000, 50000, 100000)
> `--max_iter` : Maximum of training iterator (default=160k)
> `--hidden_dim` : Size of the embeddings, dimension of the mambaformer (default=512)
> `--log_dir` : Directory to save the log (default=./logs)

<!--## Evaluation
```
python eval/eval_loss_modify.py --model_name Mambaformer_stDecoder --content_dir ../datasets/eval/cnt/  --style_dir ../datas
ets/eval/sty/  --decoder_path models_modify/cape_mambaformer_Res_smth_struct_05_orgCAPE_stDecoder/decoder_iter_160000.pth   --Trans_path models_modify/cape_mambaformer_Res_smth_struct_05_orgCAPE_stDecoder/transformer_iter_160000.pth   --embedding_path models_modify/cape_mambaformer_Res_smth_struct_05_orgCAPE_stDecoder/embedding_iter_160000.pth --output ../datasets/eval/Mambaformer_stDecoder/ --seed 123456
```
```
$ python eval/eval_artfid.py --cnt ../datasets/eval/cnt_eval/ --sty ../datasets/eval/sty_eval/ --tar ../datasets/eval/styTr2_origin/
```
-->

