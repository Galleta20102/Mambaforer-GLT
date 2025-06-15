Clone the code by :
```
$ git clone https://github.com/Galleta20102/Mambaforer-GLT.git
```

Env:
```
$ conda env create -f MambaformerGLT-env.yaml 
```
It may takes a while. . .
```
$ conda activate MambaformerGLT-env
```

Train
```
$ python train.py --style_dir ../styleTransfer/datasets/wikiart/ --content_dir ../styleTransfer/datasets/coco2014/images/ --save_dir experiments/ --batch_size 4
```

Test
```
$ python test.py  --content_dir ../styleTransfer/datasets/famouse_pain
tings/cnt_all/ --style_dir ../styleTransfer/datasets/famouse_paintings/sty   --output out_tmp --decoder_path experiments/tmp/decoder_iter_160000.pth --Trans_path experiments/tmp/transformer_iter_160000.pth --embedding_path experiments/tmp/embedding_iter_160000.pth
```