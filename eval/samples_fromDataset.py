import os
import random
import shutil
import argparse

image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')

def dir2dir(source_dir, target_dir, spl_ratio=0, spl_num=0):

    image_files = [f for f in os.listdir(source_dir) 
                        if os.path.isfile(os.path.join(source_dir, f)) 
                        and f.lower().endswith(image_extensions)]
    # total num of selected images
    if not spl_num :
        spl_num = int(len(image_files) * spl_ratio)

    # randomly select
    sampled_images = random.sample(image_files, spl_num)

    print(f"Copying the {spl_num} images of folder {source_dir}...")
    for image in sampled_images:
        src = os.path.join(source_dir, image)
        dst = os.path.join(target_dir, image)
        shutil.copy2(src, dst)

def sub2dir(source_dir, target_dir, spl_ratio=0, spl_num=0):
    # Get all images
    images_lst = []
    for subdir, _, files in os.walk(source_dir):
        subdir_name = subdir.split("/")[-1]
        image_files = [os.path.join(subdir_name, f) for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        images_lst += image_files
             
    # total num of selected images
    if not spl_num :
        spl_num = int(len(image_files) * spl_ratio)
    
    # randomly select
    sampled_images = random.sample(images_lst, spl_num)
    print(f"Copying the {spl_num} images of folder {source_dir}...")
    for image in sampled_images:
        src = os.path.join(source_dir, image)
        dst = os.path.join(target_dir, image.split('/')[-1])
        shutil.copy2(src, dst)
    

def sub2sub(source_dir, target_dir, spl_ratio=0, spl_num=0):
    for subdir, _, files in os.walk(source_dir):
        print(f"Processing he folder {subdir}...")
        # get path
        rel_path = os.path.relpath(subdir, source_dir)
        # create directory
        target_subdir = os.path.join(target_dir, rel_path)
        if not os.path.exists(target_subdir):
            os.makedirs(target_subdir)

        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        
        # total num of selected images
        if not spl_num :
            spl_num = int(len(image_files) * spl_ratio)
        
        # randomly select
        sampled_images = random.sample(image_files, spl_num)
        print(f"Copying the {spl_num} images of folder {source_dir}...")
        for image in sampled_images:
            src = os.path.join(subdir, image)
            print(src)
            dst = os.path.join(target_subdir, image)
            shutil.copy2(src, dst)

def sample_images(source_dir, target_dir, spl_ratio=0, spl_num=0, tgt_type=''):

    if not os.path.exists(source_dir):
        print(f"[Error]：source directory '{source_dir}' is not exit.")
        return

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    match tgt_type :
        case 'dir2dir':
            dir2dir(source_dir, target_dir, spl_ratio, spl_num)
        case 'sub2dir':
            sub2dir(source_dir, target_dir, spl_ratio, spl_num)
        case 'sub2sub':
            sub2sub(source_dir, target_dir, spl_ratio, spl_num)
        case _:
            print("[ERROR] 'tgt_type' must be 'dir2dir', 'sub2dir'or 'sub2sub'.")
    
    if not spl_num :
        print(f"Sample complete. The {spl_ratio*100}% images have been copied to {target_dir}.")
    else :
        print(f"Sample complete. The {spl_num} images have been copied to {target_dir}.")


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--src_dir', type=str,
                    help='File path to the source directory',
                    default='./datasets/coco2014/images/train2014') # ./datasets/wikiart
parser.add_argument('--tgt_dir', type=str,
                    help='File path to the target directory',
                    default='./datasets/eval/cnt_img') # ./datasets/eval/sty_img
parser.add_argument('--spl_num', type=int,
                    help='the sepcific sample num to the dataset',
                    default='40') # 80
parser.add_argument('--spl_ratio', type=int,
                    help='tha sample percent (%) to the dataset',
                    default='10') # 80
parser.add_argument('--tgt_type', type=str,
                    help='Type (All images are in directory/subdirectory) of source and destination',
                    choices=['sub2sub', 'sub2dir', 'dir2dir'],
                    default='dir2dir')
args = parser.parse_args()

# TODO: spl_num or spl_ratio
sample_images(args.src_dir, args.tgt_dir, 
                spl_num=args.spl_num,
                tgt_type=args.tgt_type)