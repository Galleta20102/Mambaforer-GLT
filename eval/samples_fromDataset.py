import os
import random
import shutil

# 支持的圖片格式
image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')

def dir2dir(source_dir, target_dir, sample_ratio=0, sample_size=0):

    image_files = [f for f in os.listdir(source_dir) 
                        if os.path.isfile(os.path.join(source_dir, f)) 
                        and f.lower().endswith(image_extensions)]
    # 計算要採樣的圖片數量
    if not sample_size :
        sample_size = int(len(image_files) * sample_ratio)

    # 隨機選擇圖片
    sampled_images = random.sample(image_files, sample_size)

    print(f"Copying the {sample_size} images of folder {source_dir}...")
    for image in sampled_images:
        src = os.path.join(source_dir, image)
        dst = os.path.join(target_dir, image)
        shutil.copy2(src, dst)

def sub2dir(source_dir, target_dir, sample_ratio=0, sample_size=0):
    # Get all images
    images_lst = []
    for subdir, _, files in os.walk(source_dir):
        subdir_name = subdir.split("/")[-1]
        # 過濾出圖片文件
        image_files = [os.path.join(subdir_name, f) for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        images_lst += image_files
             
    # 計算要採樣的圖片數量
    if not sample_size :
        sample_size = int(len(image_files) * sample_ratio)
    
    # 隨機選擇圖片
    sampled_images = random.sample(images_lst, sample_size)
    print(f"Copying the {sample_size} images of folder {source_dir}...")
    for image in sampled_images:
        src = os.path.join(source_dir, image)
        dst = os.path.join(target_dir, image.split('/')[-1])
        shutil.copy2(src, dst)
    

def sub2sub(source_dir, target_dir, sample_ratio=0, sample_size=0):
    for subdir, _, files in os.walk(source_dir):
        print(f"Processing he folder {subdir}...")
        # 獲取相對路徑
        rel_path = os.path.relpath(subdir, source_dir)
        # 創建目標子目錄
        target_subdir = os.path.join(target_dir, rel_path)
        if not os.path.exists(target_subdir):
            os.makedirs(target_subdir)

        # 過濾出圖片文件
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        
        # 計算要採樣的圖片數量
        if not sample_size :
            sample_size = int(len(image_files) * sample_ratio)
        
        # 隨機選擇圖片
        sampled_images = random.sample(image_files, sample_size)
        
        # 複製選中的圖片到目標目錄
        for image in sampled_images:
            src = os.path.join(subdir, image)
            print(src)
            dst = os.path.join(target_subdir, image)
            shutil.copy2(src, dst)

def sample_images(source_dir, target_dir, sample_ratio=0, sample_size=0, tgt_type=''):

    if not os.path.exists(source_dir):
        print(f"[Error]：source directory '{source_dir}' is not exit.")
        return

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    match tgt_type :
        case 'dir2dir':
            dir2dir(source_dir, target_dir, sample_ratio, sample_size)
        case 'sub2dir':
            sub2dir(source_dir, target_dir, sample_ratio, sample_size)
        case 'sub2sub':
            sub2sub(source_dir, target_dir, sample_ratio, sample_size)
        case _:
            print("[ERROR] 'tgt_type' must be 'dir2dir', 'sub2dir'or 'sub2sub'.")
    
    if not sample_size :
        print(f"Sample complete. The {sample_ratio*100}% images have been copied to {target_dir}.")
    else :
        print(f"Sample complete. The {sample_size} images have been copied to {target_dir}.")


r''' __main__ '''
source_directory = "../datasets/wikiart"#coco2014/images/train2014" # TODO: the dataset will be sampled
target_directory = "../datasets/eval_40x80/sty" # TODO: the sampled result dataser folder
sample_ratio = 0 # TODO: tha sample percent to the dataset
sample_size = 80 # TODO: the sepcific sample num to the dataset (first priority)
tgt_type = 'sub2dir' # TODO: 'tgt_type' must be 'dir2dir', 'sub2dir'or 'sub2sub'.

# TODO: sample_size or sample_ratio
sample_images(source_directory, target_directory, 
                sample_size=sample_size,
                tgt_type=tgt_type)