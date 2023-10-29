from LossCal import LossCal
from PIL import Image
from torchvision.transforms import functional as TF
import os
import matplotlib.pyplot as plt
import json
from path import path_join
from constants import output_folder

def get_image_style(image_name:str):
    parts = image_name.split("-")
    return parts[-1]

style_folder_name = "styles"
style_folder = os.listdir(style_folder_name)

results = "results"
results_folder = os.listdir(results)

content_prefix = "scene-"

os.makedirs(output_folder, exist_ok=True)

for style in style_folder:
    print(style)

    for method in results_folder:
        clip = []
        vgg = []
        pic = os.listdir(path_join(results, method, content_prefix+style))
        for p in pic:
            print(p)
            cal = LossCal()
            style_image_name = get_image_style(p)
            ori = Image.open(path_join(style_folder_name, style, style_image_name)).convert("RGB")
            ori_ts = (TF.to_tensor(ori)).unsqueeze(0).mul(2).sub(1).to(cal.device)
            target = Image.open(path_join(results, method, content_prefix+style, p)).convert("RGB")
            target_ts = (TF.to_tensor(target)).unsqueeze(0).mul(2).sub(1).to(cal.device)
            clip.append(float(cal.clip_global_loss_image(ori_ts,target_ts)))
            vgg.append(float(cal.vgg_loss_feature_gram(ori_ts,target_ts)))
            ori.close()
            target.close()        
        
        print(clip)

    json_file_path = f"{output_folder}/{method}"
    os.makedirs(json_file_path, exist_ok=True)
    with open(f"{json_file_path}/{style}-content.json", 'w') as json_file:
        json.dump(clip, json_file)