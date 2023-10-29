from LossCal import LossCal
from PIL import Image
from torchvision.transforms import functional as TF
import os
import json
from path import path_join
from constants import output_folder

def get_image_style(image_name: str):
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
        clip_loss_list = []
        vgg_loss_list = []
        method_subsets = os.listdir(
            path_join(results, method, content_prefix+style))
        for image_name in method_subsets:
            print(image_name)
            cal = LossCal()

            # style image
            style_image_name = get_image_style(image_name)
            style_image = Image.open(path_join(style_folder_name, style,
                             style_image_name)).convert("RGB")
            style_image_tensor = (TF.to_tensor(style_image)).unsqueeze(
                0).mul(2).sub(1).to(cal.device)
            
            # result image
            result_image = Image.open(
                path_join(results, method, content_prefix+style, image_name)).convert("RGB")
            result_image_tensor = (TF.to_tensor(result_image)).unsqueeze(
                0).mul(2).sub(1).to(cal.device)

            # append loss to list
            clip_loss_list.append(cal.clip_global_loss_image(style_image_tensor, result_image_tensor))
            vgg_loss_list.append(cal.vgg_loss_feature_gram(style_image_tensor, result_image_tensor))

            # close images
            style_image.close()
            result_image.close()

    # save loss data to json file
    json_file_path = f"{output_folder}/{method}"
    os.makedirs(json_file_path, exist_ok=True)
    with open(f"{json_file_path}/{style}-content.json", 'w') as json_file:
        json.dump(clip_loss_list, json_file)
