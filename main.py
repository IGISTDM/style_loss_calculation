from LossCal import LossCal
from PIL import Image
from torchvision.transforms import functional as TF
import os
import json
from path import path_join
from constants import output_folder, style_list

def get_image_style(image_name: str):
    parts = image_name.split("-")
    return parts[-1]

style_folder_name = "styles"
style_folder = os.listdir(style_folder_name)

results = "results"
results_folder = os.listdir(results)

content_prefix = "scene-"

content_amount = 50
style_amount = 100
stylized_image_amount = content_amount * style_amount
stylized_image_amount = 5

os.makedirs(output_folder, exist_ok=True)

# calculate the longest style name length
max_style_string_length = 0
for style_string in style_list:
    style_string_length = len(style_string)
    if style_string_length > max_style_string_length:
        max_style_string_length = style_string_length

for style in style_folder:
    for method in results_folder:
        print(f"{method}:")
        clip_loss_list = []
        vgg_loss_list = []
        method_subsets = os.listdir(
            path_join(results, method, content_prefix+style))
        index = 0
        for image_name in method_subsets:
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

            # print process status
            index = index + 1
            formatted_style = '{:<{}s}'.format(style, max_style_string_length)
            percentage = index / stylized_image_amount * 100
            image_status = f"({index}/{stylized_image_amount})"
            print(f"\t{formatted_style}: {percentage}% {image_status}", end="\r")

    # save loss data to json file
    json_file_path = f"{output_folder}/{method}"
    os.makedirs(json_file_path, exist_ok=True)
    with open(f"{json_file_path}/{style}-content.json", 'w') as json_file:
        json.dump(clip_loss_list, json_file)
