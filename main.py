from LossCal import LossCal

from PIL import Image
from torchvision.transforms import functional as TF
import os
import matplotlib.pyplot as plt
from path import path_join
# cal =  LossCal()
# image = Image.open("apple.jpg").convert("RGB")
# # image = image.resize((10, 10), Image.LANCZOS)
# image = (TF.to_tensor(image)).unsqueeze(0).mul(2).sub(1).to(cal.device)
# target = Image.open("imagenet3.JPEG").convert("RGB")
# # image = image.resize((10, 10), Image.LANCZOS)
# target = (TF.to_tensor(target)).unsqueeze(0).mul(2).sub(1).to(cal.device)

# # print(cal.vgg_loss_feature_gram(image,target))
# print(float(cal.clip_global_loss_image(image,target)))
# print(float(cal.clip_global_loss_text(image,"a red apple")))
def get_image_style(image_name:str):
    parts = image_name.split("-")
    return parts[-1]

style_folder_name = "styles"
style_folder = os.listdir(style_folder_name)

results = "results"
results_folder = os.listdir(results)

content_prefix = "scene-"

output_folder = 'analysis'
os.makedirs(output_folder, exist_ok=True)

for style in style_folder:
    print(style)
    plt_clip, plt_vgg = plt.figure(), plt.figure()
    plt_clip_ax, plt_vgg_ax = plt_clip.add_subplot(111), plt_vgg.add_subplot(111)


    for method in results_folder:
        clip = []
        # clip_text = []
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
            # clip_text.append(float(cal.clip_global_loss_text(ori,"a red apple")))
            vgg.append(float(cal.vgg_loss_feature_gram(ori_ts,target_ts)))
            ori.close()
            target.close()
           
            # matrix.append(float(cal.clip_global_loss_image(ori,target)))
            
            # cal = LossCal()
            # image = Image.open(ori + '/' + style).convert("RGB")
            # image = (TF.to_tensor(image)).unsqueeze(0).mul(2).sub(1).to(cal.device)
            # target = Image.open(after + '/' + _ + '/' + p).convert("RGB")
            # target = (TF.to_tensor(target)).unsqueeze(0).mul(2).sub(1).to(cal.device)
            # matrix.append(float(cal.clip_global_loss_image(image,target)))
        
        plt_clip_ax.plot(clip, label = method)
        # plt_clip_ax.plot(clip_text, label=_)
        plt_vgg_ax.plot(vgg, label = method)

        plt_clip_ax.set_title(f"{style} clip")
        plt_clip_ax.set_xlabel("number of images")
        plt_clip_ax.set_ylabel("loss")

        plt_vgg_ax.set_title(f"{style} vgg")
        plt_vgg_ax.set_xlabel("number of images")
        plt_vgg_ax.set_ylabel("loss")


    plt_clip.savefig(f"{output_folder}/{style}_clip.png")
    plt.close(plt_clip)
    plt_vgg.savefig(f"{output_folder}/{style}_vgg.png")
    plt.close(plt_vgg)
        # image = Image.open(ori + '/' + style).convert("RGB")
        # image = (TF.to_tensor(image)).unsqueeze(0).mul(2).sub(1).to(cal.device)
        # target = Image.open(after + '/' + _).convert("RGB")
        # target = (TF.to_tensor(target)).unsqueeze(0).mul(2).sub(1).to(cal.device)
        # matrix.append(float(cal.clip_global_loss_image(image,target)))
