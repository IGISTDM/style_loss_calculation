from LossCal import LossCal

from PIL import Image
from torchvision.transforms import functional as TF
import os
import matplotlib.pyplot as plt
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


ori_image = "ori_image"
ori_path = os.listdir('ori_image')

after = 'after_image'
after_path = os.listdir('after_image')

# print(ori_path)
output_folder = 'output'
os.makedirs(output_folder, exist_ok=True)

for style in ori_path:
    print(style)
    plt_clip, plt_vgg = plt.figure(), plt.figure()
    plt_clip_ax, plt_vgg_ax = plt_clip.add_subplot(111), plt_vgg.add_subplot(111)



    for _ in after_path:
        clip = []
        # clip_text = []
        vgg = []
        pic = os.listdir(after + '/' + _ + '/' + style)
        for p in pic:
            print(p)
            cal = LossCal()
            ori = Image.open(ori_image + '/' + style + '/' + p).convert("RGB")
            ori_ts = (TF.to_tensor(ori)).unsqueeze(0).mul(2).sub(1).to(cal.device)
            target = Image.open(after + '/' + _ + '/' + style + '/' + p).convert("RGB")
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
        
        plt_clip_ax.plot(clip, label=_)
        # plt_clip_ax.plot(clip_text, label=_)
        plt_vgg_ax.plot(vgg, label=_)

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
