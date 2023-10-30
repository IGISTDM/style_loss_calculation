from losses import d_clip_loss, get_features
from torchvision import transforms, models
import torch
import clip
from augmentation import ImageAugmentations

class LossCal:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_augmentations = ImageAugmentations(
            224, 0.5, 0.5, 1.0, patch=False)
        self.patch_augmentations = ImageAugmentations(
            224, 0.5, 0.5, 1.0, patch=True)
        self.clip_model = (
            clip.load("ViT-B/16", device=self.device,
                      jit=False)[0].eval().requires_grad_(False)
        )
        self.clip_normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
        )
        self.vgg = models.vgg19(pretrained=True).features.to(
            self.device).eval().requires_grad_(False)
        # self.vgg = self.vgg.to(self.device)
        # self.vgg.eval().requires_grad_(False)
        self.vgg_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ).to(self.device)

    # calculate the loss between image and text
    def clip_global_loss_text(self, image, text):
        clip_loss = torch.tensor(0).to(self.device)
        augmented_in = self.image_augmentations(image, 1).add(1).div(2)
        clip_in = self.clip_normalize(augmented_in)
        x_image_embeds = self.clip_model.encode_image(clip_in).float()
        text = self.clip_model.encode_text(
            clip.tokenize(text).to(self.device)
        ).float()

        clip_loss = d_clip_loss(x_image_embeds, text).mean()
        return clip_loss

    # calculate clip loss between 2 images
    def clip_global_loss_image(self, image, style_image) -> float:
        clip_loss = torch.tensor(0).to(self.device)
        augmented_in = self.image_augmentations(image, 1).add(1).div(2)
        clip_in = self.clip_normalize(augmented_in)
        x_image_embeds = self.clip_model.encode_image(clip_in).float()

        augmented_style = self.image_augmentations(
            style_image, 1).add(1).div(2)
        clip_style = self.clip_normalize(augmented_style)
        y_image_embeds = self.clip_model.encode_image(clip_style).float()

        clip_loss = float(d_clip_loss(x_image_embeds, y_image_embeds).mean())
        return clip_loss

    # calculate vgg loss between 2 images
    def vgg_loss_feature_gram(self, x_in, y_in) -> float:
        content_features = get_features(self.vgg_normalize(x_in), self.vgg)
        target_features = get_features(self.vgg_normalize(y_in), self.vgg)
        loss = 0.0
        layers = {'0': 'conv1_1',
                  '2': 'conv1_2',
                  '5': 'conv2_1',
                  '7': 'conv2_2',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',
                  '28': 'conv5_1',
                  '31': 'conv5_2'
                  }
        for key in layers:
            target_gram = self.gram_matrix(target_features[layers[key]])
            content_gram = self.gram_matrix(content_features[layers[key]])
            loss += torch.mean((target_gram - content_gram) ** 2)
        return float(loss)

    def gram_matrix(self, features):
        batch_size, num_channels, height, width = features.size()
        features = features.view(batch_size * num_channels, height * width)
        gram = torch.mm(features, features.t())
        return gram.div(batch_size * num_channels * height * width)
