import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GLCSAttention(nn.Module):
    def __init__(self, in_channels):
        super(GLCSAttention, self).__init__()
        self.conv1x1_local = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv1x1_global_1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv1x1_global_2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)
        self.conv7x7 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        local_channel_attention = self.sigmoid(
            self.conv1x1_local(F.adaptive_avg_pool2d(x, (1, 1)))
        )
        local_channel_attention = x * local_channel_attention
        global_channel_attention = F.adaptive_avg_pool2d(x, (1, 1))
        global_channel_attention = self.sigmoid(
            self.conv1x1_global_1(global_channel_attention)
            * self.conv1x1_global_2(global_channel_attention)
        )
        global_channel_attention = x * global_channel_attention
        local_spatial_attention = self.conv1x1_local(x)
        local_spatial_attention = (
            self.conv3x3(local_spatial_attention)
            + self.conv5x5(local_spatial_attention)
            + self.conv7x7(local_spatial_attention)
        )
        local_spatial_attention = self.sigmoid(local_spatial_attention)
        local_spatial_attention = x * local_spatial_attention
        global_spatial_attention = self.conv1x1_local(x)
        global_spatial_attention = self.sigmoid(
            global_spatial_attention * self.conv1x1_local(global_spatial_attention)
        )
        global_spatial_attention = x * global_spatial_attention
        final_attention = (
            0.5 * local_channel_attention + 0.5 * global_channel_attention
        ) * (0.5 * local_spatial_attention + 0.5 * global_spatial_attention)
        return final_attention


class AdaptivelyWeightedMultiScaleAttention(nn.Module):
    def __init__(self, scales):
        super(AdaptivelyWeightedMultiScaleAttention, self).__init__()
        self.scales = scales
        self.glcs_attention = nn.ModuleList([GLCSAttention(scale) for scale in scales])
        self.weights = nn.Parameter(torch.ones(len(scales)) * 0.25)

    def forward(self, features):
        weighted_features = []
        for i, feature in enumerate(features):
            attention = self.glcs_attention[i](feature)
            weight = self.weights[i]
            weighted_feature = F.adaptive_max_pool2d(attention, (1, 1)) * weight
            weighted_features.append(weighted_feature)
        return torch.cat(weighted_features, dim=1)


class FaceNeSt(nn.Module):
    def __init__(self, num_classes=2):
        super(FaceNeSt, self).__init__()
        self.initial_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.resnet_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(1024),
                    nn.ReLU(inplace=True),
                ),
            ]
        )
        self.adaptive_attention = AdaptivelyWeightedMultiScaleAttention(
            [64, 256, 512, 1024]
        )
        self.conv1x1 = nn.Conv2d(sum([64, 256, 512, 1024]), 512, kernel_size=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.initial_layers(x)
        features = [x]
        for block in self.resnet_blocks:
            x = block(x)
            features.append(x)
        x = self.adaptive_attention(features)
        x = self.conv1x1(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# Load the model
def load_model(model_path):
    model = FaceNeSt(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Image preprocessing function
def process_image(image):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = transform(image).unsqueeze(0).to(device)
    return image
