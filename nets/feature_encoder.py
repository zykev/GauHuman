
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader  
from PIL import Image
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description=(
        "Get image embeddings of an input image or directory of images."
    )
)

parser.add_argument(
    "--input",
    type=str,
    # required=True,
    default="/home/jli/datasets/ZJUMoCap/zju_mocap/my_377/images",
    help="Path to either a single input image or folder of images.",
)

parser.add_argument(
    "--output",
    type=str,
    # required=True,
    default="/home/jli/datasets/ZJUMoCap/zju_mocap/my_377/feature_maps",
    help=(
        "Path to the directory where embeddings will be saved. Output will be either a folder "
        "of .pt per image or a single .pt representing image embeddings."
    ),
)

parser.add_argument("--batch_size", type=int, default=16, help="The batch size to use for generation.")
parser.add_argument("--device", type=str, default="cuda:0", help="The device to run generation on.")
args = parser.parse_args()


# dino human feature map
dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
dino_model.eval()
dino_model.to(args.device)

patch_h, patch_w = 32, 32
feat_dim = 768   # base 768  large 1024



# gather all images
class ZjumocapDataset(Dataset):  
    def __init__(self, root_dir):  

        self.root_dir = root_dir 
        self.transform = transforms.Compose([
                                transforms.ToTensor(),              
                                transforms.Resize((patch_h * 14, patch_w * 14)), #should be multiple of model patch_size=14
                                transforms.CenterCrop((patch_h * 14, patch_w * 14)), #should be multiple of model patch_size=14                             
                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                ]) 
        self.image_paths = []
        self.mask_paths = [] 
  
        # 遍历所有views文件夹  
        for view_dir in os.listdir(root_dir):
            os.makedirs(os.path.join(args.output, view_dir), exist_ok=True)  
            view_path = os.path.join(root_dir, view_dir)  
            if os.path.isdir(view_path):  
                # 遍历每个view下的图片  
                for img_name in os.listdir(view_path):  
                    img_path = os.path.join(view_path, img_name)
                    mask_path = os.path.join(view_path.replace('images', 'mask'), img_name.replace('jpg', 'png'))
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path)  
  
    def __len__(self):  
        return len(self.image_paths)  
  
    def __getitem__(self, idx):  
        img_path = self.image_paths[idx]
        msk_path = self.mask_paths[idx]  
        image = Image.open(img_path).convert('RGB')  # 转换为RGB模式
        mask = Image.open(msk_path).convert('L')  # 转换为灰度图

        image = np.array(image)
        mask = np.array(mask)
        image[mask == 0] = 0
        image = self.transform(image)

        mask = torch.from_numpy(mask[None, ...]).to(torch.float32)
  
        return image, img_path, mask  
  
  
# 实例化Dataset  
dataset = ZjumocapDataset(root_dir=args.input)  
  
# 创建DataLoader  
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)  
for images, paths, masks in tqdm(dataloader, desc="Generating gt feature maps"):
    # get feature map
    with torch.no_grad():
        features_dict = dino_model.forward_features(images.to(args.device))
        human_feature = features_dict['x_norm_patchtokens']
        human_feature = human_feature.reshape(-1, patch_h, patch_w, feat_dim).contiguous()

        # obtain feature mask
        human_mask = torch.nn.functional.interpolate(masks.to(args.device), size=(patch_h*14, patch_w*14), mode='nearest')
        human_mask = torch.nn.functional.avg_pool2d(human_mask, kernel_size=14, stride=14)
        feature_mask = (human_mask > 0.5).reshape(-1, patch_h, patch_w, 1).to(torch.float32)
        human_feature = human_feature * feature_mask.expand_as(human_feature)
        human_feature = human_feature.permute(0, 3, 1, 2).contiguous()
        # save feature map for each feature in a batch
        for i, img_name in enumerate(paths):
            parts = img_name.split('/')
            view = parts[-2]  
            img_id = parts[-1].split('.')[0]  
            torch.save(human_feature[i], os.path.join(args.output, f"{view}/{img_id}_fmap.pt"))


    