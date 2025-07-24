import os
import torch
import torchvision.transforms as T
from torchvision import models
from PIL import Image
import numpy as np
from tqdm import tqdm

# === CONFIGURATION ===
IMAGE_DIR = './data/raw_images'
OUTPUT_DIR = './data/segmentation_masks'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color map
LABEL_COLORS = np.array([
    (0, 0, 0),        # 0 = background
    (128, 64,128),    # 1 = road
    (244, 35,232),    # 2 = sidewalk
    (70, 70, 70),     # 3 = building
    (102,102,156),    # 4 = wall
    (190,153,153),    # 5 = fence
    (153,153,153),    # 6 = pole
    (250,170, 30),    # 7 = traffic light
    (220,220,  0),    # 8 = traffic sign
    (107,142, 35),    # 9 = vegetation
    (152,251,152),    # 10 = terrain
    (70,130,180),     # 11 = sky
    (220, 20, 60),    # 12 = person
    (255,  0,  0),    # 13 = rider
    (0,  0,142),      # 14 = car
    (0,  0, 70),      # 15 = truck
    (0, 60,100),      # 16 = bus
    (0, 80,100),      # 17 = train
    (0,  0,230),      # 18 = motorcycle
    (119, 11, 32),    # 19 = bicycle
])

def decode_segmap(label_mask):
    """Map class index to RGB image."""
    rgb = LABEL_COLORS[label_mask % len(LABEL_COLORS)]
    return Image.fromarray(rgb.astype(np.uint8))

def load_model():
    model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()
    return model

def preprocess_image(image):
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

@torch.no_grad()
def run_inference(model, image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess_image(image)
    output = model(input_tensor)['out'][0]
    pred = torch.argmax(output, dim=0).cpu().numpy()
    return pred

def main():
    model = load_model()
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]

    for img_file in tqdm(image_files, desc='Segmenting images'):
        img_path = os.path.join(IMAGE_DIR, img_file)
        pred_mask = run_inference(model, img_path)

        # Save raw mask (class indices)
        np.save(os.path.join(OUTPUT_DIR, img_file.replace('.jpg', '.npy')), pred_mask)

        # Save colored visualization
        color_mask = decode_segmap(pred_mask)
        color_mask.save(os.path.join(OUTPUT_DIR, img_file.replace('.jpg', '_mask.png')))

if __name__ == '__main__':
    main()
