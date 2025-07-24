import os
import requests
import pandas as pd
from tqdm import tqdm

# Configuration
API_KEY = os.getenv('GSV_API')
IMAGE_SAVE_PATH = './data/raw_images'
IMAGE_SIZE = '640x640' 
HEADINGS = [0, 90, 180, 270]  # Four directions (N, E, S, W)
FOV = 90  # Field of view (degrees)
PITCH = 0  # Angle of camera

os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)

def fetch_gsv_image(lat, lon, heading, save_path):
    """Fetch and save one GSV image."""
    base_url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "size": IMAGE_SIZE,
        "location": f"{lat},{lon}",
        "heading": heading,
        "fov": FOV,
        "pitch": PITCH,
        "key": API_KEY
    }

    response = requests.get(base_url, params=params)
    
    if response.status_code == 200 and response.content:
        filename = f"{lat}_{lon}_{heading}.jpg"
        with open(os.path.join(save_path, filename), 'wb') as f:
            f.write(response.content)
    else:
        print(f"Failed to fetch image for ({lat}, {lon}) @ {heading}°")


def fetch_all_images(input_csv):
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} sample points.")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        lat, lon = row['lat'], row['lon']
        for heading in HEADINGS:
            fetch_gsv_image(lat, lon, heading, IMAGE_SAVE_PATH)


if __name__ == '__main__':
    input_csv = './data/west_midtown_points.csv'
    fetch_all_images(input_csv)
