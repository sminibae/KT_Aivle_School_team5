# imports
import pandas as pd
import numpy as np
import csv, os ,re, glob
import ast
from collections import Counter

import torch
from torchvision import transforms
from PIL import Image
import h5py


'-------------------------------------------------------------------------------------------------------------------------'

names = ['Sectional_Sofas', 'Sleeper_Sofas', 'Reclining_Sofas', 'LoveSeats', 'Futons', 'Settles', 'Convertibles', 
         'Accent_Chairs', 'Coffee_Tables', 'TV_Stands', 'End_Tables', 'Console_Tables', 'Ottomans', 'Living_Room_Sets', 
         'Decorative_Pillows', 'Throw_Blankets', 'Area_Rugs', 'Wall_Arts', 'Table_Lamps', 'Floor_Lamps', 
         'Pendants_and_Chandeliers', 'Sconces', 'Baskets_and_Storage', 'Candles', 'Live_Plants', 'Artificial_Plants', 
         'Planters', 'Decorative_Accessories', 'Window_Coverings', 'Decorative_Mirrors', 'Dining_Sets', 
         'Dining_Tables', 'Dining_Chairs', 'Bar_Stools', 'Kitchen_Islands', 'Buffets_and_Sideboards', 'China_Cabinets', 
         'Bakers_Recks', 'Bedroom_Sets', 'Mattresses', 'Nightstands', 'Dressers', 'Beds', 'Bedframes', 'Bases', 'Vanities', 
         'Entryway_Furnitures', 'Desks', 'Desk_Chairs', 'Bookcases', 
         'File_Cabinets', 'Computer_Armoires', 'Drafting_Tables', 'Cabinets', 'Furniture_Sets']

# 기본 경로 설정(사진들을 넣었던 폴더)
base_path = '/home/all'

# 파일 이름에서 특수문자와 숫자 제거하는 함수
def sanitize_filename(filename):
    return re.sub(r'[^a-zA-Z]', '', filename)

# 이미지 경로를 가져오는 함수
def get_image_path(title, category):
    sanitized_title = sanitize_filename(title[:200])
    file_path = os.path.join(base_path, 'imgs', category, f"{sanitized_title}.jpg")
    return file_path if os.path.exists(file_path) else "File not found."

# 모든 CSV 파일을 처리하고 하나의 데이터프레임으로 합치는 함수
def process_all_csv_files():
    all_dfs = []
    for name in names:
        csv_file = os.path.join(base_path, f'product_infos/{name}_product_infos.csv')
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            df['img_path'] = df['Title'].apply(lambda title: get_image_path(title, name))
            all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)

# 모든 데이터를 하나의 데이터프레임으로 합치기
combined_df = process_all_csv_files()

# path에 주소가 없는 경우(크롤링이 실패한 사례)
combined_df = combined_df[combined_df['img_path'] != 'File not found.'].reset_index(drop=True)

# Style 라벨이 없으면 버리기
def extract_style(row):
    for col in ['Product_Info', 'Product_Feature']:
        try:
            info_dict = ast.literal_eval(row[col])
            if 'Style' in info_dict:
                return info_dict['Style']
            if 'style' in info_dict:
                return info_dict['style']
        except (ValueError, SyntaxError):
            continue
    return None

# Apply the function to each row
combined_df['Style'] = combined_df.apply(extract_style, axis=1)

# Drop rows where 'Style' is None
combined_df.dropna(subset=['Style'], inplace=True)


# Define valid styles
valid_styles = ['modern', 'contemporary', 'classic', 'urban', 'country', 'unique', 'minimalism']

# Modified categorize_style function
def map_style(style):
    categories = {
        "Modern": ["Modern",'Contemporary,Modern','French','Copenhagen','Modern Contemporary','Italian','Mid-Century Modern, Contemporary','Eclectic, modern, traditional','Modern, Classic', 'Modern couch','contemporary and traditional, modern','Casual, Modern','Modern, Contemporary',"Modern Minimalist", "High Gloss", "Scandinavian", "Nordic", "European", "Japanese", "Mid Century Modern",'Mid-Century Modern,Contemporary','Mid-Centuryum', "Contemporary Modern", "Minimalist Modern"],
        "Contemporary": ["Contemporary", "Streamlined", "Unadorned", "Sleek", "Understated", "Clean Lines", "Modern Contemporary", "Contemporary Chic"],
        "Classic": [ "Classic",'Classic Contemporary', "Antique", "Art Deco", "Colonial", "Baroque", "Vintage", "European", "French", "Victorian", "Traditional", "Retro","Traditional Classic", "Vintage Classic"],
        "Urban": ["Urban", "Metropolitan", "City Style", "Modern Urban", "Urban Contemporary", "Industrial", "Loft", "Modern Industrial", "Rustic Industrial", "Industrial Retro Style", "Metropolitan","Urban Industrial", "Industrial Chic",'Retro'],
        "Country": ["Country", "Rustic Country", "Country Style", "Rural", "Pastoral", "Provincial","Rustic", "Farmhouse", "Country Rustic", "Shabby Chic", "Lodge", "Reclaimed Wood","Country Rustic", "Rustic Charm"],
        "Unique": ["Unique",  "One-of-a-Kind", "Unique Design","Free Style", "Wild", "Fantasy Plus", "Boho Style", "Bohemian","Eclectic","Bohemian Eclectic", "Eclectic Mix","Fusion", "Quirky", "Galaxy", "Stars",'Bold eclectic'],
        "Minimalism": ["Minimalism", "Simple", "Zen", "Bare", "Sparse", "Minimalist","Simplistic Minimalism"]
    }

    for key, values in categories.items():
        if style in values:
            return key
    return style
 

# Simplified is_valid_style function
def is_valid_style(style):
    mapped_style = map_style(style)
    if mapped_style.lower() in valid_styles:
        return mapped_style
    else:
        return None

# Apply the function to the 'Style' column of the DataFrame
combined_df['Style'] = combined_df['Style'].apply(is_valid_style)

# Drop rows where 'Style' is None
combined_df.dropna(subset=['Style'], inplace=True)


# 분류된 갯수가 10개 미만이면 버리기
# 1. Calculate the frequency of each style
style_counts = Counter(combined_df['Style'])

# 2. Filter out styles with only ten member
combined_df = combined_df[combined_df['Style'].map(style_counts) > 10]


# img_path 와 Style 컬럼만 남기기
combined_df = combined_df[['img_path', 'Style']].reset_index(drop=True)

'-------------------------------------------------------------------------------------------------------------------------'

# Define a transformation pipeline
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor()  # Converts to a tensor and scales to [0, 1]
])

def process_image(image_path):
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            return transform(img)
    except IOError:
        print(f"Error in processing image: {image_path}")
        return None

successful_indices = []

# Attempt to process each image and collect indices of successful ones
for index, row in combined_df.iterrows():
    img_tensor = process_image(row['img_path'])
    if img_tensor is not None:
        successful_indices.append(index)
        if index % 1000 == 0:  # Print progress every 1000 images
            print(f'Processed {index}/{len(combined_df)} images')

# Create a new DataFrame with only successfully processed images
combined_df = combined_df.loc[successful_indices].reset_index(drop=True)

def save_image_to_h5(image_tensor, h5file, index):
    if image_tensor is not None:
        h5file['images'][index, ...] = image_tensor.numpy()

# Save styles to an .npy file
styles = []

# Open an h5 file for writing
with h5py.File('/home/all/processed_data/image_torchtensor_(3,1024,1024).h5', 'w') as h5file:
    num_images = len(combined_df)
    # Create a dataset for images using chunked storage
    chunk_size = 16  # Adjust chunk size based on your memory constraints
    h5file.create_dataset('images', shape=(num_images, 3, 1024, 1024), 
                          chunks=(chunk_size, 3, 1024, 1024), dtype=np.float32)

    # Process and save each image
    for index, row in combined_df.iterrows():
        img_tensor = process_image(row['img_path'])
        if img_tensor is not None:
            save_image_to_h5(img_tensor, h5file, index)
            styles.append(row['Style'])
            if index % 100 == 0:  # Print progress every 100 images
                print(f'Processed {index}/{num_images} images')

# Convert styles list to a NumPy array
styles_np = np.array(styles)
np.save('/home/all/processed_data/styles_(3*1024*1024).npy', styles_np)
print("All images have been processed and saved.")
