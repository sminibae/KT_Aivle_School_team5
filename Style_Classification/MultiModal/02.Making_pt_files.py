# imports
from transformers import BertTokenizer, BertModel
import torch
from torchvision import transforms

import pandas as pd
import numpy as np
import os ,re
import ast
from PIL import Image
import h5py  # for .h5 file

from collections import Counter

from tqdm import tqdm

names = ['Sectional_Sofas', 'Sleeper_Sofas', 'Reclining_Sofas', 'LoveSeats', 'Futons', 'Settles', 'Convertibles', 
         'Accent_Chairs', 'Coffee_Tables', 'TV_Stands', 'End_Tables', 'Console_Tables', 'Ottomans', 'Living_Room_Sets', 
         'Decorative_Pillows', 'Throw_Blankets', 'Area_Rugs', 'Wall_Arts', 'Table_Lamps', 'Floor_Lamps', 
         'Pendants_and_Chandeliers', 'Sconces', 'Baskets_and_Storage', 'Candles', 'Live_Plants', 'Artificial_Plants', 
         'Planters', 'Decorative_Accessories', 'Window_Coverings', 'Decorative_Mirrors', 'Dining_Sets', 
         'Dining_Tables', 'Dining_Chairs', 'Bar_Stools', 'Kitchen_Islands', 'Buffets_and_Sideboards', 'China_Cabinets', 
         'Bakers_Recks', 'Bedroom_Sets', 'Mattresses', 'Nightstands', 'Dressers', 'Beds', 'Bedframes', 'Bases', 'Vanities', 
         'Entryway_Furnitures', 'Desks', 'Desk_Chairs', 'Bookcases', 
         'File_Cabinets', 'Computer_Armoires', 'Drafting_Tables', 'Cabinets', 'Furniture_Sets']

# 기본 경로 설정(01-4.Crawling_integrated를 넣었던 폴더)
base_path = '/home/all'

# 파일 이름을 안전하게 만드는 함수
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

# valid style 아니면 버리기
# Define valid styles
valid_styles = ['modern', 'contemporary', 'classic', 'urban', 'country', 'unique', 'minimalism']

# Modified categorize_style function
def map_style(style):
    categories = {
        "Modern": ["Modern",'Contemporary,Modern','French','Copenhagen','Modern Contemporary','Italian', "European",'Mid-Century Modern, Contemporary','Eclectic, modern, traditional','Modern, Classic', 'Modern couch','contemporary and traditional, modern','Casual, Modern','Modern, Contemporary',"Modern Minimalist", "High Gloss", "Scandinavian", "Nordic", "European", "Japanese", "Mid Century Modern",'Mid-Century Modern,Contemporary','Mid-Centuryum', "Contemporary Modern", "Minimalist Modern"],
        "Contemporary": ["Contemporary", "Streamlined", "Unadorned", "Sleek", "Understated", "Clean Lines", "Modern Contemporary", "Contemporary Chic"],
        "Classic": [ "Classic",'Classic Contemporary', "Antique", "Art Deco", "Colonial", "Baroque", "Vintage", "French", "Victorian", "Traditional", "Retro","Traditional Classic", "Vintage Classic"],
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

# 너무 적은 갯수를 가진 스타일 버리기
# 1. Calculate the frequency of each style
style_counts = Counter(combined_df['Style'])
print(combined_df['Style'].map(style_counts).unique())

# 2. Filter out styles with only one member
combined_df = combined_df[combined_df['Style'].map(style_counts) > 10]


# Load the tokenizer and base BERT model (not the sequence classification variant)
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = BertModel.from_pretrained('bert-large-uncased')

# Define the transformation of the image to tensor and normalization
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts to a tensor with values between 0 and 1
    transforms.Resize((1024, 1024)),  # Resize to a smaller size for demonstration purposes
])

total_length = len(combined_df)

# img 랑 text 각각 tensor로 바꾼 후 저장
def process_dataframe(df, tokenizer, model):
    # Container for all concatenated outputs
    all_concatenated_outputs = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        # Print progress every 100 rows
        if (total_length - index) % 100 == 0:
            print(f'Processing row {index}...')
        # Process text
        text = row['Product_Text']
        if isinstance(text, str) and text.strip():
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            text_tensor = outputs['pooler_output']
        else:
            text_tensor = torch.zeros(24, 1024)

        try:
            # Process image
            image_path = row['img_path']
            image = Image.open(image_path)
            image_tensor = transform(image)
        except Exception as e:
            print(f'Error processing image at row {index}: {e}')
            image_tensor_flat = torch.zeros(3, 1024, 1024)

        # Concatenate and append
        concatenated_tensor = torch.cat((text_tensor, image_tensor_flat), dim=1)
        all_concatenated_outputs.append(concatenated_tensor)

    return torch.cat(all_concatenated_outputs, dim=0)

# Process the dataframe and save all data
all_data_tensor = process_dataframe(combined_df, tokenizer, model)
torch.save(all_data_tensor, '/home/bae/Big_Project/torch_img_text.pt')

# Save 'Styles' column
styles = combined_df['Style'].to_numpy()
np.save('/home/bae/Big_Project/torch_styles.npy', styles)