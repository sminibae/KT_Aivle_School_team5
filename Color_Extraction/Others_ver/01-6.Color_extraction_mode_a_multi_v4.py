# ========================================================================================


# 2.Frequency of each cluster(n=3)
import pandas as pd
import os
import numpy as np
import csv
from sklearn.cluster import KMeans
from PIL import Image
import requests
from io import BytesIO

def extract_ordered_dominant_colors(image_url, num_colors):
    # Load image from the web URL
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    image = np.array(image) / 255.0  # Normalize the image

    # Reshape the image to be a list of pixels
    pixels = image.reshape(-1, 3)
    
    # Use KMeans to find main colors
    model = KMeans(n_clusters=num_colors)
    model.fit(pixels)
    
    # Get the colors and labels
    colors = model.cluster_centers_
    labels = model.labels_
    
    # Count labels to find the frequency of each cluster
    count_labels = np.bincount(labels, minlength=num_colors)
    total_pixels = len(pixels)
    
    # Calculate the percentage of each cluster
    percentages = 100 * count_labels / total_pixels
    percentages_rounded = np.round(percentages, 2)  # Round to 2 decimal places
    
    # Order the clusters by their frequency (most common first)
    ordered_indices = np.argsort(count_labels)[::-1]  # Descending order
    ordered_colors = colors[ordered_indices]
    ordered_percentages = percentages_rounded[ordered_indices]
    
    return ordered_colors, ordered_percentages



names = [
    # 'Sectional_Sofas', 'Sleeper_Sofas', 
        'Reclining_Sofas', 
        # 'LoveSeats', 'Futons', 'Settles', 'Convertibles', 
        #  'Accent_Chairs', 'Coffee_Tables', 'TV_Stands', 'End_Tables', 'Console_Tables', 'Ottomans', 'Living_Room_Sets', 
        #  'Decorative_Pillows', 'Throw_Blankets', 'Area_Rugs', 'Wall_Arts', 'Table_Lamps', 'Floor_Lamps', 
        #  'Pendants_and_Chandeliers', 'Sconces', 'Baskets_and_Storage', 'Candles', 'Live_Plants', 'Artificial_Plants', 
        #  'Planters', 'Decorative_Accessories', 'Window_Coverings', 'Decorative_Mirrors', 'Dining_Sets', 
        #  'Dining_Tables', 'Dining_Chairs', 'Bar_Stools', 'Kitchen_Islands', 'Buffets_and_Sideboards', 'China_Cabinets', 
        #  'Bakers_Recks', 'Bedroom_Sets', 'Mattresses', 'Nightstands', 'Dressers', 'Beds', 'Bedframes', 'Bases', 'Vanities', 
        #  'Entryway_Furnitures', 'Desks', 'Desk_Chairs', 'Bookcases', 
        #  'File_Cabinets', 'Computer_Armoires', 'Drafting_Tables', 'Cabinets', 'Furniture_Sets'
         ]



from multiprocessing import Pool
import pandas as pd
import os
import csv
# from extract_ordered_dominant_colors import extract_ordered_dominant_colors  # Ensure this is defined or imported

# Set file paths
infos_path = '/home/all/product_infos/'
colors_path = '/home/all/product_infos_colors/'

def process_row(row):
    index = row.name
    image_path = row.get('Img_URL')  # Get the image path

    if not image_path:
        print(f"No image path for row {index}. Skipping.")
        return None, None
    else:
        try:
            # Extract dominant colors
            rgb_colors, rgb_percentages = extract_ordered_dominant_colors(image_path, num_colors=3)
            return rgb_colors, rgb_percentages
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None, None

def add_dominant_color_info(infos_path, colors_path, name):
    infos_file = os.path.join(infos_path, f'{name}_product_infos.csv')
    output_file = os.path.join(colors_path, f'{name}_product_infos_with_colors.csv')

    if os.path.exists(infos_file):
        # Read the input CSV file
        infos_csv = pd.read_csv(infos_file)

        # Check if 'RGB_3colors' and 'RGB_percentages' columns already exist
        if 'RGB_3colors' in infos_csv.columns and 'RGB_percentages' in infos_csv.columns:
            print(f"'RGB_3colors' and 'RGB_percentages' columns already exist in {infos_file}. Skipping.")
            return

        # Calculate the number of workers (total cores - 2)
        num_workers = max(1, os.cpu_count() - 2)  # Ensure at least one worker is used

        # Set up a pool of workers to process rows in parallel
        with Pool(num_workers) as pool:
            results = pool.map(process_row, [row for _, row in infos_csv.iterrows()])
            
        # Write results to the output CSV file
        with open(output_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(list(infos_csv.columns) + ['RGB_3colors', 'RGB_percentages'])
            for row, (colors, percentages) in zip(infos_csv.iterrows(), results):
                writer.writerow(list(row[1]) + [str(colors), str(percentages)])
    else:
        print(f"{infos_file} does not exist. Skipping.")

if __name__ == "__main__":
    for name in names:
        add_dominant_color_info(infos_path, colors_path, name)
