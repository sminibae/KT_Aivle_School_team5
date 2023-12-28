
# ========================================================================================


# 2.Frequency of each cluster(n=3)
import pandas as pd
import os
import numpy as np
# from sklearn.cluster import KMeans
from PIL import Image
import requests
from io import BytesIO
from cuml.cluster import KMeans

def extract_ordered_dominant_colors_gpu(image_url, num_colors):
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


# ========================================================================================
names = ['Sectional_Sofas', 'Sleeper_Sofas', 
        # 'Reclining_Sofas', 'LoveSeats', 'Futons', 'Settles', 'Convertibles', 
        #  'Accent_Chairs', 'Coffee_Tables', 'TV_Stands', 'End_Tables', 'Console_Tables', 'Ottomans', 'Living_Room_Sets', 
        #  'Decorative_Pillows', 'Throw_Blankets', 'Area_Rugs', 'Wall_Arts', 'Table_Lamps', 'Floor_Lamps', 
        #  'Pendants_and_Chandeliers', 'Sconces', 'Baskets_and_Storage', 'Candles', 'Live_Plants', 'Artificial_Plants', 
        #  'Planters', 'Decorative_Accessories', 'Window_Coverings', 'Decorative_Mirrors', 'Dining_Sets', 
        #  'Dining_Tables', 'Dining_Chairs', 'Bar_Stools', 'Kitchen_Islands', 'Buffets_and_Sideboards', 'China_Cabinets', 
        #  'Bakers_Recks', 'Bedroom_Sets', 'Mattresses', 'Nightstands', 'Dressers', 'Beds', 'Bedframes', 'Bases', 'Vanities', 
        #  'Entryway_Furnitures', 'Desks', 'Desk_Chairs', 'Bookcases', 
        #  'File_Cabinets', 'Computer_Armoires', 'Drafting_Tables', 'Cabinets', 'Furniture_Sets'
         ]
            

# Set file paths
infos_path = '/home/all/product_infos/'
colors_path = '/home/all/product_infos_colors/'

def add_dominant_color_info(infos_path, colors_path):
    # Iterate over each category
    for name in names:
        infos_file = os.path.join(infos_path, f'{name}_product_infos.csv')
        
        # Check if the infos_file exists
        if os.path.exists(infos_file):
            infos_csv = pd.read_csv(infos_file)
            rgb_colors = []  # Store the RGB values for each product
            rgb_percentages = []  # Store the color percentages for each product

            # Check if 'RGB_3colors' and 'RGB_percentages' columns already exist
            if 'RGB_3colors' in infos_csv.columns and 'RGB_percentages' in infos_csv.columns:
                print(f"'RGB_3colors' and 'RGB_percentages' columns already exist in {infos_file}. Skipping.")
                continue

            # Iterate over each product
            for index, row in infos_csv.iterrows():
                image_path = row.get('Img_URL')  # Get the image path

                # Check if the image path is empty
                if not image_path:
                    print(f"No image path for row {index}. Skipping.")
                    rgb_colors.append(None)
                    rgb_percentages.append(None)
                    continue
        
                try:
                    # Extract dominant colors
                    ordered_colors, ordered_percentages = extract_ordered_dominant_colors(image_path, num_colors=3)
                    rgb_colors.append(str(ordered_colors))
                    rgb_percentages.append(str(ordered_percentages))
                    if index%100 ==0 :
                        
                        print("="*50)
                        print(f"Product {index}/{len(infos_csv)}:")
                        print("Ordered Dominant Colors (RGB):", ordered_colors)
                        print("Percentage of each color (rounded):", ordered_percentages)
                except Exception as e:
                    # Log any errors encountered
                    print(f"Error processing {image_path}: {e}")
                    rgb_colors.append(None)
                    rgb_percentages.append(None)

            # Add new columns for RGB values and percentages
            infos_csv['RGB_3colors'] = rgb_colors
            infos_csv['RGB_percentages'] = rgb_percentages
            # Save the updated DataFrame to a new file
            infos_csv.to_csv(os.path.join(colors_path, f'{name}_product_infos_with_colors.csv'), index=False)
        else:
            # Inform the user if a file does not exist
            print(f"{infos_file} does not exist. Skipping.")

if __name__ == "__main__":
    add_dominant_color_info(infos_path, colors_path)
