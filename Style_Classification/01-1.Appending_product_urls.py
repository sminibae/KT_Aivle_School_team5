import pandas as pd
import re, os
from urllib.parse import urlparse, unquote

# remove metacharacters from title
def processed_title(text):
    return re.sub(r'\W+', ' ', text).lower()

# make list of keywords from title.
def extract_keywords(title):
    # This function should be improved with a better keyword extraction algorithm
    return title.split()

# match title and url, return score
def match_url(title_keywords, url):
    parsed_url = urlparse(url)
    url_path = unquote(parsed_url.path + parsed_url.query)
    url_keywords = re.findall(r'\b\w+\b', url_path)
    score = sum(keyword in url_keywords for keyword in title_keywords)
    return score

# search through urls, find best url for the title
def find_best_url_match(title, urls_df):
    title_keywords = extract_keywords(processed_title(title))
    best_score = 0
    best_url = None
    for url in urls_df['Product URL']:
        score = match_url(title_keywords, url)
        if score > best_score:
            best_score = score
            best_url = url
    return best_url


names = ['Sectional_Sofas', 'Sleeper_Sofas', 'Reclining_Sofas', 'LoveSeats', 'Futons', 'Settles', 'Convertibles', 
         'Accent_Chairs', 'Coffee_Tables', 'TV_Stands', 'End_Tables', 'Console_Tables', 'Ottomans', 'Living_Room_Sets', 
         'Decorative_Pillows', 'Throw_Blankets', 'Area_Rugs', 'Wall_Arts', 'Table_Lamps', 'Floor_Lamps', 
         'Pendants_and_Chandeliers', 'Sconces', 'Baskets_and_Storage', 'Candles', 'Live_Plants', 'Artificial_Plants', 
         'Planters', 'Decorative_Accessories', 'Window_Coverings', 'Decorative_Mirrors', 'Dining_Sets', 
         'Dining_Tables', 'Dining_Chairs', 'Bar_Stools', 'Kitchen_Islands', 'Buffets_and_Sideboards', 'China_Cabinets', 
         'Bakers_Recks', 'Bedroom_Sets', 'Mattresses', 'Nightstands', 'Dressers', 'Beds', 'Bedframes', 'Bases', 'Vanities', 
         'Entryway_Furnitures', 'Desks', 'Desk_Chairs', 'Bookcases', 
         'File_Cabinets', 'Computer_Armoires', 'Drafting_Tables', 'Cabinets', 'Furniture_Sets']


infos_path = '/home/all/product_infos/'
urls_path = '/home/all/product_urls/'

def main():
    for name in names:
        infos_file = os.path.join(infos_path, f'{name}_product_infos.csv')
        urls_file = os.path.join(urls_path, f'{name}_product_urls.csv')

        if os.path.exists(infos_file) and os.path.exists(urls_file):
            infos_csv = pd.read_csv(infos_file)
            urls_csv = pd.read_csv(urls_file)
            
            # Check if 'Product URL' column already exists
            if 'Product URL' in infos_csv.columns:
                print(f"'Product URL' column already exists in {infos_file}. Skipping.")
                continue

            # Apply the function
            print(f'Processing with {infos_file}')
            infos_csv['Product URL'] = infos_csv['Title'].apply(lambda title: find_best_url_match(title, urls_csv))

            # Save the modified infos_csv back to a file without 'Unnamed: 0' column
            infos_csv.to_csv(infos_file, index=False)
        else:
            if not os.path.exists(infos_file) and not os.path.exists(urls_file):
                print(f'No both files for {name}')
            else:
                if not os.path.exists(infos_file):
                 print(f'No infos csv for {name}')
                if not os.path.exists(urls_file):
                    print(f'No url csv for {name}')
            

if __name__ == "__main__":
    main()
