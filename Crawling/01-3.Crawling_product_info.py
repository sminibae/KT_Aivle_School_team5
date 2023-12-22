# imports
import pandas as pd
import requests

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from selenium_stealth import stealth

from browsermobproxy import Server


import time, os, csv, random, html
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup

from selenium.common.exceptions import TimeoutException

'-------------------------------------------------------------------------------'

# Fetch Data from product page
def fetch_info_from_product_page(driver):
    # fetch whole html from url
    html_content = driver.page_source
    
    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Get each information    
    
    # img_url
    selected_imgs = soup.select('#imgTagWrapperId img, .imgTagWrapper img')
    for img in selected_imgs:
        image_url = img.get('data-old-hires', img.get('src'))  # Default to src if data-old-hires is not present

    
    # category
    breadcrumb_links = soup.select('#wayfinding-breadcrumbs_feature_div ul a')
    category = [html.unescape(link.get_text().strip()) for link in breadcrumb_links]
    
    # title
    title = soup.find('span', {'id': 'productTitle'}).get_text(strip=True)

    # price
    price_whole_part = soup.find('span', class_='a-price-whole')
    price_fraction_part = soup.find('span', class_='a-price-fraction')
    price = f'${price_whole_part.get_text(strip=True)}{price_fraction_part.get_text(strip=True)}' if price_whole_part and price_fraction_part else 'Price not Found'

    # product info (manufacturer)
    product_info = {}
    table = soup.find('table', class_='a-normal a-spacing-micro')
    if table:
        # Loop through each row in the table
        for row in table.find_all('tr'):
            # Extract columns: key and value
            columns = row.find_all('td')
            if len(columns) == 2:  # Ensure that there are exactly 2 columns
                key = columns[0].get_text(strip=True)
                value = columns[1].get_text(strip=True)
                product_info[key] = value
    else:
        # Handle the case where the table is not found
        product_info = None  # or {} if you prefer an empty dictionary


    # product feature
    tables = soup.find_all('table', class_='a-normal')
    product_features = {}
    for table in tables:
        rows = table.find_all('tr')
        for row in rows:
            # Extract the feature name
            feature_name_element = row.find('span', class_='a-text-bold')
            if feature_name_element:
                feature_name = feature_name_element.get_text(strip=True)
                # Extract the feature value
                feature_values = row.find_all('span', class_='a-size-base handle-overflow')
                if feature_values:
                    feature_value = feature_values[-1].get_text(strip=True)
                    product_features[feature_name] = feature_value
    
    # product detail text
    about_section = soup.find('div', id='feature-bullets')
    list_items = about_section.find_all('span', class_='a-list-item') if about_section else []
    product_text = ' '.join(item.get_text(strip=True) for item in list_items)
        
    # return
    return category, title, price, product_info, product_features, product_text, image_url
    
'-------------------------------------------------------------------------------'

# configure selenium chrome driver
def configure_driver():
    # Start the BrowserMob Proxy server
    server = Server(r"C:\browsermob-proxy-2.1.4\bin\browsermob-proxy")
    server.start()
    proxy = server.create_proxy()
        
    # chromedriver_path = '/home/bae/.cache/selenium/chromedriver/linux64/120.0.6099.71/chromedriver'
    # possible paths
    # /home/bae/.cache/selenium/chromedriver
    # /home/bae/.cache/selenium/chromedriver/linux64/120.0.6099.71/chromedriver
    # /mnt/c/Users/user/.cache/selenium/chromedriver
    # service = Service(chromedriver_path)
    
    chrome_options = Options()
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 11.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.6099.110 Safari/537.36")
    chrome_options.add_argument("--proxy-server={0}".format(proxy.proxy)) # Set up Selenium to use the proxy
    # chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("disable-infobars")
    chrome_options.add_argument("--disable-extensions")
    # chrome_options.add_argument("--remote-debugging-port=9222")
    
    driver = webdriver.Chrome(options=chrome_options)
    
    # Create custom headers (including Referer)
    headers = {
        "Referer": "https://www.amazon.com/Living-Room-Furniture/b?node=3733551",
        # Add any other custom headers you want
    }
    proxy.add_to_capabilities(headers)
    
    # Apply stealth settings    
    stealth(driver,
            languages=["en-US", "en"],  # List of languages
            vendor="Google Inc.",  # Vendor name
            platform="Win64",  # Platform
            webgl_vendor="Intel Inc.",  # WebGL vendor
            renderer="Intel Iris OpenGL Engine",  # WebGL renderer
            fix_hairline=True,  # Fix for thin lines issue
            )
    
    return driver

'-------------------------------------------------------------------------------'

def scrape_product_infos(name):      
    csv_file_path_products = f'product_urls/{name}_product_urls.csv'
    
    csv_file_path_product_infos = f'product_infos/{name}_product_infos.csv'
    csv_file_path_product_errors = f'error_products/{name}_product_info_fetch_error_page_urls.csv'
    
    is_Product_csv_exist = os.path.exists(csv_file_path_products)
    if not is_Product_csv_exist:
        print(f'{name}_product_urls.csv file doesnt exist')
        return None
    headers_needed_product_infos_csv = not os.path.exists(csv_file_path_product_infos)
    
    # if no error log file, make one
    headers_needed_error_csv = not os.path.exists(csv_file_path_product_errors)
    if headers_needed_error_csv:
        with open(csv_file_path_product_errors, mode='a', newline='', encoding='utf-8') as error_file:
            error_writer = csv.writer(error_file)
            error_writer.writerow(['Product Pages which made error getting each product infos'])
            error_file.flush()
        
    # Read each product url csv file
    products = pd.read_csv(csv_file_path_products)
    i = 0  # Initialize i
    
    with open(csv_file_path_product_infos, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # if no file, add header
        if headers_needed_product_infos_csv:
            writer.writerow(['Category' , 'Title', 'Price', 'Product_Info', 'Product_Feature', 'Product_Text', 'Img_URL'])

        # configure selenium chrome driver
        driver = configure_driver()
        
        retry_count = 0  # retry count initialize
        max_retries = 10  # Adjust as needed

        while i < len(products):  # while product_url csv file lasts
            product_url = products.iloc[i,0]  # each row of the product_url csv file
            
            driver.get(product_url)  # open the page
            
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))  # Waits until the body tag is present or max 10 sec
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")  # Execute script to disable navigator.webdriver flag
            time.sleep(random.randint(2000, 5000)/1000)  # Random sleep to mimic human behavior
            
            # exit when may retry count 
            if retry_count >= max_retries:
                break
            
            product_infos = fetch_info_from_product_page(driver)  # which is a list of lists with around 25 sublists.
            if product_infos:             
                writer.writerow([product_infos])
                file.flush()  # Force writing to CSV file
                retry_count = 0
                
                i += 1
                product_url = products.iloc[i,0]  # next row(next url) of the csv file
                
            else:
                print("Product info not found or opening webpage failed. Retrying...")
                # if failed fetching product infos from a product, log that product for later retrying
                with open(csv_file_path_product_errors, mode='a', newline='', encoding='utf-8') as error_file:
                    error_writer = csv.writer(error_file)
                    error_writer.writerow([product_url])
                    error_file.flush()
                retry_count += 1
            
        driver.quit()
        print(f'{name} done. moving to next category')
        
'-------------------------------------------------------------------------------'

Sofa_and_Couches = {
    'Sectional_Sofas' : 'https://www.amazon.com/s?bbn=3733551&rh=n%3A1055398%2Cn%3A%211063498%2Cn%3A1063306%2Cn%3A1063318%2Cn%3A3733551%2Cp_n_feature_two_browse-bin%3A3248836011&dc&fst=as%3Aoff&pf_rd_i=3733551&pf_rd_m=ATVPDKIKX0DER&pf_rd_p=877c0809-fc53-42a6-91f2-7c8f50e6e9be&pf_rd_r=0X2CMRE53H2HSQR09781&pf_rd_s=merchandised-search-2&pf_rd_t=101&qid=1528841766&rnid=3248834011&ref=s9_acss_bw_cg_HarSofa_2a1_w',
    'Sleeper_Sofas' : 'https://www.amazon.com/s?i=garden&bbn=3733551&rh=n%3A1055398%2Cn%3A%211063498%2Cn%3A1063306%2Cn%3A1063318%2Cn%3A3733551%2Cp_n_feature_two_browse-bin%3A3248838011&pf_rd_i=3733551&pf_rd_i=3733551&pf_rd_m=ATVPDKIKX0DER&pf_rd_m=ATVPDKIKX0DER&pf_rd_p=3009ccc5-2584-454e-b732-ae8e525543fe&pf_rd_p=877c0809-fc53-42a6-91f2-7c8f50e6e9be&pf_rd_r=0X2CMRE53H2HSQR09781&pf_rd_r=79FTK5EJP86JTXQETKYX&pf_rd_s=merchandised-search-2&pf_rd_s=merchandised-search-2&pf_rd_t=101&pf_rd_t=101&ref=s9_acss_bw_cg_HarSofa_2b1_w',
    'Reclining_Sofas' : 'https://www.amazon.com/s?i=garden&bbn=3733551&rh=n%3A1055398%2Cn%3A%211063498%2Cn%3A1063306%2Cn%3A1063318%2Cn%3A3733551%2Cp_n_feature_two_browse-bin%3A12012870011&pf_rd_i=3733551&pf_rd_i=3733551&pf_rd_m=ATVPDKIKX0DER&pf_rd_m=ATVPDKIKX0DER&pf_rd_p=3009ccc5-2584-454e-b732-ae8e525543fe&pf_rd_p=877c0809-fc53-42a6-91f2-7c8f50e6e9be&pf_rd_r=0X2CMRE53H2HSQR09781&pf_rd_r=79FTK5EJP86JTXQETKYX&pf_rd_s=merchandised-search-2&pf_rd_s=merchandised-search-2&pf_rd_t=101&pf_rd_t=101&ref=s9_acss_bw_cg_HarSofa_2c1_w',
    'LoveSeats' : 'https://www.amazon.com/s?i=garden&bbn=3733551&rh=n%3A1055398%2Cn%3A%211063498%2Cn%3A1063306%2Cn%3A1063318%2Cn%3A3733551%2Cp_n_feature_two_browse-bin%3A3248835011&pf_rd_i=3733551&pf_rd_i=3733551&pf_rd_m=ATVPDKIKX0DER&pf_rd_m=ATVPDKIKX0DER&pf_rd_p=3009ccc5-2584-454e-b732-ae8e525543fe&pf_rd_p=877c0809-fc53-42a6-91f2-7c8f50e6e9be&pf_rd_r=79FTK5EJP86JTXQETKYX&pf_rd_r=QZ6K0G39A8CCYDMBWJRD&pf_rd_s=merchandised-search-2&pf_rd_s=merchandised-search-2&pf_rd_t=101&pf_rd_t=101&ref=s9_acss_bw_cg_HarSofa_2d1_w',
    'Futons' : 'https://www.amazon.com/Futons/b/ref=s9_acss_bw_cg_SofaType_1f1_w/ref=s9_acss_bw_cg_HarSofa_3a1_w?ie=UTF8&node=13753041&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-2&pf_rd_r=79FTK5EJP86JTXQETKYX&pf_rd_t=101&pf_rd_p=3009ccc5-2584-454e-b732-ae8e525543fe&pf_rd_i=3733551&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-2&pf_rd_r=QZ6K0G39A8CCYDMBWJRD&pf_rd_t=101&pf_rd_p=877c0809-fc53-42a6-91f2-7c8f50e6e9be&pf_rd_i=3733551',
    'Settles' : 'https://www.amazon.com/s?i=garden&bbn=3733551&rh=n%3A1055398%2Cn%3A%211063498%2Cn%3A1063306%2Cn%3A1063318%2Cn%3A3733551%2Cp_n_feature_two_browse-bin%3A3248837011&pf_rd_i=3733551&pf_rd_i=3733551&pf_rd_m=ATVPDKIKX0DER&pf_rd_m=ATVPDKIKX0DER&pf_rd_p=3009ccc5-2584-454e-b732-ae8e525543fe&pf_rd_p=877c0809-fc53-42a6-91f2-7c8f50e6e9be&pf_rd_r=79FTK5EJP86JTXQETKYX&pf_rd_r=QZ6K0G39A8CCYDMBWJRD&pf_rd_s=merchandised-search-2&pf_rd_s=merchandised-search-2&pf_rd_t=101&pf_rd_t=101&ref=s9_acss_bw_cg_HarSofa_3b1_w',
    'Convertibles' : 'https://www.amazon.com/s?bbn=3733551&rh=n%3A1055398%2Cn%3A%211063498%2Cn%3A1063306%2Cn%3A1063318%2Cn%3A3733551%2Cp_n_feature_two_browse-bin%3A12012869011&dc&fst=as%3Aoff&pf_rd_i=3733551&pf_rd_m=ATVPDKIKX0DER&pf_rd_p=877c0809-fc53-42a6-91f2-7c8f50e6e9be&pf_rd_r=QZ6K0G39A8CCYDMBWJRD&pf_rd_s=merchandised-search-2&pf_rd_t=101&qid=1528765569&rnid=3248834011&ref=s9_acss_bw_cg_HarSofa_3c1_w',
}

Other_Livingroom_Furniture = {
    'Accent_Chairs' : 'https://www.amazon.com/b?node=3733491&ref=s9_acss_bw_cg_SBR2019_3b1_w', # 42 pages
    'Coffee_Tables' : 'https://www.amazon.com/b?node=3733631&ref=s9_acss_bw_cg_SBR2019_3c1_w', # 35 pages
    'TV_Stands' : 'https://www.amazon.com/b?node=14109851&ref=s9_acss_bw_cg_SBR2019_3d1_w',
    'End_Tables' : 'https://www.amazon.com/b?node=3733641&ref=s9_acss_bw_cg_SBR2019_4a1_w',
    'Console_Tables' : 'https://www.amazon.com/b?node=3733651&ref=s9_acss_bw_cg_SBR2019_4b1_w',
    'Ottomans' : 'https://amazon.com/b/ref=sv_hg_fl_3254639011/ref=s9_acss_bw_cg_SBR2019_4c1_w?ie=UTF8&node=3254639011&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-3&pf_rd_r=HH88E0EETR0MQZ658N3D&pf_rd_t=101&pf_rd_p=f126b4c9-d5f7-41d5-85e0-9db2657ec29e&pf_rd_i=14544497011',
    'Living_Room_Sets' : 'https://www.amazon.com/b?node=3733481&ref=s9_acss_bw_cg_SBR2019_4d1_w',
}

Decor_and_Soft_Furnishings = {
    'Decorative_Pillows' : 'https://www.amazon.com/s?rh=n%3A3732321&fs=true&ref=lp_3732321_sar', # 161 pages
    'Throw_Blankets' : 'https://www.amazon.com/s?rh=n%3A14058581&fs=true&ref=lp_14058581_sar', # 400 pages
    'Area_Rugs' : 'https://www.amazon.com/s?rh=n%3A684541011&fs=true&ref=lp_684541011_sar', # 400 pages
    'Wall_Arts' : 'https://www.amazon.com/s?rh=n%3A3736081&fs=true&ref=lp_3736081_sar' , # 400 pages
    'Table_Lamps' : 'https://www.amazon.com/b?node=1063296&ref=s9_acss_bw_cg_SBR2019_7a1_w', # 347 pages
    'Floor_Lamps' : 'https://www.amazon.com/b?node=1063294&ref=s9_acss_bw_cg_SBR2019_7b1_w', # 131 pages
    'Pendants_and_Chandeliers' : 'https://www.amazon.com/lighting-ceiling-fans/b/ref=s9_acss_bw_cg_SBR2019_7c1_w?ie=UTF8&node=495224&ref_=sv_hg_5&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-3&pf_rd_r=HH88E0EETR0MQZ658N3D&pf_rd_t=101&pf_rd_p=f126b4c9-d5f7-41d5-85e0-9db2657ec29e&pf_rd_i=14544497011', #298 pages
    'Sconces' : 'https://www.amazon.com/b?node=3736721&ref=s9_acss_bw_cg_SBR2019_7d1_w', # 238 pages
    'Baskets_and_Storage' : 'https://www.amazon.com/s?rh=n%3A2422430011&fs=true&ref=lp_2422430011_sar', # 244 pages
    'Candles' : 'https://www.amazon.com/s?rh=n%3A3734391&fs=true&ref=lp_3734391_sar', # 400 pages
    'Live_Plants' : 'https://www.amazon.com/b/ref=sv_hg_fl_553798/ref=s9_acss_bw_cg_SBR2019_8c1_w?node=3480662011&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-3&pf_rd_r=HH88E0EETR0MQZ658N3D&pf_rd_t=101&pf_rd_p=f126b4c9-d5f7-41d5-85e0-9db2657ec29e&pf_rd_i=14544497011', # 14 pages
    'Artificial_Plants' : 'https://www.amazon.com/b?node=14087351&ref=s9_acss_bw_cg_SBR2019_8d1_w', # 400 pags    
    'Planters' : 'https://www.amazon.com/b?node=553798&ref=s9_acss_bw_cg_SBR2019_9a1_w', # 263 pages
    'Decorative_Accessories' : 'https://www.amazon.com/s?rh=n%3A3295676011&fs=true&ref=lp_3295676011_sar', # 400 pages
    'Window_Coverings' : 'https://www.amazon.com/b?node=1063302&ref=s9_acss_bw_cg_SBR2019_9c1_w', # 400 pages
    'Decorative_Mirrors' : 'https://www.amazon.com/b?node=3736371&ref=s9_acss_bw_cg_SBR2019_9d1_w', # 314 pages
}

Kitchen_and_Dining_Furnitures = {
    'Dining_Sets' : 'https://www.amazon.com/b/ref=s9_acss_bw_cg_HarFK_2a1_w?node=8566630011&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-2&pf_rd_r=RDTFVCNEA5ZFMYQZYZ3E&pf_rd_t=101&pf_rd_p=fde2e008-b6c2-4e44-9488-63f47804b655&pf_rd_i=3733781',
    'Dining_Tables' : 'https://www.amazon.com/b/ref=s9_acss_bw_cg_HarFK_2b1_w?node=3733811&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-2&pf_rd_r=RDTFVCNEA5ZFMYQZYZ3E&pf_rd_t=101&pf_rd_p=fde2e008-b6c2-4e44-9488-63f47804b655&pf_rd_i=3733781',
    'Dining_Chairs' : 'https://www.amazon.com/b/ref=s9_acss_bw_cg_HarFK_2c1_w?node=3733821&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-2&pf_rd_r=RDTFVCNEA5ZFMYQZYZ3E&pf_rd_t=101&pf_rd_p=fde2e008-b6c2-4e44-9488-63f47804b655&pf_rd_i=3733781',
    'Bar_Stools' : 'https://www.amazon.com/b/ref=s9_acss_bw_cg_HarFK_2d1_w?node=3733851&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-2&pf_rd_r=RDTFVCNEA5ZFMYQZYZ3E&pf_rd_t=101&pf_rd_p=fde2e008-b6c2-4e44-9488-63f47804b655&pf_rd_i=3733781',
    'Kitchen_Islands' : 'https://www.amazon.com/b/ref=s9_acss_bw_cg_HarFK_3a1_w?node=8521400011&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-2&pf_rd_r=RDTFVCNEA5ZFMYQZYZ3E&pf_rd_t=101&pf_rd_p=fde2e008-b6c2-4e44-9488-63f47804b655&pf_rd_i=3733781',
    'Buffets_and_Sideboards' : 'https://www.amazon.com/b/ref=s9_acss_bw_cg_HarFK_3b1_w?node=3733831&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-2&pf_rd_r=RDTFVCNEA5ZFMYQZYZ3E&pf_rd_t=101&pf_rd_p=fde2e008-b6c2-4e44-9488-63f47804b655&pf_rd_i=3733781',
    'China_Cabinets' : 'https://www.amazon.com/b/ref=s9_acss_bw_cg_HarFK_3c1_w?node=3733841&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-2&pf_rd_r=RDTFVCNEA5ZFMYQZYZ3E&pf_rd_t=101&pf_rd_p=fde2e008-b6c2-4e44-9488-63f47804b655&pf_rd_i=3733781',
    'Bakers_Recks' : 'https://www.amazon.com/s?rh=n%3A3744061&fs=true&ref=lp_3744061_sar',
}

Bedroom_Furnitures = {
    'Bedroom_Sets' : 'https://www.amazon.com/b/ref=s9_acss_bw_cg_HarFLiv_2a1_w?node=3732931&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-2&pf_rd_r=MNGW2YC86GM462Z294K6&pf_rd_t=101&pf_rd_p=d8003c9c-74d3-490c-ae40-6aac6ed45f61&pf_rd_i=1063308',
    'Mattresses' : 'https://www.amazon.com/s?rh=n%3A3732961&fs=true&ref=lp_3732961_sar',
    'Nightstands' : 'https://www.amazon.com/b/ref=s9_acss_bw_cg_HarFLiv_2c1_w?node=3733251&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-2&pf_rd_r=MNGW2YC86GM462Z294K6&pf_rd_t=101&pf_rd_p=d8003c9c-74d3-490c-ae40-6aac6ed45f61&pf_rd_i=1063308',
    'Dressers' : 'https://www.amazon.com/b/ref=s9_acss_bw_cg_HarFLiv_2d1_w?node=3733261&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-2&pf_rd_r=MNGW2YC86GM462Z294K6&pf_rd_t=101&pf_rd_p=d8003c9c-74d3-490c-ae40-6aac6ed45f61&pf_rd_i=1063308',
    'Beds' : 'https://www.amazon.com/b/ref=s9_acss_bw_cg_HarFLiv_3a1_w?node=3248804011&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-2&pf_rd_r=MNGW2YC86GM462Z294K6&pf_rd_t=101&pf_rd_p=d8003c9c-74d3-490c-ae40-6aac6ed45f61&pf_rd_i=1063308',
    'Bedframes' : 'https://www.amazon.com/b/ref=s9_acss_bw_cg_HarFLiv_3b1_w?node=3248801011&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-2&pf_rd_r=MNGW2YC86GM462Z294K6&pf_rd_t=101&pf_rd_p=d8003c9c-74d3-490c-ae40-6aac6ed45f61&pf_rd_i=1063308',
    'Bases' : 'https://www.amazon.com/s?rh=n%3A17873917011&language=en_US&brr=1&pf_rd_i=1063308&pf_rd_m=ATVPDKIKX0DER&pf_rd_p=d8003c9c-74d3-490c-ae40-6aac6ed45f61&pf_rd_r=MNGW2YC86GM462Z294K6&pf_rd_s=merchandised-search-2&pf_rd_t=101&rd=1',
    'Vanities' : 'https://www.amazon.com/b/ref=s9_acss_bw_cg_HarFLiv_3d1_w?node=3733291&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-2&pf_rd_r=MNGW2YC86GM462Z294K6&pf_rd_t=101&pf_rd_p=d8003c9c-74d3-490c-ae40-6aac6ed45f61&pf_rd_i=1063308',    
}

Entryway_Furnitures = {
    'Entryway_Furnitures' : 'https://www.amazon.com/b/ref=s9_acss_bw_cg_FNNAV_5e1_w?node=3249856011&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-7&pf_rd_r=AMH2XC97ZGGNAWND0C03&pf_rd_t=101&pf_rd_p=d0c31dca-221f-4a26-9f62-a9e18ca9f354&pf_rd_i=1063306',
}

Home_Office_Furnitures = {
    'Desks' : 'https://www.amazon.com/b/ref=s9_acss_bw_cg_HarHOF_2a1_w?node=3733671&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-2&pf_rd_r=S6B16AQ2FYJDBJXK6X9C&pf_rd_t=101&pf_rd_p=b0e12b53-6018-44fe-b7a4-d1871cf08ee9&pf_rd_i=1063312',
    'Desk_Chairs' : 'https://www.amazon.com/b/ref=s9_acss_bw_cg_HarHOF_2b1_w?node=3733721&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-2&pf_rd_r=S6B16AQ2FYJDBJXK6X9C&pf_rd_t=101&pf_rd_p=b0e12b53-6018-44fe-b7a4-d1871cf08ee9&pf_rd_i=1063312',
    'Bookcases' : 'https://www.amazon.com/b/ref=s9_acss_bw_cg_HarHOF_2c1_w?node=10824421&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-2&pf_rd_r=S6B16AQ2FYJDBJXK6X9C&pf_rd_t=101&pf_rd_p=b0e12b53-6018-44fe-b7a4-d1871cf08ee9&pf_rd_i=1063312',
    'File_Cabinets' : 'https://www.amazon.com/b/ref=s9_acss_bw_cg_HarHOF_2d1_w?node=1069166&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-2&pf_rd_r=S6B16AQ2FYJDBJXK6X9C&pf_rd_t=101&pf_rd_p=b0e12b53-6018-44fe-b7a4-d1871cf08ee9&pf_rd_i=1063312',
    'Computer_Armoires' : 'https://www.amazon.com/s?rh=n%3A3733751&fs=true&ref=lp_3733751_sar',
    'Drafting_Tables' : 'https://www.amazon.com/s?rh=n%3A3733771&fs=true&ref=lp_3733771_sar',
    'Cabinets' : 'https://www.amazon.com/s?rh=n%3A3733761&fs=true&ref=lp_3733761_sar',
    'Furniture_Sets' : 'https://www.amazon.com/b/ref=s9_acss_bw_cg_HarHOF_3d1_w?node=3733661&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-2&pf_rd_r=S6B16AQ2FYJDBJXK6X9C&pf_rd_t=101&pf_rd_p=b0e12b53-6018-44fe-b7a4-d1871cf08ee9&pf_rd_i=1063312',
}

dictionaries = [
    Sofa_and_Couches, Other_Livingroom_Furniture, Decor_and_Soft_Furnishings, 
    Kitchen_and_Dining_Furnitures,Bedroom_Furnitures, Entryway_Furnitures, Home_Office_Furnitures,
]

'-------------------------------------------------------------------------------'

if __name__ == "__main__":
    for each_dictionary in dictionaries:
        for name, url in each_dictionary.items():
            scrape_product_infos(name=name)