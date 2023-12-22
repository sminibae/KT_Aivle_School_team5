# imports
import pandas as pd
import requests

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, TimeoutException, WebDriverException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from selenium_stealth import stealth

from browsermobproxy import Server
from amazoncaptcha import AmazonCaptcha


import time, os, csv, random, html, re
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup

'-------------------------------------------------------------------------------'

# Function to click on next page link
def click_next_page(driver):
    try:
        # Find the 'Next Page' link using its aria-label
        next_page_link = driver.find_element(By.XPATH, "//a[contains(@aria-label, 'Go to next page')]")

        # If the link is found, click it
        if next_page_link:
            actions = ActionChains(driver)
            actions.move_to_element(next_page_link).click().perform()

            # Optionally, wait for the page to load after clicking
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
            time.sleep(random.randint(2000, 5000)/1000)  # Random sleep to mimic human behavior

            # Return the driver object
            return True, driver
        
    except NoSuchElementException:
        # If the link is not found
        print("Next page link not found.")
        return False, driver  # Return None to indicate that the next page link was not found

    
# Get Next Page's URL
def get_nextpage_url(driver):
    try:
        # get the html

        time.sleep(1)
        
        html_content = driver.page_source
        
        # Initialize BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # Find all <a> tags and filter with a lambda function for the one with 'Go to next page' in aria-label
        next_page_tags = soup.find_all(lambda tag: tag.name == 'a' and 'aria-label' in tag.attrs and 'Go to next page' in tag['aria-label'])

        # Extract the href attribute from the first matching tag and prepend with 'https://www.amazon.com'
        next_page_url = 'https://www.amazon.com' + next_page_tags[0]['href'] if next_page_tags else None

        return next_page_url
    except Exception as e:
        print('get_nextpage_url error: ', e)
        return None
        
'-------------------------------------------------------------------------------'

# Get Each Product's URL from page
def get_productpage_url(driver):
    try:
        # get the html        
        html_content = driver.page_source
        
        # Initialize BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # Find all divs with the 'data-index' attribute
        divs_with_data_index = soup.find_all('div', attrs={'data-index': True})


        urls = []
        # Loop through the found divs and extract the href attributes
        for div in divs_with_data_index:
            data_index = div['data-index']
            a_tag = div.find('a', class_='a-link-normal')
            if a_tag and 'href' in a_tag.attrs:
                full_url = 'https://www.amazon.com' + a_tag['href']
                if 'help/customer' in full_url:
                    continue
                if 'aax-us-iad.amazon.com' in full_url:
                    continue
                
                if type(full_url) == str:
                    urls.append([full_url])

        return urls
    
    except Exception as e:
        print('Get ProductPage Error: ', e)
        return []
    

'-------------------------------------------------------------------------------'

# Fetch Data from product page
def fetch_info_from_product_page(driver):
    try:
        # fetch whole html from url
        html_content = driver.page_source
        
        # Parse the HTML content
        soup = BeautifulSoup(html_content, 'html.parser')

        output = []
        # Get each information    
        
        # category
        breadcrumb_links = soup.select('#wayfinding-breadcrumbs_feature_div ul a')
        category = [html.unescape(link.get_text().strip()) for link in breadcrumb_links]
        output.append(category)
        
        # title
        title = soup.find('span', {'id': 'productTitle'}).get_text(strip=True)
        output.append(title)

        # price
        price_whole_part = soup.find('span', class_='a-price-whole')
        price_fraction_part = soup.find('span', class_='a-price-fraction')
        price = f'${price_whole_part.get_text(strip=True)}{price_fraction_part.get_text(strip=True)}' if price_whole_part and price_fraction_part else 'Price not Found'
        output.append(price)
        
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
            product_info = {}  # or None if you prefer None
        output.append(product_info)

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
        output.append(product_features)
        
        # product detail text
        about_section = soup.find('div', id='feature-bullets')
        list_items = about_section.find_all('span', class_='a-list-item') if about_section else []
        product_text = ' '.join(item.get_text(strip=True) for item in list_items)
        output.append(product_text)
        
        # product_url
        product_url = driver.current_url
        output.append(product_url)
        
        # img_url
        selected_imgs = soup.select('#imgTagWrapperId img, .imgTagWrapper img')
        for img in selected_imgs:
            image_url = img.get('data-old-hires', img.get('src'))  # Default to src if data-old-hires is not present
        output.append(image_url)
        
        # return
        return output  # output = [category, title, price, product info, product feature, product detail, product url, img url]

    except Exception as e:
        print('Fetch Info Error: ', e)        
        return []

def sanitize_filename(filename):
    # Keep only alphabets (uppercase and lowercase)
    return re.sub(r'[^a-zA-Z]', '', filename)

def save_img(name, title, image_url):
    try:
        # Send a GET request to the image URL
        response = requests.get(image_url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Create directory if it doesn't exist
        os.makedirs(f'imgs/{name}', exist_ok=True)
        
        # Sanitize title for filename
        sanitized_title = sanitize_filename(title[:200])
        file_path = f'imgs/{name}/{sanitized_title}.jpg'

        # Check if the image is already in JPEG format
        if image_url.lower().endswith('.jpg'):
            # Save image directly if it is already a JPEG
            with open(file_path, 'wb') as f:
                f.write(response.content)
            return True  # return True when Success
        else:
            # Convert and save in JPEG format if the image is in a different format
            img = Image.open(BytesIO(response.content))
            img.convert('RGB').save(file_path, 'JPEG')
            return True  # return True when Success
            
    except Exception as e:
        print(f"Error downloading image: {e}")
        return False  # return False when failure
    
'-------------------------------------------------------------------------------'

# configure selenium chrome driver
def configure_driver():
    try:
        # # Start the BrowserMob Proxy server
        # server = Server(r"C:\browsermob-proxy-2.1.4\bin\browsermob-proxy")
        # server.start()
        # proxy = server.create_proxy()
            
        # chromedriver_path = '/home/bae/.cache/selenium/chromedriver/linux64/120.0.6099.71/chromedriver'
        # possible paths
        # /home/bae/.cache/selenium/chromedriver
        # /home/bae/.cache/selenium/chromedriver/linux64/120.0.6099.71/chromedriver
        # /mnt/c/Users/user/.cache/selenium/chromedriver
        # service = Service(chromedriver_path)
        
        chrome_options = Options()
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 11.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.6099.110 Safari/537.36")
        # chrome_options.add_argument("--proxy-server={0}".format(proxy.proxy)) # Set up Selenium to use the proxy
        chrome_options.add_argument("--disable-gpu")
        # chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-software-rasterizer")    
        chrome_options.add_argument("disable-infobars")
        chrome_options.add_argument("--disable-extensions")
        # chrome_options.add_argument("--remote-debugging-port=9222")
        
        driver = webdriver.Chrome(options=chrome_options)
        
        # Create custom headers (including Referer)
        headers = {
            "Referer": "https://www.amazon.com/Living-Room-Furniture/b?node=3733551",
            # Add any other custom headers you want
        }
        # proxy.add_to_capabilities(headers)
        
        # Apply stealth settings    
        stealth(driver,
                languages=["en-US", "en"],  # List of languages
                vendor="Google Inc.",  # Vendor name
                platform="Win64",  # Platform
                webgl_vendor="Intel Inc.",  # WebGL vendor
                renderer="Intel Iris OpenGL Engine",  # WebGL renderer
                fix_hairline=True,  # Fix for thin lines issue
                )
        
        return True, driver
    
    except Exception as e:
        print('Error Configuring Driver: ',e)
        return False, driver

def safe_get(driver, url):
    max_retries = 5
    attempts = 0
    while attempts < max_retries:
        try:
            driver.get(url)
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
            # driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            time.sleep(random.randint(2000, 5000)/1000)
            
            # Find the captcha
            html_content = driver.page_source
            soup = BeautifulSoup(html_content, 'html.parser')
            captcha_text = soup.find('h4', string="Type the characters you see in this image:")
            if captcha_text:
                captcha_image = soup.find('img')
                captcha_image_url = captcha_image['src'] if captcha_image else None
                if captcha_image_url:
                    # solve captcha with amazoncaptcha library
                    captcha = AmazonCaptcha.fromlink(captcha_image_url)
                    solution = captcha.solve()
                    solution = list(solution)
                    # find captcha input box
                    captcha_input_field = driver.find_element(By.ID, "captchacharacters")
                    # type in captcha alphabets one by one just like human
                    for alphabets in solution:
                        captcha_input_field.send_keys(alphabets)
                        time.sleep(random.randint(100, 200)/1000)
                    time.sleep(random.randint(9000, 15000)/1000)  # sleep some for clicking submit
                    
                    # if 0 press enter, if 1 click button
                    is_Enter_Click = random.randint(0,1)
                    if is_Enter_Click == 0:
                        # press enter
                        captcha_input_field.send_keys(Keys.ENTER)
                    else:
                        # Locate the submit button using its class and the button text
                        submit_button = driver.find_element(By.XPATH, "//button[@class='a-button-text' and text()='Continue shopping']")
                        # Click the submit button
                        submit_button.click()
                    
                    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
                    # driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
                    time.sleep(random.randint(2000, 5000)/1000)
                                       
                    
            return True, driver  # Page load successful
        
        except (TimeoutException, WebDriverException) as e:
            print(f"Network error encountered: {e}. Retrying...")
            time.sleep(random.randint(2000, 5000)/1000)
            attempts += 1
       
        except Exception as e:
            print('Error in safe_get: ', e)
            time.sleep(random.randint(2000, 5000)/1000)
            attempts += 1

    return False, driver  # Page load failed

'-------------------------------------------------------------------------------'

Sofa_and_Couches = {
    # 'Sectional_Sofas' : 'https://www.amazon.com/s?bbn=3733551&rh=n%3A1055398%2Cn%3A%211063498%2Cn%3A1063306%2Cn%3A1063318%2Cn%3A3733551%2Cp_n_feature_two_browse-bin%3A3248836011&dc&fst=as%3Aoff&pf_rd_i=3733551&pf_rd_m=ATVPDKIKX0DER&pf_rd_p=877c0809-fc53-42a6-91f2-7c8f50e6e9be&pf_rd_r=0X2CMRE53H2HSQR09781&pf_rd_s=merchandised-search-2&pf_rd_t=101&qid=1528841766&rnid=3248834011&ref=s9_acss_bw_cg_HarSofa_2a1_w',
    # 'Sleeper_Sofas' : 'https://www.amazon.com/s?i=garden&bbn=3733551&rh=n%3A1055398%2Cn%3A%211063498%2Cn%3A1063306%2Cn%3A1063318%2Cn%3A3733551%2Cp_n_feature_two_browse-bin%3A3248838011&pf_rd_i=3733551&pf_rd_i=3733551&pf_rd_m=ATVPDKIKX0DER&pf_rd_m=ATVPDKIKX0DER&pf_rd_p=3009ccc5-2584-454e-b732-ae8e525543fe&pf_rd_p=877c0809-fc53-42a6-91f2-7c8f50e6e9be&pf_rd_r=0X2CMRE53H2HSQR09781&pf_rd_r=79FTK5EJP86JTXQETKYX&pf_rd_s=merchandised-search-2&pf_rd_s=merchandised-search-2&pf_rd_t=101&pf_rd_t=101&ref=s9_acss_bw_cg_HarSofa_2b1_w',
    'Reclining_Sofas' : 'https://www.amazon.com/s?i=garden&bbn=3733551&rh=n%3A1055398%2Cn%3A%211063498%2Cn%3A1063306%2Cn%3A1063318%2Cn%3A3733551%2Cp_n_feature_two_browse-bin%3A12012870011&pf_rd_i=3733551&pf_rd_i=3733551&pf_rd_m=ATVPDKIKX0DER&pf_rd_m=ATVPDKIKX0DER&pf_rd_p=3009ccc5-2584-454e-b732-ae8e525543fe&pf_rd_p=877c0809-fc53-42a6-91f2-7c8f50e6e9be&pf_rd_r=0X2CMRE53H2HSQR09781&pf_rd_r=79FTK5EJP86JTXQETKYX&pf_rd_s=merchandised-search-2&pf_rd_s=merchandised-search-2&pf_rd_t=101&pf_rd_t=101&ref=s9_acss_bw_cg_HarSofa_2c1_w',
#     'LoveSeats' : 'https://www.amazon.com/s?i=garden&bbn=3733551&rh=n%3A1055398%2Cn%3A%211063498%2Cn%3A1063306%2Cn%3A1063318%2Cn%3A3733551%2Cp_n_feature_two_browse-bin%3A3248835011&pf_rd_i=3733551&pf_rd_i=3733551&pf_rd_m=ATVPDKIKX0DER&pf_rd_m=ATVPDKIKX0DER&pf_rd_p=3009ccc5-2584-454e-b732-ae8e525543fe&pf_rd_p=877c0809-fc53-42a6-91f2-7c8f50e6e9be&pf_rd_r=79FTK5EJP86JTXQETKYX&pf_rd_r=QZ6K0G39A8CCYDMBWJRD&pf_rd_s=merchandised-search-2&pf_rd_s=merchandised-search-2&pf_rd_t=101&pf_rd_t=101&ref=s9_acss_bw_cg_HarSofa_2d1_w',
#     'Futons' : 'https://www.amazon.com/Futons/b/ref=s9_acss_bw_cg_SofaType_1f1_w/ref=s9_acss_bw_cg_HarSofa_3a1_w?ie=UTF8&node=13753041&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-2&pf_rd_r=79FTK5EJP86JTXQETKYX&pf_rd_t=101&pf_rd_p=3009ccc5-2584-454e-b732-ae8e525543fe&pf_rd_i=3733551&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-2&pf_rd_r=QZ6K0G39A8CCYDMBWJRD&pf_rd_t=101&pf_rd_p=877c0809-fc53-42a6-91f2-7c8f50e6e9be&pf_rd_i=3733551',
#     'Settles' : 'https://www.amazon.com/s?i=garden&bbn=3733551&rh=n%3A1055398%2Cn%3A%211063498%2Cn%3A1063306%2Cn%3A1063318%2Cn%3A3733551%2Cp_n_feature_two_browse-bin%3A3248837011&pf_rd_i=3733551&pf_rd_i=3733551&pf_rd_m=ATVPDKIKX0DER&pf_rd_m=ATVPDKIKX0DER&pf_rd_p=3009ccc5-2584-454e-b732-ae8e525543fe&pf_rd_p=877c0809-fc53-42a6-91f2-7c8f50e6e9be&pf_rd_r=79FTK5EJP86JTXQETKYX&pf_rd_r=QZ6K0G39A8CCYDMBWJRD&pf_rd_s=merchandised-search-2&pf_rd_s=merchandised-search-2&pf_rd_t=101&pf_rd_t=101&ref=s9_acss_bw_cg_HarSofa_3b1_w',
#     'Convertibles' : 'https://www.amazon.com/s?bbn=3733551&rh=n%3A1055398%2Cn%3A%211063498%2Cn%3A1063306%2Cn%3A1063318%2Cn%3A3733551%2Cp_n_feature_two_browse-bin%3A12012869011&dc&fst=as%3Aoff&pf_rd_i=3733551&pf_rd_m=ATVPDKIKX0DER&pf_rd_p=877c0809-fc53-42a6-91f2-7c8f50e6e9be&pf_rd_r=QZ6K0G39A8CCYDMBWJRD&pf_rd_s=merchandised-search-2&pf_rd_t=101&qid=1528765569&rnid=3248834011&ref=s9_acss_bw_cg_HarSofa_3c1_w',
}

# Other_Livingroom_Furniture = {
#     'Accent_Chairs' : 'https://www.amazon.com/b?node=3733491&ref=s9_acss_bw_cg_SBR2019_3b1_w', # 42 pages
#     'Coffee_Tables' : 'https://www.amazon.com/b?node=3733631&ref=s9_acss_bw_cg_SBR2019_3c1_w', # 35 pages
#     'TV_Stands' : 'https://www.amazon.com/b?node=14109851&ref=s9_acss_bw_cg_SBR2019_3d1_w',
#     'End_Tables' : 'https://www.amazon.com/b?node=3733641&ref=s9_acss_bw_cg_SBR2019_4a1_w',
#     'Console_Tables' : 'https://www.amazon.com/b?node=3733651&ref=s9_acss_bw_cg_SBR2019_4b1_w',
#     'Ottomans' : 'https://amazon.com/b/ref=sv_hg_fl_3254639011/ref=s9_acss_bw_cg_SBR2019_4c1_w?ie=UTF8&node=3254639011&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-3&pf_rd_r=HH88E0EETR0MQZ658N3D&pf_rd_t=101&pf_rd_p=f126b4c9-d5f7-41d5-85e0-9db2657ec29e&pf_rd_i=14544497011',
#     'Living_Room_Sets' : 'https://www.amazon.com/b?node=3733481&ref=s9_acss_bw_cg_SBR2019_4d1_w',
# }

# Decor_and_Soft_Furnishings = {
#     'Decorative_Pillows' : 'https://www.amazon.com/s?rh=n%3A3732321&fs=true&ref=lp_3732321_sar', # 161 pages
#     'Throw_Blankets' : 'https://www.amazon.com/s?rh=n%3A14058581&fs=true&ref=lp_14058581_sar', # 400 pages
#     'Area_Rugs' : 'https://www.amazon.com/s?rh=n%3A684541011&fs=true&ref=lp_684541011_sar', # 400 pages
#     'Wall_Arts' : 'https://www.amazon.com/s?rh=n%3A3736081&fs=true&ref=lp_3736081_sar' , # 400 pages
#     'Table_Lamps' : 'https://www.amazon.com/b?node=1063296&ref=s9_acss_bw_cg_SBR2019_7a1_w', # 347 pages
#     'Floor_Lamps' : 'https://www.amazon.com/b?node=1063294&ref=s9_acss_bw_cg_SBR2019_7b1_w', # 131 pages
#     'Pendants_and_Chandeliers' : 'https://www.amazon.com/lighting-ceiling-fans/b/ref=s9_acss_bw_cg_SBR2019_7c1_w?ie=UTF8&node=495224&ref_=sv_hg_5&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-3&pf_rd_r=HH88E0EETR0MQZ658N3D&pf_rd_t=101&pf_rd_p=f126b4c9-d5f7-41d5-85e0-9db2657ec29e&pf_rd_i=14544497011', #298 pages
#     'Sconces' : 'https://www.amazon.com/b?node=3736721&ref=s9_acss_bw_cg_SBR2019_7d1_w', # 238 pages
#     'Baskets_and_Storage' : 'https://www.amazon.com/s?rh=n%3A2422430011&fs=true&ref=lp_2422430011_sar', # 244 pages
#     'Candles' : 'https://www.amazon.com/s?rh=n%3A3734391&fs=true&ref=lp_3734391_sar', # 400 pages
#     'Live_Plants' : 'https://www.amazon.com/b/ref=sv_hg_fl_553798/ref=s9_acss_bw_cg_SBR2019_8c1_w?node=3480662011&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-3&pf_rd_r=HH88E0EETR0MQZ658N3D&pf_rd_t=101&pf_rd_p=f126b4c9-d5f7-41d5-85e0-9db2657ec29e&pf_rd_i=14544497011', # 14 pages
#     'Artificial_Plants' : 'https://www.amazon.com/b?node=14087351&ref=s9_acss_bw_cg_SBR2019_8d1_w', # 400 pags    
#     'Planters' : 'https://www.amazon.com/b?node=553798&ref=s9_acss_bw_cg_SBR2019_9a1_w', # 263 pages
#     'Decorative_Accessories' : 'https://www.amazon.com/s?rh=n%3A3295676011&fs=true&ref=lp_3295676011_sar', # 400 pages
#     'Window_Coverings' : 'https://www.amazon.com/b?node=1063302&ref=s9_acss_bw_cg_SBR2019_9c1_w', # 400 pages
#     'Decorative_Mirrors' : 'https://www.amazon.com/b?node=3736371&ref=s9_acss_bw_cg_SBR2019_9d1_w', # 314 pages
# }

# Kitchen_and_Dining_Furnitures = {
#     'Dining_Sets' : 'https://www.amazon.com/b/ref=s9_acss_bw_cg_HarFK_2a1_w?node=8566630011&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-2&pf_rd_r=RDTFVCNEA5ZFMYQZYZ3E&pf_rd_t=101&pf_rd_p=fde2e008-b6c2-4e44-9488-63f47804b655&pf_rd_i=3733781',
#     'Dining_Tables' : 'https://www.amazon.com/b/ref=s9_acss_bw_cg_HarFK_2b1_w?node=3733811&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-2&pf_rd_r=RDTFVCNEA5ZFMYQZYZ3E&pf_rd_t=101&pf_rd_p=fde2e008-b6c2-4e44-9488-63f47804b655&pf_rd_i=3733781',
#     'Dining_Chairs' : 'https://www.amazon.com/b/ref=s9_acss_bw_cg_HarFK_2c1_w?node=3733821&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-2&pf_rd_r=RDTFVCNEA5ZFMYQZYZ3E&pf_rd_t=101&pf_rd_p=fde2e008-b6c2-4e44-9488-63f47804b655&pf_rd_i=3733781',
#     'Bar_Stools' : 'https://www.amazon.com/b/ref=s9_acss_bw_cg_HarFK_2d1_w?node=3733851&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-2&pf_rd_r=RDTFVCNEA5ZFMYQZYZ3E&pf_rd_t=101&pf_rd_p=fde2e008-b6c2-4e44-9488-63f47804b655&pf_rd_i=3733781',
#     'Kitchen_Islands' : 'https://www.amazon.com/b/ref=s9_acss_bw_cg_HarFK_3a1_w?node=8521400011&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-2&pf_rd_r=RDTFVCNEA5ZFMYQZYZ3E&pf_rd_t=101&pf_rd_p=fde2e008-b6c2-4e44-9488-63f47804b655&pf_rd_i=3733781',
#     'Buffets_and_Sideboards' : 'https://www.amazon.com/b/ref=s9_acss_bw_cg_HarFK_3b1_w?node=3733831&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-2&pf_rd_r=RDTFVCNEA5ZFMYQZYZ3E&pf_rd_t=101&pf_rd_p=fde2e008-b6c2-4e44-9488-63f47804b655&pf_rd_i=3733781',
#     'China_Cabinets' : 'https://www.amazon.com/b/ref=s9_acss_bw_cg_HarFK_3c1_w?node=3733841&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-2&pf_rd_r=RDTFVCNEA5ZFMYQZYZ3E&pf_rd_t=101&pf_rd_p=fde2e008-b6c2-4e44-9488-63f47804b655&pf_rd_i=3733781',
#     'Bakers_Recks' : 'https://www.amazon.com/s?rh=n%3A3744061&fs=true&ref=lp_3744061_sar',
# }

# Bedroom_Furnitures = {
#     'Bedroom_Sets' : 'https://www.amazon.com/b/ref=s9_acss_bw_cg_HarFLiv_2a1_w?node=3732931&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-2&pf_rd_r=MNGW2YC86GM462Z294K6&pf_rd_t=101&pf_rd_p=d8003c9c-74d3-490c-ae40-6aac6ed45f61&pf_rd_i=1063308',
#     'Mattresses' : 'https://www.amazon.com/s?rh=n%3A3732961&fs=true&ref=lp_3732961_sar',
#     'Nightstands' : 'https://www.amazon.com/b/ref=s9_acss_bw_cg_HarFLiv_2c1_w?node=3733251&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-2&pf_rd_r=MNGW2YC86GM462Z294K6&pf_rd_t=101&pf_rd_p=d8003c9c-74d3-490c-ae40-6aac6ed45f61&pf_rd_i=1063308',
#     'Dressers' : 'https://www.amazon.com/b/ref=s9_acss_bw_cg_HarFLiv_2d1_w?node=3733261&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-2&pf_rd_r=MNGW2YC86GM462Z294K6&pf_rd_t=101&pf_rd_p=d8003c9c-74d3-490c-ae40-6aac6ed45f61&pf_rd_i=1063308',
#     'Beds' : 'https://www.amazon.com/b/ref=s9_acss_bw_cg_HarFLiv_3a1_w?node=3248804011&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-2&pf_rd_r=MNGW2YC86GM462Z294K6&pf_rd_t=101&pf_rd_p=d8003c9c-74d3-490c-ae40-6aac6ed45f61&pf_rd_i=1063308',
#     'Bedframes' : 'https://www.amazon.com/b/ref=s9_acss_bw_cg_HarFLiv_3b1_w?node=3248801011&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-2&pf_rd_r=MNGW2YC86GM462Z294K6&pf_rd_t=101&pf_rd_p=d8003c9c-74d3-490c-ae40-6aac6ed45f61&pf_rd_i=1063308',
#     'Bases' : 'https://www.amazon.com/s?rh=n%3A17873917011&language=en_US&brr=1&pf_rd_i=1063308&pf_rd_m=ATVPDKIKX0DER&pf_rd_p=d8003c9c-74d3-490c-ae40-6aac6ed45f61&pf_rd_r=MNGW2YC86GM462Z294K6&pf_rd_s=merchandised-search-2&pf_rd_t=101&rd=1',
#     'Vanities' : 'https://www.amazon.com/b/ref=s9_acss_bw_cg_HarFLiv_3d1_w?node=3733291&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-2&pf_rd_r=MNGW2YC86GM462Z294K6&pf_rd_t=101&pf_rd_p=d8003c9c-74d3-490c-ae40-6aac6ed45f61&pf_rd_i=1063308',    
# }

# Entryway_Furnitures = {
#     'Entryway_Furnitures' : 'https://www.amazon.com/b/ref=s9_acss_bw_cg_FNNAV_5e1_w?node=3249856011&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-7&pf_rd_r=AMH2XC97ZGGNAWND0C03&pf_rd_t=101&pf_rd_p=d0c31dca-221f-4a26-9f62-a9e18ca9f354&pf_rd_i=1063306',
# }

# Home_Office_Furnitures = {
#     'Desks' : 'https://www.amazon.com/b/ref=s9_acss_bw_cg_HarHOF_2a1_w?node=3733671&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-2&pf_rd_r=S6B16AQ2FYJDBJXK6X9C&pf_rd_t=101&pf_rd_p=b0e12b53-6018-44fe-b7a4-d1871cf08ee9&pf_rd_i=1063312',
#     'Desk_Chairs' : 'https://www.amazon.com/b/ref=s9_acss_bw_cg_HarHOF_2b1_w?node=3733721&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-2&pf_rd_r=S6B16AQ2FYJDBJXK6X9C&pf_rd_t=101&pf_rd_p=b0e12b53-6018-44fe-b7a4-d1871cf08ee9&pf_rd_i=1063312',
#     'Bookcases' : 'https://www.amazon.com/b/ref=s9_acss_bw_cg_HarHOF_2c1_w?node=10824421&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-2&pf_rd_r=S6B16AQ2FYJDBJXK6X9C&pf_rd_t=101&pf_rd_p=b0e12b53-6018-44fe-b7a4-d1871cf08ee9&pf_rd_i=1063312',
#     'File_Cabinets' : 'https://www.amazon.com/b/ref=s9_acss_bw_cg_HarHOF_2d1_w?node=1069166&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-2&pf_rd_r=S6B16AQ2FYJDBJXK6X9C&pf_rd_t=101&pf_rd_p=b0e12b53-6018-44fe-b7a4-d1871cf08ee9&pf_rd_i=1063312',
#     'Computer_Armoires' : 'https://www.amazon.com/s?rh=n%3A3733751&fs=true&ref=lp_3733751_sar',
#     'Drafting_Tables' : 'https://www.amazon.com/s?rh=n%3A3733771&fs=true&ref=lp_3733771_sar',
#     'Cabinets' : 'https://www.amazon.com/s?rh=n%3A3733761&fs=true&ref=lp_3733761_sar',
#     'Furniture_Sets' : 'https://www.amazon.com/b/ref=s9_acss_bw_cg_HarHOF_3d1_w?node=3733661&pf_rd_m=ATVPDKIKX0DER&pf_rd_s=merchandised-search-2&pf_rd_r=S6B16AQ2FYJDBJXK6X9C&pf_rd_t=101&pf_rd_p=b0e12b53-6018-44fe-b7a4-d1871cf08ee9&pf_rd_i=1063312',
# }

dictionaries = [
    Sofa_and_Couches, 
    # Other_Livingroom_Furniture, 
    # Decor_and_Soft_Furnishings, 
    # Kitchen_and_Dining_Furnitures,
    # Bedroom_Furnitures, 
    # Entryway_Furnitures, 
    # Home_Office_Furnitures,
]

'-------------------------------------------------------------------------------'

def scrape_infos(name, url):
    
    # initialize statue log txt file            
    status_log_txt = 'crawling_status.txt'    
    
    # Start
    next_operation = "start"
    with open(status_log_txt, 'w', encoding='utf-8') as file:
        file.write(next_operation + '\n')
        file.write(url + '\n')
        
    # Each csv files
    csv_file_path1 = f'page_urls/{name}_page_urls.csv'  
    csv_file_path2 = f'product_urls/{name}_product_urls.csv'
    csv_file_path3 = f'product_infos/{name}_product_infos.csv'
    csv_file_path4 = f'error_urls/no_product_page_urls.csv'
    csv_file_path5 = f'error_urls/no_info_product_urls.csv'
    csv_file_path6 = f'error_urls/no_next_page_urls.csv'
    csv_file_path7 = f'error_urls/no_img_product_urls.csv'
    
    # Create directory if it doesn't exist
    os.makedirs(f'page_urls', exist_ok=True)
    os.makedirs(f'product_urls', exist_ok=True)
    os.makedirs(f'product_infos', exist_ok=True)
    os.makedirs(f'error_urls', exist_ok=True)
    
    # Headers for each csv
    headers_needed1 = not os.path.exists(csv_file_path1)    
    headers_needed2 = not os.path.exists(csv_file_path2)
    headers_needed3 = not os.path.exists(csv_file_path3)
    headers_needed4 = not os.path.exists(csv_file_path4)
    headers_needed5 = not os.path.exists(csv_file_path5)
    headers_needed6 = not os.path.exists(csv_file_path6)
    headers_needed7 = not os.path.exists(csv_file_path7)
    
    if headers_needed1:
        with open(csv_file_path1, mode='a', newline='', encoding='utf-8') as file1:
            writer1 = csv.writer(file1)
            writer1.writerow(['Page URL'])
            file1.flush()
    if headers_needed2:
        with open(csv_file_path2, mode='a', newline='', encoding='utf-8') as file2:
            writer2 = csv.writer(file2)
            writer2.writerow(['Product URL'])
            file2.flush()
    if headers_needed3:
        with open(csv_file_path3, mode='a', newline='', encoding='utf-8') as file3:
            writer3 = csv.writer(file3)
            writer3.writerow(['Category' , 'Title', 'Price', 'Product_Info', 'Product_Feature', 'Product_Text', 'Product URL', 'Img_URL'])
            file3.flush()  
    if headers_needed4:
        with open(csv_file_path4, mode='a', newline='', encoding='utf-8') as file4:
            writer4 = csv.writer(file4)
            writer4.writerow(['Error page URLs'])
            file4.flush()  
    if headers_needed5:
        with open(csv_file_path5, mode='a', newline='', encoding='utf-8') as file5:
            writer5 = csv.writer(file5)
            writer5.writerow(['Error product URLs'])
            file5.flush()          
    if headers_needed6:
        with open(csv_file_path6, mode='a', newline='', encoding='utf-8') as file6:
            writer6 = csv.writer(file6)
            writer6.writerow(['No next page URLs'])
            file6.flush()          
    if headers_needed7:
        with open(csv_file_path7, mode='a', newline='', encoding='utf-8') as file7:
            writer7 = csv.writer(file7)
            writer7.writerow(['image get error product URLs'])
            file7.flush() 
     
    # 1. Open page
    # 2. Get each product urls 
    # 3. Open each product urls, save infos, save img
    # 4. Go to next page
    # Repeat 1~4.
    
    # Configure driver
    next_operation = 'configure driver'  
    with open(status_log_txt, 'w', encoding='utf-8') as file:
        file.write(next_operation + '\n')
        file.write(url + '\n')
    is_Configured, driver = configure_driver()
    
    # error in configuring driver
    if is_Configured == False:
        while is_Configured:
            print('Re-configuring Driver...')
            is_Configured, driver = configure_driver()  # configure driver over and over again
             
    # 1. Open page
    next_operation = "open page"
    with open(status_log_txt, 'w', encoding='utf-8') as file:
        file.write(next_operation + '\n')
        file.write(url + '\n')
    is_PageOpened, driver = safe_get(driver, url) 
    
    # error in opening first page
    if is_PageOpened == False:
        print(f"Failed to load initial page: {url}")
        return  # end when Failed to load initial page
    
    # repeat 1~4. loop
    is_Done = False
    while not is_Done:  
        # log current page url
        current_page_url = driver.current_url
        with open(csv_file_path1, mode='a', newline='', encoding='utf-8') as file1:  # page_urls/{name}_page_urls.csv
            writer1 = csv.writer(file1)
            writer1.writerow([current_page_url])
            file1.flush()  # Force writing to CSV file
            
        next_page_url = get_nextpage_url(driver)  # returns None when fail to get next page
    
        # 2. get each product urls
        next_operation = "extract products url"
        with open(status_log_txt, 'w', encoding='utf-8') as file:
            file.write(next_operation + '\n')
            file.write(current_page_url + '\n')
        product_urls = get_productpage_url(driver)  # returns [] when fail to get productpage urls / returns [[url1],[url2],...]
        
        # if no product on page (error)
        if len(product_urls) == 0:
            with open(csv_file_path4, mode='a', newline='', encoding='utf-8') as file4:  # no_product_page_urls.csv'
                writer4 = csv.writer(file4)
                writer4.writerow([current_page_url])
                file4.flush()  
            return  # end when fail to get product urls from list page
        
        # log product urls
        with open(csv_file_path2, mode='a', newline='', encoding='utf-8') as file2:  # product_urls/{name}_product_urls.csv
            writer2 = csv.writer(file2)   
            writer2.writerows(product_urls)
            file2.flush()     
        
        # 3. open each product urls, save infos
        for [product_url] in product_urls:
            time.sleep(random.randint(9000, 15000)/1000)  # sleep some time between each product / not solving multiple captcah right away
            
            # open each product page
            next_operation =  'open product page'
            with open(status_log_txt, 'w', encoding='utf-8') as file:
                file.write(next_operation + '\n')
                file.write(current_page_url + '\n')
                file.write(product_url + '\n')
            is_ProductPageOpened, driver = safe_get(driver, product_url)

            # error opening product page
            if is_ProductPageOpened == False:
                print(f"Failed to load product page: {product_url}")
                # Handled later
                # if ProductPageOpened == False, no product_info will be fetched. so it will be logged as fail to fetch info.
            
            # extract infos
            next_operation = 'get infos'
            with open(status_log_txt, 'w', encoding='utf-8') as file:
                file.write(next_operation + '\n')
                file.write(current_page_url + '\n')
                file.write(product_url + '\n')            
            product_infos = fetch_info_from_product_page(driver)  # returns [] when fails to fetch infos

            # product infos well extracted   
            if len(product_infos) != 0:
                # save product infos
                next_operation = 'log infos'
                with open(status_log_txt, 'w', encoding='utf-8') as file:
                    file.write(next_operation + '\n')
                    file.write(current_page_url + '\n')
                    file.write(product_url + '\n')
                with open(csv_file_path3, mode='a', newline='', encoding='utf-8') as file3:  # product_infos/{name}_product_infos.csv
                    writer3 = csv.writer(file3)
                    writer3.writerow(product_infos)
                    file3.flush()  # Force writing to CSV file
                # save product img
                title, image_url = product_infos[1], product_infos[-1]
                next_operation = 'save img'
                with open(status_log_txt, 'w', encoding='utf-8') as file:
                    file.write(next_operation + '\n')
                    file.write(current_page_url + '\n')
                    file.write(image_url + '\n')
                is_ImgSaved = save_img(name, title, image_url)
                
                # error saving img
                if is_ImgSaved == False:
                    # log that product url for later retrying
                    with open(csv_file_path7, mode='a', newline='', encoding='utf-8') as file7:  # error_urls/no_img_product_urls.csv
                        writer7 = csv.writer(file7)
                        writer7.writerow([product_url])
                        file7.flush()

            # product info extraction failed
            else:
                with open(csv_file_path5, mode='a', newline='', encoding='utf-8') as file5:  # error_urls/no_info_product_urls.csv
                    writer5 = csv.writer(file5)
                    writer5.writerow([product_url])
                    file5.flush()  # Force writing to CSV file

        product_urls = []  # make product_urls list empty when iteration ended.
        
        # 4. go to next page
        
        # go back to list page
        next_operation = 'go back to page'
        with open(status_log_txt, 'w', encoding='utf-8') as file:
            file.write(next_operation + '\n')
            file.write(current_page_url + '\n')
        is_BeforePageOpened, driver = safe_get(driver, current_page_url)
        
        # error opening before list page
        if is_BeforePageOpened == False:
            print(f"Failed to load before page: {current_page_url}")
            # Handled later
            # it will try to click next page, but will fail -> handled as is_NextPageClicked == False
            
        # click next page
        next_operation = 'click next page'
        with open(status_log_txt, 'w', encoding='utf-8') as file:
            file.write(next_operation + '\n')
            file.write(current_page_url + '\n')
        is_NextPageClicked, driver = click_next_page(driver)

        # error clicking NextPage
        if is_NextPageClicked == False:
            print(f'Failed to click next page: {current_page_url}')
            with open(csv_file_path6, mode='a', newline='', encoding='utf-8') as file6:  # no_next_page_urls.csv
                writer6 = csv.writer(file6)
                writer6.writerow([current_page_url])
                file6.flush()  # Force writing to CSV file
                
            # if there is next_page_url
            if next_page_url:
                # no "next page" icon to click, directly re-open next page with link
                print('Trying to directly open next page')
                next_operation = 'directly open next page'
                with open(status_log_txt, 'w', encoding='utf-8') as file:
                    file.write(next_operation + '\n')
                    file.write(current_page_url + '\n')
                # re-open the chrome
                driver.quit()
                is_Configured, driver = configure_driver()
                
                # error in configuring driver
                if is_Configured == False:
                    while is_Configured:
                        print('Re-configuring Driver...')
                        is_Configured, driver = configure_driver()  # configure driver over and over again
            
                # open next page directly
                is_NextPageOpened, driver = safe_get(driver, next_page_url)
                if is_NextPageOpened == False:
                    print(f'Failed to diretly open next page: {current_page_url}')
                    return  # end when Failed to load next page when there exist next page
                else: 
                    print(f"Success to open next page")
        
            # no next_page_url and click_next_page fails == Done with current category
            else:
                is_Done = True
            
    # go to next category
    next_operation = 'each start url finished'
    # Remove the file from the filesystem
    if os.path.exists(status_log_txt):
        os.remove(status_log_txt)
    print(f'{name} done. Moving to next category')

'-------------------------------------------------------------------------------'


if __name__ == "__main__":
    for each_dictionary in dictionaries:
        for name, url in each_dictionary.items():
            
            status_log_txt = 'crawling_status.txt'
            last_operation = None
            
            # Check if status file exists and read the last operation and URL
            if os.path.exists(status_log_txt):
                with open(status_log_txt, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                    if len(lines) >= 2:
                        last_operation = lines[0].strip()
                        last_url = lines[1].strip()
            
                # last_operation = 'start', 'configure driver', "open page", "extract products url", 'open product page',
                #                   'get infos', 'log infos', 'save img', 'go back to page', 'click next page',
                #                   'each start url finished'
            
            if last_operation:  # if there is last_operation
                url = last_url
                scrape_infos(name, url)
            
            else:  # if just starting
                scrape_infos(name, url)
                