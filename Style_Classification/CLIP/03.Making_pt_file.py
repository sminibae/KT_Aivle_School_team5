# imports
from re import M
import torch
import clip
import open_clip
from PIL import Image

import pandas as pd
import numpy as np

import os
import datetime

from tqdm import tqdm


'----------------------------------------------------------------------------------------------------------'


# df chunk 별로 따로 일하기
def process_text(df_chunk, tokenizer, model):
    all_tensors = []
    for index, row in df_chunk.iterrows():
        # Process text
        try:
            text = row['Product_Text']
            if isinstance(text, str) and text.strip():
                text_input = tokenizer(row['Product_Text'])
                with torch.no_grad():
                    text_features = model.encode_text(text_input)
            else:
                text_features = torch.zeros(1,768)

        except:
            text_features = torch.zeros(1, 768)
        
        all_tensors.append(text_features)
    
    return torch.cat(all_tensors, dim=0)

def process_img(df_chunk, model, preprocess):
    all_tensors = []
    for index, row in df_chunk.iterrows():
        try:
            # Process image
            image_path = row['img_path']
            image = preprocess(Image.open(image_path)).unsqueeze(0)
            image_features = model.encode_image(image)
        except Exception as e:
            print(f'Error processing image at row {index}: {e}')
            image_features = torch.zeros(1, 768)
        
        all_tensors.append(image_features)
            
    return torch.cat(all_tensors, dim=0)


'----------------------------------------------------------------------------------------------------------'


def main_text(df, batch_size=1000):
    # import tokenizer
    tokenizer = open_clip.get_tokenizer('ViT-L-14-336')
    print('tokenizer import success')
    
    # importing model, preprocess
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14-336', pretrained='openai')
    print('model import success')
    
    # Calculate the number of batches
    num_batches = len(df) // batch_size + int(len(df) % batch_size != 0)
    
    for batch_num in range(num_batches):
        # Define start and end of the batch
        start_idx = batch_num * batch_size
        end_idx = start_idx + batch_size
        
        # Slice the dataframe to get the batch
        df_batch = df.iloc[start_idx:end_idx]
        # print(f'Batch {batch_num} start processing')
        
        # Process the batch
        text_output = process_text(df_batch, tokenizer, model)
        
        # Save the batch output
        torch.save(text_output, f'pt_files/CLIP_text_{batch_num}.pt')
        
        if batch_num%1000 == 0 :
            current_time = datetime.datetime.now().strftime('%H:%M:%S')
            print(f'Text processed {batch_num}/{len(df)} at {current_time}')
    
    print('Processed all text')
    return
        

def main_img(df, batch_size=1000):
    # importing model, preprocess
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14-336', pretrained='openai')
    print('model import success')
    
    # Calculate the number of batches
    num_batches = len(df) // batch_size + int(len(df) % batch_size != 0)
        
    for batch_num in range(num_batches):
        # Define start and end of the batch
        start_idx = batch_num * batch_size
        end_idx = start_idx + batch_size
        
        # Slice the dataframe to get the batch
        df_batch = df.iloc[start_idx:end_idx]
        # print(f'Batch {batch_num} start processing')
        
        # Process the batch
        text_output = process_img(df_batch, model, preprocess)
        
        # Save the batch output
        torch.save(text_output, f'pt_files/CLIP_img_{batch_num}.pt')
        
        if batch_num%1000 == 0 :
            current_time = datetime.datetime.now().strftime('%H:%M:%S')
            print(f'Img processed {batch_num}/{len(df)} at {current_time}')
    
    print('Processed all img')
    return


import torch
import os
def main_concat():
    # Define the directory where the files are stored
    directory = 'pt_files/'

    # Initialize a list to hold all concatenated tensors
    all_tensors = []

    # List all files in the directory and sort them to maintain order
    file_list = os.listdir(directory)
    file_list.sort()  # This helps to keep the order

    # Loop through the range of file indices
    for i in range(len(file_list) // 2):  # Assuming pairs of img and text files
        img_tensor = torch.load(f"{directory}CLIP_img_{i}.pt")
        text_tensor = torch.load(f"{directory}CLIP_text_{i}.pt")
        
        # Concatenating along the second dimension (assuming first dimension is batch size)
        concat_tensor = torch.cat((img_tensor, text_tensor), dim=1)
        all_tensors.append(concat_tensor)

    # Concatenate all tensors along the first dimension (stacking them)
    all_tensors = torch.cat(all_tensors, dim=0)

    # Save the final concatenated tensor
    torch.save(all_tensors, "CLIP_tensors.pt")


'----------------------------------------------------------------------------------------------------------'
 
 
if __name__ == "__main__":
    combined_df = pd.read_csv('combined_df.csv')
    print('df import success')
    os.makedirs('pt_files', exist_ok= True)
    
    np.save('CLIP_style.npy', np.array(combined_df['Style']),allow_pickle=True)
    
    main_img(combined_df, batch_size=1)
    main_text(combined_df, batch_size=1)
    main_concat()
    

    