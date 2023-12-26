Data Preprocessing
    For text, used CLIP
        tokenizer = open_clip.get_tokenizer('ViT-L-14-336')

    For image, used CLIP
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14-336', pretrained='openai')


    For integration, just concat text and image vectors.

Modeling
    just used SimpleNN, ML methods