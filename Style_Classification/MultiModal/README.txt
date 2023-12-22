Data Preprocessing
    For text, used BERT
        BertTokenizer.from_pretrained('bert-large-uncased')
        BertModel.from_pretrained('bert-large-uncased')

    For image, used transformer from torchvision

    For integration, just concat text and image vectors.

Modeling
    just used SimpleNN, CNN.