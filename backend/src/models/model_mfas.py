import torch
torch.manual_seed(42)
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import BertModel, BertTokenizer, BertConfig
transformers.logging.set_verbosity_error()
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
import requests
from io import BytesIO

result_converter = {1: 'Anti-Vaxx', 0: 'Pro-Vaxx'}

class multiModel(nn.Module):
  
    def __init__(self, num_labels=2, config=None, device=torch.device("cuda:0")):
        super(multiModel, self).__init__()
        
        # Common layers
        self.bn = nn.BatchNorm1d(1536, momentum=0.99)
        self.dense1 = nn.Linear(in_features=1536, out_features=768) #Add ReLu in forward loop
        self.dropout = nn.Dropout(p=0.2)
        self.dense2 = nn.Linear(in_features=768, out_features=num_labels)
        self.device = device
        
        # Txt model
        self.txt_model = BertModel.from_pretrained('bert-base-uncased', config=config)

        self.img_model = torchvision.models.densenet161(pretrained=True)
        num_ftrs = self.img_model.classifier.in_features
        self.img_model.classifier = nn.Linear(num_ftrs, 768)

    def forward(self, inputs, imgs, attention_mask=None, labels=None):

        text_input_ids_in = inputs[:,0,:].long()
        text_input_masks_in = inputs[:,1,:].long()
        
        text_input_token_in = inputs[:,2,:].long()
        text_embedding_layer = self.txt_model(text_input_ids_in, attention_mask=text_input_masks_in, token_type_ids=text_input_token_in)[0]
        
        text_cls_token = text_embedding_layer[:,0,:]
        
        img_features = self.img_model(imgs)
        
        combined_features = torch.cat((text_cls_token, img_features), 1) # 2304 x 1
        
        
        X = self.bn(combined_features)
        X = F.relu(self.dense1(X))
        X = self.dropout(X)
        X = F.log_softmax(self.dense2(X))
        
        return X

def make_bert_input_w_OCR(text_data, ocr_data, max_len):
    # For every sentence...
    input_ids = []
    attention_masks = []
    token_ids = []
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True, add_special_tokens=True, max_length=100, pad_to_max_length=True)
    for i in range(len(text_data)):
        encoded_dict = tokenizer.encode_plus(
                            text_data[i],                      # Sentence to encode.
                            text_pair = ocr_data[i],
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = max_len,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_token_type_ids = True,
                       )

        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

        token_ids.append(encoded_dict['token_type_ids'])

    input_ids = np.asarray(input_ids, dtype='int32')
    attention_masks = np.asarray(attention_masks, dtype='int32')
    token_ids = np.asarray(token_ids, dtype='int32')

    return input_ids, attention_masks, token_ids

def init_mfas():
    model_path = './src/models/model_weights/MFAS_model_fold_1.h5'
    config = BertConfig.from_pretrained('bert-base-uncased', num_labels=2)
    model = multiModel(num_labels=2, config=config)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def run_mfas_inference(tweet_text, img_url, model):

    img_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    response = requests.get(img_url)
    pillow_img = Image.open(BytesIO(response.content)).convert('RGB')

    img = img_transform(pillow_img)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True, add_special_tokens=True, max_length=100, pad_to_max_length=True)
    # text_data = ["Sorry to inform you 'vaccinated' folks, You ARE the experiment!"]
    text_data = [tweet_text]
    input_id, attn_mask, token_id = make_bert_input_w_OCR(text_data, [""], 100)
    text_inp = np.vstack((np.vstack((input_id, attn_mask)), token_id))

    inputs = torch.Tensor(np.expand_dims(text_inp, axis=0))
    imgs = torch.Tensor(np.expand_dims(img, axis=0))
    outputs = model(inputs, imgs)
    preds = torch.max(outputs, 1)[1]

    return result_converter[int(preds[0])]