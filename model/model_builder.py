from transformers import BertModel, BertTokenizer
import timm
from .multimodal_classifier import MultimodalClassifierWithLSTM

def model_build_multimodal(num_classes, trans_num_layers):
    text_model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', force_download=True)
    image_model = timm.create_model('vit_base_patch16_224', pretrained=True)

    return MultimodalClassifierWithLSTM(num_classes = num_classes,
                             text_model = text_model,
                             tokenizer = tokenizer,
                             image_model = image_model,
                             trans_num_layers = trans_num_layers)