from .vmca import ValueMixedCrossAttention
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# BiLSTM Layer
class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        # BiLSTM output shape will be: (batch_size, seq_len, hidden_dim * 2)
        # For hidden_dim=256, output shape is (batch_size, seq_len, 512)
        return out

class MultimodalClassifierWithLSTM(nn.Module):
    def __init__(self, num_classes, text_model, tokenizer, image_model, trans_num_layers):
        super(MultimodalClassifierWithLSTM, self).__init__()

        # Text encoder using BERT
        self.bert_model = text_model
        self.text_tokenizer = tokenizer
        self.reducer = nn.Conv1d(256,197,kernel_size = 1, padding = 0, stride =1)
        # Image encoder using Vision Transformer
        self.image_encoder = image_model

        self.cross_attention = ValueMixedCrossAttention(768)
        self.bilstmvit = BiLSTM(input_dim=768, hidden_dim=384, num_layers=1)
        self.bilstmbert = BiLSTM(input_dim=768, hidden_dim=384, num_layers=1)
        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=768, nhead=12), num_layers=trans_num_layers)

        self.bilstmfinal = BiLSTM(input_dim=768, hidden_dim=384, num_layers=1)
        # Fully connected layer for classification
        self.fc = nn.Linear(768, num_classes)

        # Define linear layers for computing the attention scores
        input_size = 768
        self.attention_query = nn.Linear(input_size, input_size)
        self.attention_key = nn.Linear(input_size, input_size)

        # Define a linear layer for the output
        self.out_layer = nn.Linear(input_size, input_size)

        self.dropout = nn.Dropout(0.2)



    def forward(self, text, image):
        # Text encoding using BERT
        input_ids = self.text_tokenizer(text, padding='max_length',truncation=True, return_tensors='pt', max_length=256)['input_ids'].to(device)
        attention_mask = self.text_tokenizer(text, padding='max_length',truncation=True, return_tensors='pt', max_length=256)['attention_mask'].to(device)
        bert_output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:,:,:]

        # Image encoding using Vision Transformer
        image_encoding = self.image_encoder.forward_features(image.to(device))


        bert_output = self.reducer(bert_output)
     
        bert_output = self.bilstmvit(bert_output)
        image_encoding = self.bilstmbert(image_encoding)
        encoding = self.cross_attention(torch.reshape(bert_output, (bert_output.size()[0], 768, 197)), torch.reshape(image_encoding, (bert_output.size()[0],768,197)))

        encoding = self.transformer_encoder(encoding)

        encoding = self.bilstmfinal(encoding)
        # Classification using fully connected layer
        x = self.fc(encoding[:,0,:])
        return x