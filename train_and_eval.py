import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import pandas as pd
from sklearn.model_selection import train_test_split

from .dataloader import create_data_loader
from model.model_builder import model_build_multimodal
from .utils import train_fn, check_accuracy

# Define hyperparameters
RANDOM_SEED = 42
learning_rate = 5e-5
num_epochs = 20
test_split = 0.2
val_split = 0.5 # How much percent of validation data will be the test data
num_workers = 4
batch_size = 16
num_classes = 2  # Number of sentiment classes (e.g., negative, neutral, positive)
# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Read dataset
df = pd.read_csv('/home/jayanta_rs/Project/jayanta/all_project_code/Multimodal_sentiment_analysis/all_train.csv')

# Split the data
df_train, df_test = train_test_split(df, test_size = test_split, random_state=RANDOM_SEED)
df_val, df_test = train_test_split(df_test, test_size=val_split, random_state=RANDOM_SEED)

# Define data transforms
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_dataloader = create_data_loader(df_train, batch_size = batch_size, transform = data_transforms)
val_dataloader = create_data_loader(df_val, batch_size = batch_size, transform = data_transforms)

model = model_build_multimodal(2,6).to(device=device) #Best case result for binary classification

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(),lr=learning_rate)
#optimizer.zero_grad()
scaler = torch.cuda.amp.GradScaler()


def trainer():
    for epoch in range(num_epochs):
        # print("Epoch:",epoch)
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 10)
        train_loss = train_fn(train_dataloader, model, optimizer, loss_fn, scaler, device=device)
        #print(f"Train Loss: {train_loss}")
        check_accuracy(val_dataloader, model, train_loss, loss_fn, device =device)


def test():
    model.eval()
    cuda0 = torch.device('cuda:0')
    correct = 0
    total = 0
    preds=torch.empty(0, device=cuda0)
    target=torch.empty(0,device=cuda0)
    label=torch.empty(0,device=cuda0)

    with torch.no_grad():
        for i, data in enumerate(val_dataloader, 0):
            text, image, label = data
            label=label.to(cuda0)
            outputs = model(text, image).cuda()
            _, predicted = torch.max(outputs.data, 1)
            print(predicted)
            preds=torch.cat((preds,predicted))
            target=torch.cat((target,label)).to(cuda0)
            total += label.size(0)
            correct += (predicted == label.cuda()).sum().item()

    print('Accuracy of the network on the validation set: %d %%' % (
        100 * correct / total))
    

if __name__=="__main__":
    trainer()
    test()