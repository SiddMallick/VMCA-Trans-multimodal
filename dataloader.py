from torch.utils.data import Dataset, DataLoader
from PIL import Image

class SentimentDataset(Dataset):

  def __init__(self, text, image, label, transform = None):
    self.text = text
    self.image = image
    self.label = label
    self.transform = transform

  def __len__(self):
    return len(self.text)

  def __getitem__(self, item):
    text = self.text[item]
    try:
      image_path = "/home/jayanta_rs/Project/jayanta/all_project_code/Multimodal_sentiment_analysis/Images/"+str(self.image[item])
      image = Image.open(image_path).convert('RGB')
    except:
      return None, None, None

    if self.transform:
      image = self.transform(image)

    label = self.label[item]

    return text, image, label

def create_data_loader(df, batch_size, num_workers, transform = None):
  ds = SentimentDataset(
    text = df.Text.to_numpy(),
    image = df.IMAGES_ID.to_numpy(),
    label = df.Derogatory.to_numpy(),
    transform = transform
  )

  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=num_workers
  )
