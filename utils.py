import torch
from torchmetrics.classification import MulticlassPrecision,MulticlassRecall



def check_accuracy(loader, model, train_loss, loss_fn, device = "cuda"):
  correct = 0
  total = 0
  recall = 0
  precision = 0

  cuda0 = torch.device('cuda:0')
  model.eval()

  with torch.no_grad():
    # for i, data in enumerate(val_dataloader, 0):
    for data in loader:
      text, image, label = data
      label=label.to(cuda0)
      outputs = model(text, image).cuda()
      _, predictions = torch.max(outputs.data, 1)

      preds = predictions.type(torch.cuda.FloatTensor)
      labs = label.type(torch.cuda.FloatTensor)
      #val_loss += loss_fn(preds, labs)

      #Calculate accuracy
      total += label.size(0)
      correct += (predictions == label.cuda()).sum().item()

      prec = MulticlassPrecision(num_classes = 2).to(cuda0)
      precision += prec(predictions, label)

      rec = MulticlassRecall(num_classes = 2).to(cuda0)
      recall += rec(predictions, label)


  print('Accuracy of the network on the validation set: %d %%' % (
    100 * correct / total))

  print(f"Train Loss: {train_loss}")
  #print(f"Val Loss: {val_loss/len(loader)}")
  print(f"Precision: {precision/len(loader)}")
  print(f"Recall: {recall/len(loader)}")



  model.train()

def train_fn(loader, model, optimizer, loss_fn, scaler,device):
  train_loss = []
  # for batch_idx, data in enumerate(loop):
  for data in loader:
    text, image, label = data
    if text == None:
          continue

    label = label.type(torch.LongTensor)
        # print(outputs)
    with torch.cuda.amp.autocast():
      outputs = model(text, image)
      outputs, label = outputs.to(device), label.to(device)
      loss = loss_fn(outputs, label)

    #backward
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    train_loss.append(loss.item())
    # loop.set_postfix(loss = loss.item())

  return sum(train_loss)/len(train_loss)