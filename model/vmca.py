import torch
import torch.nn as nn

class ValueMixedCrossAttention(nn.Module):
  def __init__(self, input_size):
    super(ValueMixedCrossAttention, self).__init__()
    self.conv_k = nn.Conv1d(input_size*2, input_size, kernel_size = 1, stride = 1, padding = 0)
    self.conv_q = nn.Conv1d(input_size*2, input_size, kernel_size = 1, stride = 1, padding = 0)
    #self.conv_q = nn.Conv1d(input_size*2, input_size, kernel_size = 1, stride = 1, padding = 0)
    self.conv_final = nn.Conv1d(input_size*2, input_size, kernel_size = 1, stride = 1, padding = 0)

  def forward(self, text, image):
    k = torch.cat((image, text), dim = 1)
    q = torch.cat((image, text), dim = 1)

    k = self.conv_k(k)
    q = self.conv_q(q)
    k = torch.reshape(k, (k.size()[0], 197, 768))
    q = torch.reshape(q, (q.size()[0], 197, 768))
    attn_scores = torch.matmul(q, k.transpose(-2, -1))
    attn_weights = torch.softmax(attn_scores, dim=-1)

    vt= torch.matmul(attn_weights.transpose(-2, -1), torch.reshape(text, (text.size()[0], 197, 768)))
    vi = torch.matmul(attn_weights.transpose(-2,-1), torch.reshape(image, (image.size()[0], 197, 768)))


    return vi+vt