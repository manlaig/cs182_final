from collections import Counter
import torch as th
import word_embed as embed
from lstm import LSTM
from parameters import *

device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(device)

model = LSTM()
model.load_state_dict(th.load(root_folder + "model16.pt"))
model.to(device)
model.eval()

review = "always come here"

tensor = th.FloatTensor(embed.embed_review(review)).unsqueeze(0)
tensor.to(device)

output = model(tensor)
prediction = th.argmax(output)
print(int(prediction) / 2, "stars")