from lstm import LSTM
import torch as th
from torch import nn
import json
import word_embed as embed
from parameters import *

device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(device)

model = LSTM()
model.load_state_dict(th.load(root_folder + "model64.pt", map_location="cpu"))
model.to(device)
model.eval()

loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

data = []
with open(root_folder + 'dataset/validation_data.jsonl', 'r') as file:
    data = [json.loads(jline) for jline in file.read().splitlines()]

total_diff = 0
correct = 0
count = 0
for elem in data:
    tensor = th.FloatTensor(embed.embed_review(elem["text"])).unsqueeze(0)
    tensor.to(device)

    output = model(tensor)
    loss = loss_fn(output, th.LongTensor([int(float(elem["stars"]) * 2)], device=device))
    prediction = th.argmax(output)
    
    correct += 1 if int(prediction) / 2 == float(elem["stars"]) else 0
    count += 1
    print("Predicted:", int(prediction) / 2,
        "\tActual:", elem["stars"],
        "\tAccuracy:", round(correct / count, 3),
        "\tVal Loss:", loss.item())
