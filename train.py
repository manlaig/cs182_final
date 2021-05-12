import json
from collections import Counter
import torch as th
from torch import nn
import torch.optim as optim
import word_embed as embed
from lstm import LSTM
from parameters import *

device = th.device("cuda" if th.cuda.is_available() else "cpu")

def train(model_name_to_save, trained_model_name = None):
    model = LSTM()
    if trained_model_name != None:
      model.load_state_dict(th.load(root_folder + trained_model_name))
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(device)

    optimizer = optim.Adam(model.parameters(), lr = lr)

    data = []
    with open(root_folder + 'dataset/training_data.jsonl', 'r') as file:
        data = [json.loads(jline) for jline in file.read().splitlines()]

    for epoch in range(epochs):
        count = 0
        batch_data = []
        batch_labels = []

        for elem in data:
            batch_data += [th.cuda.FloatTensor(embed.embed_review(elem["text"]))]
            batch_labels += [int(float(elem["stars"]) * 2)]
            count += 1

            if count % batch_size == 0:
                # forward and backward passes
                padded = nn.utils.rnn.pad_sequence(batch_data, batch_first=True)
                padded.to(device)
                targets = th.cuda.LongTensor(batch_labels)

                model.zero_grad()
                output = model(padded).squeeze()
                loss = loss_fn(output, targets)
                loss.backward()
                optimizer.step()

                """
                correct = 0
                for i in range(len(output)):
                  predicted = getStar(output[i])
                  actual = batch_labels[i]
                  print(len(predicted))

                  if isCorrect(predicted, actual):
                    correct += 1
                """

                batch_data = []
                batch_labels = []
                
                print("Epoch: " + str(epoch) + "\tBatch: " + str(count // batch_size) + "\tLoss: " + str(loss.item()))
        
        th.save(model.state_dict(), root_folder + model_name_to_save)