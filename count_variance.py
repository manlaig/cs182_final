import json

data = []
with open('dataset/test_data.jsonl', 'r') as file:
    data = [json.loads(jline) for jline in file.read().splitlines()]

labels = {}
for elem in data:
    stars = elem["stars"]

    if stars in labels:
        labels[stars] += 1
    else:
        labels[stars] = 1

# training dataset = {1.0: 91365, 2.0: 25058, 3.0: 24085, 4.0: 50803, 5.0: 182196}
# validation dataset = {1.0: 25006, 2.0: 7305, 3.0: 7158, 4.0: 15016, 5.0: 52231}
# test dataset = {1.0: 13507, 2.0: 3495, 3.0: 3020, 4.0: 6603, 5.0: 26733}
print(labels)