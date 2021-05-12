import json, sys
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
import preprocess_text as preprocess
from transformers import *
import numpy as np

from datetime import datetime

MAX_LEN = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

print("Loading tokenizer")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False, remove_space=False)
print("Tokenizer loaded")
print("Loading model")
model = BertForSequenceClassification.from_pretrained("./BERT_lr_26", num_labels=5)
model.to(device)
model.eval()
print("Model loaded")

def eval(text):
	text = preprocess.process(text)
	data = tokenizer.tokenize("[CLS] " + text + " [SEP]")

	data = tokenizer.convert_tokens_to_ids(data)
	data = pad_sequences([data], maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

	attention_masks = [[float(i>0) for i in data[0]]]

	data = torch.tensor(data).to(device)
	attention_masks = torch.tensor(attention_masks).to(device)

	with torch.no_grad():
		outputs = model(data, token_type_ids=None, attention_mask=attention_masks)
		logits = outputs[0]

	logits = logits.detach().cpu().numpy()
	pred = np.argmax(logits[0])
	return str(pred + 1)

if len(sys.argv) > 1:
	validation_file = sys.argv[1]
	with open("output.jsonl", "w") as fw:
		with open(validation_file, "r") as fr:
			for line in fr:
				review = json.loads(line)
				fw.write(json.dumps({"review_id": review['review_id'], "predicted_stars": eval(review['text'])})+"\n")
	print("Output prediction file written")
else:
	print("No validation file given")