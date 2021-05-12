import json
import preprocess_text as preprocess

out = open("test2_formatted.jsonl", "w")
with open('dataset/test_data3.jsonl', 'r') as file:
    data = file.read()
    j = [json.loads(jline) for jline in data.splitlines()]
    count = 0
    for elem in j:
        elem["text"] = preprocess.process(elem["text"])
        out.write(json.dumps(elem) + '\n')
        if count % 50 == 0:
            print(count)
        count += 1
out.close()