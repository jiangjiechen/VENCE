import jsonlines
import random
import json

if __name__ == '__main__':
    with jsonlines.open('../data/whitebox_ir_train_genre_50_2.jsonl') as reader, jsonlines.open('data/verfication_data.jsonl', mode='w') as writer:
        support_num = 0
        refute_num = 0
        for obj in reader:
            if obj['original']['veracity'] == "SUPPORTS":
                writer.write(
                    {'sentence1': obj['sentence1'], 'sentence2': obj['sentence2'], 'label': 'SUPPORTS'})
                support_num += 1
            else:
                writer.write(
                    {'sentence1': obj['sentence1'], 'sentence2': obj['sentence2'], 'label': 'REFUTES'})
                refute_num += 1
        print(support_num, refute_num)

    # Convert jsonl file to json file
    with jsonlines.open('data/verfication_data.jsonl') as reader:
        data = [obj for obj in reader]
        with open('data/verfication_data.json', 'w') as f:
            json.dump(data, f)

    # Divide the json file into training set, validation set and test set
    with open('data/verfication_data.json', 'r') as f:
        data = json.load(f)
        random.shuffle(data)
        train_data = data[:int(len(data)*0.8)]
        valid_data = data[int(len(data)*0.8):int(len(data)*0.9)]
        test_data = data[int(len(data)*0.9):]
        with open('data/train.json', 'w') as f:
            json.dump(train_data, f)
        with open('data/valid.json', 'w') as f:
            json.dump(valid_data, f)
        with open('data/test.json', 'w') as f:
            json.dump(test_data, f)
