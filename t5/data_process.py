import jsonlines
import json
import random
import os
import sys
from tqdm import tqdm

sys.path.append('..')

from ner.ner import entity_extract
import utils


# A string is a sentence, enter the start and end positions of a word in the string and calculate how many words in the sentence the word is.
def get_word_index(text, word_start, word_end):
    # print(word_start,word_end)
    text_1 = text[:word_start]
    text_1 = text_1.split()
    text_2 = text[:word_end]
    text_2 = text_2.split()
    return [len(text_1), len(text_2)]


# Split a list into a list of n elements each
def split_list(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


# Replace entities or words in the text with a [MASK]
def mask_words(text, entity_index, entity_or_word='entity'):
    text = text.split()
    # entity mask
    if entity_or_word == 'entity':
        choice_entity_index=random.randint(0,len(entity_index)-1)
        entity_index = entity_index[choice_entity_index]
        text[entity_index[0]] = '[MASK]'
        for i in range(entity_index[0] + 1, entity_index[1]):
            text[i] = ''
        return ' '.join((' '.join(text)).split()), choice_entity_index
    # word mask
    else:
        temp=1
        continue_flag = True
        word_index = 0
        while(continue_flag):
            temp+=1
            if temp>100:
                break
            word_index = random.randint(0, len(text) - 1)
            continue_flag = False
            for index in entity_index:
                if word_index in range(index[0], index[1]):
                    continue_flag = True
                    break
        text[word_index] = '[MASK]'
        return ' '.join(text), word_index



if __name__ == '__main__':
    #Extract the location of the entity
    with jsonlines.open('../data/whitebox_ir_train_genre_50_2.jsonl') as reader, jsonlines.open('data/entity_loc.jsonl', mode='w') as writer:
        text_100 = split_list([[obj['sentence1'],obj['sentence2']] for obj in reader if obj['original']['veracity']=="SUPPORTS"] , 100)
        num = 0
        for text in text_100:
            text_text=[item[0] for item in text]
            text_evidence=[item[1] for item in text]
            for _, sentence in enumerate(entity_extract(text_text)):
                entity_loc = [get_word_index(text_text[_], sentence[index][1], sentence[index][2]) for index in
                              range(len(sentence))]
                writer.write([text_text[_],[item[0] for item in sentence], entity_loc,text_evidence[_]])
            num += 100
            print(num)


    #Write data
    with jsonlines.open('data/t5_data.jsonl',mode='w') as writer, jsonlines.open('data/entity_loc.jsonl') as reader:
        for obj in tqdm(reader):
            try:
                masked_entity=mask_words(obj[0], obj[2],entity_or_word='entity')
                masked_word=mask_words(obj[0], obj[2],entity_or_word='word')
                writer.write({'input':'substituted entity : ' + '[evidence] : '+obj[-1]+' [claim] : '+masked_entity[0],'output':obj[1][masked_entity[1]]})
                writer.write({'input':'substituted one word : ' + '[evidence] : ' + obj[-1] + ' [claim] : ' + masked_word[0], 'output': obj[0].split()[masked_word[1]]})
            except:
                pass

    #Divide the jsonl data into training set, validation set and test set with a ratio of 8:1:1
    with open('data/t5_data.jsonl') as reader:
        data = [item for item in reader]
        random.shuffle(data)
        train_data = data[:int(len(data) * 0.8)]
        val_data = data[int(len(data) * 0.8):int(len(data) * 0.9)]
        test_data = data[int(len(data) * 0.9):]
        with open('data/t5_train.jsonl', 'w') as writer:
            writer.writelines(train_data)
        with open('data/t5_val.jsonl', 'w') as writer:
            writer.writelines(val_data)
        with open('data/t5_test.jsonl', 'w') as writer:
            writer.writelines(test_data)


    # Convert jsonl file to json file
    with jsonlines.open('data/t5_train.jsonl') as reader, open('data/t5_train.json', 'w') as writer:
        json.dump([obj for obj in reader], writer, indent=4)
    with jsonlines.open('data/t5_val.jsonl') as reader, open('data/t5_val.json', 'w') as writer:
        json.dump([obj for obj in reader], writer, indent=4)
    with jsonlines.open('data/t5_test.jsonl') as reader, open('data/t5_test.json', 'w') as writer:
        json.dump([obj for obj in reader], writer, indent=4)

    with jsonlines.open('data/t5_data.jsonl') as reader, open('data/t5_data.json', 'w') as writer:
        json.dump([obj for obj in reader], writer, indent=4)


    #Extract the first 500 data from the json file and write them into a new json file
    with open('data/t5_train.json') as reader, open('data/t5_train_500.json', 'w') as writer:
        data = json.load(reader)
        json.dump(data[:500], writer, indent=4)
    with open('data/t5_val.json') as reader, open('data/t5_val_100.json', 'w') as writer:
        data = json.load(reader)
        json.dump(data[:100], writer, indent=4)
    with open('data/t5_test.json') as reader, open('data/t5_test_100.json', 'w') as writer:
        data = json.load(reader)
        json.dump(data[:100], writer, indent=4)