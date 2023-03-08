from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import torch.nn as nn
from transformers import pipeline
from functools import reduce
import numpy as np
import os

tokenizer = AutoTokenizer.from_pretrained(
    f"{os.environ['PJ_HOME']}/t5/model/checkpoint-15000")
model = AutoModelForSeq2SeqLM.from_pretrained(
    f"{os.environ['PJ_HOME']}/t5/model/checkpoint-15000")
model.eval()


def generate(text, candi_token=''):
    inputs = tokenizer(text, return_tensors="pt")
    model.config.output_scores = True
    outputs = model.generate(inputs['input_ids'], max_length=100, num_beams=5,
                             early_stopping=True, output_scores=True, return_dict_in_generate=True)
    candi_token_id = tokenizer.convert_tokens_to_ids(
        tokenizer.tokenize(candi_token))
    output_seq = torch.Tensor.tolist(outputs.sequences[0])[1:]
    token_pro = reduce(lambda x, y: x * y,
                       [nn.Softmax(dim=1)(torch.unsqueeze(outputs.scores[i][0], 0))[0][output_seq[i]].float() for i in
                        range(len(output_seq))])
    if candi_token != '':
        candi_token_id = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(candi_token))
        token_pro_can = reduce(lambda x, y: x * y,
                               [nn.Softmax(dim=1)(torch.unsqueeze(outputs.scores[i][0], 0))[0][candi_token_id[i]].float() for i in
                                range(len(candi_token_id))])
    else:
        token_pro_can = 0
    return tokenizer.decode(outputs.sequences[0], skip_special_tokens=True), token_pro, token_pro_can


def entity_pro_generate(text, entity_list):
    model.config.output_scores = True
    score = []
    for entity in entity_list:
        text_tensor = torch.tensor(
            [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))])
        entity_tensor = torch.tensor(
            [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(entity))])
        loss = model(text_tensor, labels=entity_tensor)[0]
        ppl = np.exp(-loss.data.item())
        score.append(ppl)
    return [s/sum(score) for s in score]


if __name__ == '__main__':
    test_text = "substituted token : [evidence] : In the 2020 US election, Democrat Joe Biden defeated Republican Donald Trump and was successfully elected as the 46th President of the United States. [claim] : Joe Biden [MASK] the 2020 US election."
    print(generate(test_text, 'lost'))
