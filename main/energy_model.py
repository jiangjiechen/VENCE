'''
energy
1. Language Modeling Score
2. Truthfulness Score
3. Hamming Distance Score
'''

import warnings
import distance
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"


warnings.filterwarnings('ignore')

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

CLS = "[CLS]"
SEP = "[SEP]"
MASK = "[MASK]"
mask_id = tokenizer.convert_tokens_to_ids([MASK])[0]
sep_id = tokenizer.convert_tokens_to_ids([SEP])[0]
cls_id = tokenizer.convert_tokens_to_ids([CLS])[0]

cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'

lm_score_model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
lm_score_model.eval()


if cuda:
    lm_score_model = lm_score_model.cuda()


def tokenize_batch(batch):
    return [tokenizer.convert_tokens_to_ids(sent) for sent in batch]


def get_init_text(seed_text, max_len, batch_size=1, rand_init=False):
    """ Get initial sentence by padding seed_text with either masks or random words to max_len """
    batch = [[CLS] + seed_text + [SEP] for _ in range(batch_size)]  # TODO
    return tokenize_batch(batch)


def language_modeling_score(text):
    text = tokenizer.tokenize(text)
    batch = torch.tensor(get_init_text(
        text, max_len=15, batch_size=1)).to(device)
    seq_len = len(batch[0]) - 2
    posns = [i + 1 for i in range(seq_len)]
    norm_score = [0.0] * batch.shape[0]
    raw_score = [0.0] * batch.shape[0]
    for posn in posns:
        old_wrd = batch[:, posn].clone()
        batch[:, posn] = mask_id
        output = lm_score_model(batch)['logits'][:, posn, :]
        norm_output = output.log_softmax(dim=-1)
        for i in range(batch.shape[0]):
            raw_score[i] += output[i, old_wrd[i]].item()
            norm_score[i] += norm_output[i, old_wrd[i]].item()
        batch[:, posn] = old_wrd
    return [-1.0 * raw_s for raw_s in raw_score], [-1.0 * norm_s for norm_s in norm_score]


def bert_ppl(sent):
    tokenize_input = tokenizer.tokenize(sent)
    tensor_input = torch.tensor(
        [tokenizer.convert_tokens_to_ids(tokenize_input)])
    sen_len = len(tokenize_input)
    sent_loss = 0.
    for i, word in enumerate(tokenize_input):
        tokenize_input[i] = tokenizer.mask_token
        mask_input = torch.tensor(
            [tokenizer.convert_tokens_to_ids(tokenize_input)]).to(device)
        output = lm_score_model(mask_input)
        pred_scores = output[0]
        ps = torch.log_softmax(pred_scores[0, i], dim=0)
        word_loss = ps[tensor_input[0, i]]
        sent_loss += word_loss.item()
        tokenize_input[i] = word
    ppl = np.exp(-sent_loss/sen_len)
    return ppl


def distance_score(text, raw_text):
    text = tokenizer.tokenize(text)
    raw_text = tokenizer.tokenize(raw_text)
    return distance.levenshtein(text, raw_text)
