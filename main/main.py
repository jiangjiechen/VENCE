'''
1. sample edit token position
2. sample an edit action(insert,delete,replace)
3. Sample a corrected token/entity to replace the masked token/entity
'''
from energy_model import language_modeling_score, bert_ppl
from t5.main import generate, entity_pro_generate
from verfication.main import token_logit, only_pro
from ner.ner import entity_extract
from transformers import BertTokenizer, AutoModelForMaskedLM, AutoTokenizer
import numpy as np
import random
import distance
import torch
import argparse
import transformers
import json
import jsonlines
import os
import sys
import math

sys.path.append('..')


actions_3 = ['insert', 'delete', 'replace']
actions_2 = ['entity', 'token']

parser = argparse.ArgumentParser()
parser.add_argument('-iter_num', type=int, default=15,
                    help='Number of Iterations')
parser.add_argument('-model_version', type=str,
                    default='bert-base-uncased',
                    help='The base model used for the energy model')
parser.add_argument('-tokenizer_version', type=str,
                    default='bert-base-uncased',
                    help='The base tokenizer used for the energy model')
parser.add_argument('-es_lm', type=float,
                    default=0.08,
                    help='Weighting of the language model in the energy model')
parser.add_argument('-es_ver', type=float,
                    default=100,
                    help='The weight of the verifaction model in the energy model')
parser.add_argument('-es_dis', type=float,
                    default=8,
                    help='Weighting of distance in the energy model')
parser.add_argument('-early_stop', type=bool,
                    default=True,
                    help='early_stop')
parser.add_argument('-eval', type=bool,
                    default=False,
                    help='Evaluation Mode')

args = parser.parse_args()

# load model and tokenizer
model = AutoModelForMaskedLM.from_pretrained(args.model_version)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_version)

CLS = "[CLS]"
SEP = "[SEP]"
MASK = "[MASK]"
mask_id = tokenizer.convert_tokens_to_ids([MASK])[0]
sep_id = tokenizer.convert_tokens_to_ids([SEP])[0]
cls_id = tokenizer.convert_tokens_to_ids([CLS])[0]


def del_dot(batch):
    if batch[-1] == '.':
        batch = batch[:-1]
    return batch.strip()


def add_dot(batch):
    if batch[-1] != '.':
        batch = batch + '.'
    return batch.strip()


# tokenizer
def tokenize_batch(batch):
    return [tokenizer.convert_tokens_to_ids(sent) for sent in batch]


# init sentence
def init_batch(seed_text):
    """ Get initial sentence by padding seed_text with either masks or random words to max_len """
    batch = [CLS] + seed_text + [SEP]  # TODO
    return tokenize_batch(batch)


# sample action (insert,delete,replace)
def sample_edit_action(ch=2):
    if ch == 2:
        return actions_2[random.randint(0, 1)]
    else:
        pro = [0.1, 0.1, 0.8]
        return actions_3[normal_sample(pro)]


# sample from a list
def normal_sample(prob_list, value=False):
    temp = random.uniform(0, sum(prob_list))
    for i in range(len(prob_list)):
        if temp < sum(prob_list[:i + 1]):
            if value == True:
                return i, prob_list[i]
            else:
                return i


# merge token
def merge_token(token_list, merge_list):
    t_token_list = token_list[:]
    merge_list = sorted(merge_list, key=lambda x: x[0], reverse=True)
    for n in merge_list:
        if len(n) == 1:
            continue
        else:
            t_token_list[n[0]] = ''.join([t_token_list[i] for i in n])
            for i in range(1, len(n)):
                t_token_list.pop(n[0] + 1)
    return t_token_list


def remove_space_before_punctuation(s):
    punctuations = ['.', ',', ';', ':', '?', '!', '(', ')', '[', ']', '{', '}', '<', '>', '/', '\\', '|', '-', '+', '*',
                    '=', '&', '^', '%', '$', '#', '@', '`', '~', '_']
    result = ''
    for i in range(len(s)):
        if s[i] in punctuations:
            if i > 0 and s[i - 1] == ' ':
                result = result[:-1]
            result += s[i]
            if i < len(s) - 1 and s[i + 1] == ' ':
                i += 1
        else:
            result += s[i]
    return result


def add_space_before_punctuation(s):
    punctuations = ['.', ',', ';', ':', '?', '!', '(', ')', '[', ']', '{', '}', '<', '>', '/', '\\', '|', '-', '+', '*',
                    '=', '&', '^', '%', '$', '#', '@', '`', '~', '_']
    result = ''
    for i in range(len(s)):
        if s[i] in punctuations:
            if i > 0 and s[i - 1] != ' ':
                result += ' '
            result += s[i]
            if i < len(s) - 1 and s[i + 1] != ' ':
                result += ' '
        else:
            result += s[i]
    return result


def remove_sublists(a, b):
    result = []
    for lst in a:
        if not all(item in b for item in lst):
            result.append(lst)
    return result


# accept rate
def accept_rate(energy_score_f, energy_score_b, g_forward, g_backward):
    pro = (g_backward * math.exp(-energy_score_b)) / \
        (g_forward * math.exp(-energy_score_f))
    final_rate = min(1, pro)
    return final_rate


def main(data, args):
    last_correct_claim = []
    for i, all_data in enumerate(data):
        raw_claim = del_dot(all_data['original_claim'])
        evidence = all_data['evidence']
        iter_claim = [''] * args.iter_num
        iter_claim[0] = add_space_before_punctuation(raw_claim)
        claim_ner_result = entity_extract([raw_claim])
        evidence_ner_result = entity_extract([evidence])
        all_entity = list(set(
            [en[0] for en in claim_ner_result[0]] + [en[0] for en in evidence_ner_result[0]]))
        # print(all_entity)
        for j in range(args.iter_num - 1):
            # if new entity
            is_new_entity = False
            # 2. sample position
            claim_ner_result = entity_extract([iter_claim[j]])
            tokenizer_list_raw, final_logit, final_prob, spilt_token = token_logit(iter_claim[j], evidence, list(
                set([en[0] for en in claim_ner_result[0]])))  # entailment,neutral,contradiction
            forward_pro = 0
            backward_pro = 0
            # 3. sample action
            act_3 = sample_edit_action(3)
            if len(claim_ner_result[0]) == 0:
                act_2 = 'token'
            elif claim_ner_result[0][0][0] == raw_claim:
                act_2 = 'entity'
                while act_3 == actions_3[1]:
                    act_3 = sample_edit_action(3)
            else:
                act_2 = sample_edit_action(2)
            if args.eval == False:
                print(act_2, act_3)
            # action_2 : entity or token
            if act_2 == actions_2[1]:  # token
                token_prob = final_prob['token']
                token_sample_id, pos_pro = normal_sample(
                    [tt['prob'] for tt in token_prob], value=True)
                token_sample_pos = 0
                token_sample_token = ''
                for n in token_prob:
                    if n['id'] == token_sample_id:
                        token_sample_pos = n['pos']
                        token_sample_token = n['token']
                        break
                if act_3 == actions_3[2]:  # replace
                    if type(token_sample_pos) == list:
                        token_len = len(token_sample_pos)
                        for _, ww in enumerate(spilt_token):
                            if set(ww).issubset(set(token_sample_pos)):
                                spilt_token.pop(_)
                        for _, k in enumerate(spilt_token):
                            if k[0] > token_sample_pos[-1]:
                                spilt_token[_] = [
                                    k[iii] - token_len + 1 for iii in range(len(k))]
                    if type(token_sample_pos) == int:
                        tokenizer_list_raw[token_sample_pos] = MASK
                    else:
                        for nn in range(len(token_sample_pos)):
                            tokenizer_list_raw.pop(token_sample_pos[0])
                        tokenizer_list_raw.insert(token_sample_pos[0], MASK)
                    mask_token_claim = ' '.join(
                        merge_token(tokenizer_list_raw, spilt_token))
                    gen_claim, token_pro, backward_pro = generate(
                        'substituted one token : ' +
                        '[evidence] :' + evidence +
                        ' [claim] : ' + mask_token_claim,
                        candi_token=token_sample_token)
                    if args.eval == False:
                        print(gen_claim)
                    if type(token_sample_pos) == int:
                        tokenizer_list_raw[token_sample_pos] = gen_claim
                    else:
                        tokenizer_list_raw.pop(token_sample_pos[0])
                        tokenizer_list_raw.insert(
                            token_sample_pos[0], gen_claim)
                    tokenizer_list_raw = merge_token(
                        tokenizer_list_raw, spilt_token)
                    correct_claim = ' '.join(tokenizer_list_raw)
                    # prob
                    forward_pro = token_pro
                elif act_3 == actions_3[0]:  # insert
                    if type(token_sample_pos) == list:
                        for _, k in enumerate(spilt_token):
                            if k[0] > token_sample_pos[-1]:
                                spilt_token[_] = [
                                    k[iii] + 1 for iii in range(len(k))]
                    else:
                        for _, k in enumerate(spilt_token):
                            if k[0] > token_sample_pos:
                                spilt_token[_] = [
                                    k[iii] + 1 for iii in range(len(k))]
                    if type(token_sample_pos) == int:
                        change_pos = token_sample_pos
                    else:
                        change_pos = token_sample_pos[0]
                    tokenizer_list_raw.insert(change_pos, MASK)
                    mask_token_claim = ' '.join(
                        merge_token(tokenizer_list_raw, spilt_token))
                    gen_claim, token_pro, _ = generate(
                        'substituted one token : ' + '[evidence] :' + evidence + ' [claim] : ' + mask_token_claim)
                    if args.eval == False:
                        print(gen_claim)
                    if type(token_sample_pos) == int:
                        tokenizer_list_raw[token_sample_pos] = gen_claim
                    else:
                        tokenizer_list_raw[token_sample_pos[-1]] = gen_claim
                    tokenizer_list_raw = merge_token(
                        tokenizer_list_raw, spilt_token)
                    correct_claim = ' '.join(tokenizer_list_raw)
                    # print(correct_claim)
                    # prob
                    forward_pro = token_pro * pos_pro
                    claim_ner_result_2 = entity_extract([correct_claim])
                    for en in claim_ner_result_2[0]:
                        if en[0] not in all_entity:
                            is_new_entity = True
                            break
                    if not is_new_entity:
                        _, _, final_prob_2, _ = token_logit(correct_claim, evidence,
                                                            list(set([en[0] for en in claim_ner_result_2[0]])))
                        token_prob_2 = final_prob_2['token']
                        for n in token_prob_2:
                            if n['token'] == token_sample_token:
                                backward_pro = n['prob']
                else:  # delete
                    back_tokenizer_list_raw = tokenizer_list_raw.copy()
                    if type(token_sample_pos) == list:
                        token_len = len(token_sample_pos)
                        for _, ww in enumerate(spilt_token):
                            if set(ww).issubset(set(token_sample_pos)):
                                spilt_token.pop(_)
                        for _, k in enumerate(spilt_token):
                            if k[0] > token_sample_pos[-1]:
                                spilt_token[_] = [
                                    k[iii] - token_len for iii in range(len(k))]
                    else:
                        for _, k in enumerate(spilt_token):
                            if k[0] > token_sample_pos:
                                spilt_token[_] = [
                                    k[iii] - 1 for iii in range(len(k))]
                    if type(token_sample_pos) == int:
                        tokenizer_list_raw.pop(token_sample_pos)
                        back_tokenizer_list_raw.pop(token_sample_pos)
                        back_tokenizer_list_raw.insert(token_sample_pos, MASK)
                    else:
                        for nn in range(len(token_sample_pos)):
                            tokenizer_list_raw.pop(token_sample_pos[0])
                            back_tokenizer_list_raw.pop(token_sample_pos[0])
                        back_tokenizer_list_raw.insert(
                            token_sample_pos[0], MASK)
                    tokenizer_list_raw = merge_token(
                        tokenizer_list_raw, spilt_token)
                    back_tokenizer_list_raw = merge_token(
                        back_tokenizer_list_raw, spilt_token)
                    correct_claim = ' '.join(tokenizer_list_raw)
                    back_claim = ' '.join(back_tokenizer_list_raw)
                    # prob
                    forward_pro = pos_pro
                    su = 'substituted one token : ' + \
                        '[evidence] :' + evidence + ' [claim] : ' + back_claim
                    backward_pro = pos_pro * \
                        generate(su, candi_token=token_sample_token)[-1]
            else:  # entity
                entity_prob = final_prob['entity']
                if len(entity_prob) != 0:
                    entity_sample_id, pos_pro = normal_sample(
                        [tt['prob'] for tt in entity_prob], value=True)
                else:
                    continue
                entity_sample_pos = 0
                entity_sample_entity = ''
                for n in entity_prob:
                    if n['id'] == entity_sample_id:
                        entity_sample_pos = n['pos']
                        entity_sample_entity = n['entity']
                        break
                if act_3 == actions_3[2]:  # replace
                    if type(entity_sample_pos) == list:
                        entity_len = len(entity_sample_pos)
                        spilt_token = remove_sublists(
                            spilt_token, entity_sample_pos)
                        for _, k in enumerate(spilt_token):
                            if k[0] > entity_sample_pos[-1]:
                                spilt_token[_] = [
                                    k[iii] - entity_len + 1 for iii in range(len(k))]
                    if type(entity_sample_pos) == int:
                        tokenizer_list_raw[entity_sample_pos] = MASK
                    else:
                        for nn in range(len(entity_sample_pos)):
                            tokenizer_list_raw.pop(entity_sample_pos[0])
                        tokenizer_list_raw.insert(entity_sample_pos[0], MASK)
                    mask_entity_claim = ' '.join(
                        merge_token(tokenizer_list_raw, spilt_token))
                    su = ' substituted entity : ' + \
                        '[evidence] : ' + evidence + \
                        ' [claim] : ' + mask_entity_claim
                    # print(su)
                    entity_p_sum = entity_pro_generate(su, all_entity)
                    sample_pos = entity_p_sum.index(max(entity_p_sum))
                    gen_claim = all_entity[sample_pos]
                    if args.eval == False:
                        print(gen_claim)
                    if type(entity_sample_pos) == int:
                        tokenizer_list_raw[entity_sample_pos] = gen_claim
                    else:
                        tokenizer_list_raw.pop(entity_sample_pos[0])
                        tokenizer_list_raw.insert(
                            entity_sample_pos[0], gen_claim)
                    tokenizer_list_raw = merge_token(
                        tokenizer_list_raw, spilt_token)
                    correct_claim = ' '.join(tokenizer_list_raw)
                    # prob
                    forward_pro = entity_p_sum[sample_pos]
                    backward_pro = entity_p_sum[all_entity.index(
                        entity_sample_entity)]
                elif act_3 == actions_3[0]:  # insert
                    if type(entity_sample_pos) == list:
                        for _, k in enumerate(spilt_token):
                            if k[0] > entity_sample_pos[-1]:
                                spilt_token[_] = [
                                    k[iii] + 1 for iii in range(len(k))]
                    else:
                        for _, k in enumerate(spilt_token):
                            if k[0] > entity_sample_pos:
                                spilt_token[_] = [
                                    k[iii] + 1 for iii in range(len(k))]
                    if type(entity_sample_pos) == int:
                        change_pos = entity_sample_pos
                    else:
                        change_pos = entity_sample_pos[0]
                    tokenizer_list_raw.insert(change_pos, MASK)
                    mask_entity_claim = ' '.join(
                        merge_token(tokenizer_list_raw, spilt_token))
                    su = 'substituted entity : ' + \
                        '[evidence] :' + evidence + \
                        ' [claim] : ' + mask_entity_claim
                    gen_claim = all_entity[
                        entity_pro_generate(su, all_entity).index(max(entity_pro_generate(su, all_entity)))]
                    if args.eval == False:
                        print(gen_claim)
                    tokenizer_list_raw.pop(change_pos)
                    tokenizer_list_raw.insert(change_pos, gen_claim)
                    tokenizer_list_raw = merge_token(
                        tokenizer_list_raw, spilt_token)
                    correct_claim = ' '.join(tokenizer_list_raw)
                    # prob
                    forward_pro = max(entity_pro_generate(
                        su, all_entity)) * pos_pro
                    claim_ner_result_2 = entity_extract([correct_claim])
                    for en in claim_ner_result_2[0]:
                        if en[0] not in all_entity:
                            is_new_entity = True
                            break
                    if not is_new_entity:
                        _, _, final_prob_2, _ = token_logit(correct_claim, evidence,
                                                            list(set([en[0] for en in claim_ner_result_2[0]])))
                        for ent in final_prob_2['entity']:
                            if ent['entity'] == gen_claim:
                                backward_pro = ent['prob']
                else:  # delete
                    back_tokenizer_list_raw = tokenizer_list_raw.copy()
                    if type(entity_sample_pos) == list:
                        entity_len = len(entity_sample_pos)
                        spilt_token = remove_sublists(
                            spilt_token, entity_sample_pos)
                        for _, k in enumerate(spilt_token):
                            if k[0] > entity_sample_pos[-1]:
                                spilt_token[_] = [
                                    k[iii] - entity_len for iii in range(len(k))]
                    else:
                        for _, k in enumerate(spilt_token):
                            if k[0] > entity_sample_pos:
                                spilt_token[_] = [
                                    k[iii] - 1 for iii in range(len(k))]
                    if type(entity_sample_pos) == int:
                        tokenizer_list_raw.pop(entity_sample_pos)
                        back_tokenizer_list_raw.pop(entity_sample_pos)
                        back_tokenizer_list_raw.insert(entity_sample_pos, MASK)
                    else:
                        for nn in range(len(entity_sample_pos)):
                            tokenizer_list_raw.pop(entity_sample_pos[0])
                            back_tokenizer_list_raw.pop(entity_sample_pos[0])
                        back_tokenizer_list_raw.insert(
                            entity_sample_pos[0], MASK)
                    tokenizer_list_raw = merge_token(
                        tokenizer_list_raw, spilt_token)
                    back_tokenizer_list_raw = merge_token(
                        back_tokenizer_list_raw, spilt_token)
                    correct_claim = ' '.join(tokenizer_list_raw)
                    back_claim = ' '.join(back_tokenizer_list_raw)
                    # prob
                    forward_pro = pos_pro
                    claim_ner_result_2 = entity_extract([correct_claim])
                    for en in claim_ner_result_2[0]:
                        if en[0] not in all_entity:
                            is_new_entity = True
                            break
                    if not is_new_entity:
                        su = ' substituted entity : ' + \
                            '[evidence] : ' + evidence + \
                            ' [claim] : ' + back_claim
                        entity_p_sum = entity_pro_generate(su, all_entity)
                        backward_pro = pos_pro * \
                            entity_p_sum[all_entity.index(
                                entity_sample_entity)]
            new_ner_entity = entity_extract([correct_claim])
            for en in new_ner_entity[0]:
                if en[0] not in all_entity:
                    is_new_entity = True
                    break
            if is_new_entity:
                iter_claim[j + 1] = iter_claim[j]
            else:
                # less is more
                def energy_sum(co_claim):
                    ver_score = args.es_ver * only_pro(co_claim, evidence)[2]
                    dis_score = args.es_dis * \
                        distance.levenshtein(
                            co_claim.split(), iter_claim[0].split())
                    lm_score = args.es_lm * bert_ppl(co_claim)
                    return ver_score, dis_score, lm_score, 0.1 * (ver_score + dis_score + lm_score)

                a_rate = accept_rate(energy_sum(iter_claim[j])[-1], energy_sum(correct_claim)[-1],
                                     g_forward=float(forward_pro), g_backward=float(backward_pro))
                # random 0-1
                random_num = random.random()
                if random_num < a_rate:
                    iter_claim[j + 1] = correct_claim
                    if args.eval == False:
                        print('accept')
                else:
                    iter_claim[j + 1] = iter_claim[j]
                    if args.eval == False:
                        print('reject')
            if args.eval == False:
                print(iter_claim[j + 1])
                print('-----------------')
        if args.eval == False:
            print('-----------------')  # end of iter
        last_claim = add_dot(remove_space_before_punctuation(iter_claim[-1]))
        last_correct_claim.append(last_claim)
    return last_correct_claim


if __name__ == '__main__':
    test_data = [{'original_claim': 'Trump won the 2022 election.',
                  'mutated_claim': 'Joe Biden won the 2022 election',
                  'evidence': 'Democrat Joe Biden beats Republican Trump in 2022 election'}]
    print(main(new_data, args))
