# from main import main
import json
import time

from evaluate import load
from main import main
from tqdm import tqdm
import argparse

sari = load("sari")

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
# args.eval=True

with open('../data/raw_test.jsonl') as reader:
    data = [item for item in reader]
    for d in tqdm(data[0:50]):
        claim, evidence, mut_claim = eval(d)["metadata"]['source'], eval(d)["metadata"]['evidence'], \
                                     eval(d)["metadata"]['target']
        batch = [{'original_claim': claim, 'mutated_claim': mut_claim, 'evidence': evidence}]
        try:
            cor_batch = main(batch, args)
            print(cor_batch)
            print(sari.compute(sources=[claim], predictions=cor_batch, references=[[mut_claim]]))
        except:
            pass
