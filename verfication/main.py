from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import torch
import torch.nn as nn

tokenizer = AutoTokenizer.from_pretrained("ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli")
model = AutoModelForSequenceClassification.from_pretrained("ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli")
model.eval()

# ['aaa','b','aaa'] ['aaa b'] [] [[0,1]
# ['aa','a','b','aa','a'] ['aaa'] [[0,1],[3,4]] [[0,1],[3,4]]
def entity_position_com(claim, entity_position,sp_token):
    temp_entity_position=entity_position[:]
    return_position = []
    tokenizer_list = [t[1:] if t[0] == 'Ġ' else t for t in tokenizer.tokenize(claim)]
    entity_dict={}
    for entity in temp_entity_position:
        entity_dict[entity]=0
    for p in sp_token:
        for ent in temp_entity_position:
            if ''.join([tokenizer_list[cc] for cc in p])==ent:
                return_position.append(p)
                temp_entity_position.remove(''.join([tokenizer_list[cc] for cc in p]))
            elif ''.join([tokenizer_list[cc] for cc in p]) in ent:
                duo_entity=[len(p)-1]
                for _,ii in enumerate(ent.split()):
                    if ii==''.join([tokenizer_list[cc] for cc in p]):
                        duo_entity.append(_)
                entity_dict[ent]=duo_entity
    for entity in temp_entity_position:
        if ' ' not in entity:
            return_position.append(tokenizer_list.index(entity))
        else:
            temp_en = entity.split(' ')
            temp_pos = []
            for k in range(len(tokenizer_list)):
                if tokenizer_list[k] == temp_en[0]:
                    for kk in range(len(temp_en)):
                        if tokenizer_list[k + kk] == temp_en[kk]:
                            temp_pos.append(k + kk)
                        else:
                            temp_pos=[]
                            break
                    return_position.append(temp_pos)
                    temp_pos=[]
    return return_position

def entity_position_com2(claim, entity_position):
    print('claim:',claim)
    print('entity_position:',entity_position)
    return_position = []
    tokenizer_list = [t[1:] if t[0] == 'Ġ' else t for t in tokenizer.tokenize(claim)]
    for entity in entity_position:
        entity_list_raw = [t[1:] if t[0] == 'Ġ' else t for t in tokenizer.tokenize(entity)]
        if len(entity_list_raw)==1:
            return_position.append(tokenizer_list.index(entity))
        else:
            temp_en = entity_list_raw
            temp_pos = []
            for k in range(len(tokenizer_list)):
                if tokenizer_list[k] == temp_en[0]:
                    for kk in range(len(temp_en)):
                        if tokenizer_list[k + kk] == temp_en[kk]:
                            temp_pos.append(k + kk)
                        else:
                            temp_pos = []
                            break
                    if temp_pos!=[]:
                        return_position.append(temp_pos)
                        temp_pos = []
    return return_position

def del_list(list1):
    list2=[]
    if len(list1)==1:
        return None
    for i,item in enumerate(list1):
        if i==0:
            if list1[i+1]-list1[i]==1:
                list2.append(item)
        elif i==len(list1)-1:
            if list1[i]-list1[i-1]==1:
                list2.append(item)
        else:
            if list1[i]-list1[i-1]==1 or list1[i+1]-list1[i]==1:
                list2.append(item)
    return list2

def t_token_pos(list1,list2):
    tag_1=0
    tag_2=0
    temp=''
    output=[]
    out_t=[]
    while tag_1<=len(list1) and tag_2<len(list2):
        if list2[tag_2]!=list1[tag_1] and list2[tag_2] in list1[tag_1]:
            out_t.append(tag_2)
            temp+=list2[tag_2]
            if temp==list1[tag_1]:
                tag_1+=1
                temp=''
                if len(out_t)>1:
                    output.append(out_t)
                    out_t=[]
            tag_2 += 1
        else:
            out_t=[]
            tag_1+=1
            tag_2+=1
    return output


def only_pro(claim,evidence):
    tokenized_input_seq_pair = tokenizer.encode_plus(claim, evidence,
                                                     max_length=512,
                                                     return_token_type_ids=True, truncation=True)
    input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0)
    token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0)
    attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0)
    outputs = model(input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=None)
    return torch.softmax(outputs[0], dim=1)[0].tolist()

def token_logit(claim, evidence, claim_entity):
    final_prob = {'entity': [], 'token': []}
    tokenized_input_seq_pair = tokenizer.encode_plus(claim, evidence,
                                                     max_length=512,
                                                     return_token_type_ids=True, truncation=True)

    tokenizer_list_raw = [t[1:] if t[0] == 'Ġ' else t for t in tokenizer.tokenize(claim)]
    # 将claim通过空格分割
    tokenizer_blank=claim.split()
    spilt_token=t_token_pos(tokenizer_blank,tokenizer_list_raw)
    input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0)

    token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0)
    attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0)
    outputs = model(input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=None)
    final_logit = torch.softmax(outputs[0], dim=1)[0].tolist()
    true_label = torch.Tensor([0]).long()
    loss = nn.CrossEntropyLoss()
    loss_value = loss(outputs.logits, true_label)
    loss_value.backward()

    for name, param in model.named_parameters():
        if param.requires_grad and name == 'roberta.encoder.layer.23.output.LayerNorm.weight':
            out_logit = param.grad[1:len(tokenizer_list_raw) + 1]
            temp_prob = nn.Softmax(dim=1)(torch.Tensor([out_logit.tolist()]))
            entity_pos = entity_position_com2(claim, claim_entity)
            token_pos = list(range(len(tokenizer_list_raw)))
            sp_token_pos=spilt_token
            id_token=0
            id_entity=0
            for nnn in entity_pos:
                if type(nnn) == int:
                    token_pos.remove(nnn)
                else:
                    for kk in nnn:
                        token_pos.remove(kk)
            sp_token_sum=[]
            for mm in sp_token_pos:
                for mn in mm:
                    sp_token_sum.append(mn)
            token_pos=[item for item in token_pos if item not in sp_token_sum]
            for n in token_pos:
                token_sum = {'token':tokenizer_list_raw[n],'id':id_token, 'pos': n, 'prob': temp_prob[0][n].item()}
                final_prob['token'].append(token_sum)
                id_token+=1
            for p in sp_token_pos:
                temp_p=''.join([tokenizer_list_raw[cc] for cc in p])
                if temp_p in claim_entity:
                    entity_sum = {'entity':temp_p, 'id': id_entity, 'pos': p, 'prob': temp_prob[0][p[0]:p[-1]].mean()}
                    if entity_sum['pos'] not in [mt['pos'] for mt in final_prob['entity']]:
                        final_prob['entity'].append(entity_sum)
                        id_entity+=1
                else:
                    token_flag=True
                    for ce in claim_entity:
                        if temp_p in ce:
                            token_flag=False
                    if token_flag:
                        p_token_sum = {'token':temp_p,'id':id_token, 'pos': p, 'prob': temp_prob[0][p[0]:p[-1]].mean()}
                        final_prob['token'].append(p_token_sum)
                        if p_token_sum['pos'] not in [mt['pos'] for mt in final_prob['token']]:
                            final_prob['token'].append(p_token_sum)
                            id_token+=1
            for e in entity_pos:
                if type(e) == int:
                    entity_sum = {'entity':tokenizer_list_raw[e],'id':id_entity, 'pos': e, 'prob': temp_prob[0][e]}
                    if entity_sum['pos'] not in [mt['pos'] for mt in final_prob['entity']]:
                        final_prob['entity'].append(entity_sum)
                        id_entity += 1
                else:
                    he_nn=[]
                    for nnn in sp_token_pos:
                        if set(nnn)<set(e):
                            for n in nnn[1:]:
                                he_nn.append(n)
                    zb_entity=''
                    for _,eee in enumerate(e):
                        if _!=0:
                            if eee not in he_nn:
                                zb_entity=zb_entity+' '+tokenizer_list_raw[eee]
                            else:
                                zb_entity=zb_entity+tokenizer_list_raw[eee]
                        else:
                            zb_entity=tokenizer_list_raw[eee]
                    entity_sum = {'entity':zb_entity ,'id':id_entity, 'pos': e, 'prob': temp_prob[0][e[0]:e[-1]].mean()}
                    if entity_sum['pos'] not in [mt['pos'] for mt in final_prob['entity']]:
                        final_prob['entity'].append(entity_sum)
                        id_entity += 1
    return tokenizer_list_raw,final_logit, final_prob, spilt_token




if __name__ == '__main__':
    claim = 'Joe Biden wins in the 2020 US election'
    evidence = 'In the 2020 US election, Democrat Joe Biden defeated Republican Donald Trump and was successfully elected as the 46th President of the United States.'
    print(entity_position_com2('Watertown, Massachusetts is in Massachusetts',['Massachusetts', 'Watertown']))

