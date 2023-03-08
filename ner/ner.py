from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")
model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner")

from transformers import pipeline

nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")


def entity_extract(sentence):
    output=[]
    for item in nlp(sentence):
        temp_output = []
        for key in item:
            temp_output.append([key['word'],key['start'],key['end']])
        output.append(temp_output)
    return output

if __name__=='__main__':
    print(entity_extract(['SummerSlam is crushed by the WWE.', 'Alexandra Daddario was not born on March 16, 1986.']))
    print(entity_extract(['who are you','i']))