import json
import os
import numpy as np
import random

'''
dataformate
original file is
LDC2017E55.df.en
LDC2017E02.2015.eval.en
LDC2017E02.2016.eval.nw.en
LDC2015E29.en
LDC2015E78.en
LDC2017E02.2015.train.en
LDC2016E31.df.en
LDC2017E55.nw.en
LDC2015E68.en
LDC2017E02.2016.eval.df.en

inputfile format:
Convert original to xx.jsonl

each line is a document (dictionary format)
{"sentences" [sentence1,sentence2,...],"event":[event1,event2]}
sentence1 = {"tokens": [token1,token2,...]}
token1 = {"word":token}
event1 = {"mentions":[mention1,mention2]}
mention1 = {"type":event type, "subtype": event sub type, "nugget": {"text":trigger word, "tokens": [[sentenceindex,token1index],[sentenceindex,token2index],...]}}

outputfile format:

kbpsentence.json: the data is a list [sentence1,sentence2,...]:
sentence = {
    "words":tokenlist,
    "event":eventlist
}
eventlist = [event1,event2,..]
event1 = [label,{"text":triggerword,"start":startindex,"end":endindex+1}]

kbpfiltertrain.json: the data is a list [sentenceid1, sentenceid2, ...], sentenceid is the sentence id for train set
kbpdev.json: the data is a list [sentenceid1, sentenceid2, ...], sentenceid is the sentence id for dev set
kbptest.json: the data is a list [sentenceid1, sentenceid2, ...], sentenceid is the sentence id for test set

'''

def kbppreprocess(file,data,kbpevent):
    labellist = []
    with open(file) as f:
        for line in f:
            line = json.loads(line)
            instances = []
            for index,sentence in enumerate(line['sentences']):
                instance = {
                    'words': [i['word'] if len(i['word']) < 20 else "," for i in sentence['tokens']],
                    'event': []
                }
                instances.append(instance)
            for event in line['event']:
                for mention in event['mentions']:
                    eventtype = mention['type'] + ':' + mention['subtype']
                    if eventtype not in labellist:
                        labellist.append(eventtype)
                    nugget = mention['nugget']
                    if len(nugget['tokens']) <=0 :
                        continue
                    trigger = {
                        'text' : nugget['text'],
                        'start': nugget['tokens'][0][1],
                        'end': nugget['tokens'][-1][1] + 1,
                    }
                    assert trigger['end'] - trigger['start'] == len(nugget['tokens'])
                    eventinstance = [eventtype,trigger]

                    sentid = nugget['tokens'][0][0]
                    instances[sentid]['event'].append(eventinstance)
            for instance in instances:
                if len(instance['event']) > 0 and len(instance['words']) < 200:
                    for event in instance['event']:
                        if event[0] not in kbpevent.keys():
                            kbpevent[event[0]] = [len(data)]
                        else:
                            kbpevent[event[0]].append(len(data))
                    data.append(instance)
    return data,kbpevent

kbpdata = []
kbpevent = {}
for filename in os.listdir():
    if filename[-5:] == 'jsonl':
        kbpdata,kbpevent = kbppreprocess(filename,kbpdata,kbpevent)
json.dump(kbpdata,open('kbpsentence.json','w'))
json.dump(kbpevent,open('kbpevent.json','w'))


event = kbpevent
labelnum = np.array([len(event[i]) for i in event.keys()])
labelindex = np.argsort(-labelnum)

train = labelindex[:25]
test = labelindex[25:]
labellist = list(event.keys())
train = [labellist[i] for i in train]
test = [labellist[i] for i in test]

trainevent = {}
devevent = {}
testevent = {}

SEED = 0
random.seed(SEED)

for i in event.keys():
    if len(event[i]) < 12:
        continue
    for j in train:
        if j in i:
            trainevent[i] = event[i]
            break
    for j in test:
        if j in i:
            length = len(event[i])
            e = event[i]
            random.shuffle(e)
            devevent[i] = e[:length//2]
            testevent[i] = e[length//2:]
            assert len(devevent[i]) + len(testevent[i]) == length



json.dump(trainevent, open('kbptrain.json','w'))
json.dump(devevent, open('kbpdev.json','w'))
json.dump(testevent, open('kbptest.json','w'))

testsentence = []
for i in devevent.keys():
    for j in devevent[i]:
        if j not in testsentence:
            testsentence.append(j)
for i in testevent.keys():
    for j in testevent[i]:
        if j not in testsentence:
            testsentence.append(j)

filtersentence = []
newtrain = {}
for i in trainevent.keys():
    print(len(trainevent[i]))
    newtrain[i] = []
    for j in trainevent[i]:
        if j not in testsentence:
            newtrain[i].append(j)
        else:
            filtersentence.append(j)
    print(len(newtrain[i]))
json.dump(newtrain, open('kbpfiltertrain.json','w'))