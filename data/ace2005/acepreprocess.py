import json
import random
import numpy as np


'''
dataformate
inputfile format:
train.json, dev.json and test.json: the data is a list [instance1,instance2,...]
instance1 = {
    "words": [token1, token2, ...], # token list
    "golden-event-mentions": [event1, event2, ...], # event list
}
event1 = {
    "trigger": {
        "text": trigger word,
        "start": the start index in the token list,
        "end": the end index in the token list, (trigger does not includes this index)
    },
     "event_type": "type", # for example "Personnel:Nominate" 
}


outputfile format:

acesentence.json: the data is a list [sentence1,sentence2,...]:
sentence = {
    "words":tokenlist,
    "event":eventlist
}
eventlist = [event1,event2,..]
event1 = [label,{"text":triggerword,"start":startindex,"end":endindex+1}]

acefiltertrain.json: the data is a list [sentenceid1, sentenceid2, ...], sentenceid is the sentence id for train set
acedev.json: the data is a list [sentenceid1, sentenceid2, ...], sentenceid is the sentence id for dev set
acetest.json: the data is a list [sentenceid1, sentenceid2, ...], sentenceid is the sentence id for test set

'''


def loaddata(filename,data,event):
    index = len(data)
    file = json.load(open(filename))
    for i in file:
        if len(i["golden-event-mentions"]) > 0:
            newevent = []
            for j in i["golden-event-mentions"]:
                newevent.append([j["event_type"],j["trigger"]])
                if j["event_type"] not in event.keys():
                    event[j["event_type"]] = [index]
                else:
                    event[j["event_type"]].append(index)
            index += 1
            newdata = {
                "words":i["words"],
                "event":newevent
            }
            data.append(newdata)
    return data,event


train = 'train.json'
dev = 'dev.json'   
test = 'test.json'

data = []
event = {}

data,event = loaddata(train,data,event)
data,event = loaddata(dev,data,event)
data,event = loaddata(test,data,event)

json.dump(data, open('acesentence.json','w'))
json.dump(event, open('aceevent.json','w'))


labelnum = np.array([len(event[i]) for i in event.keys()])
labelindex = np.argsort(-labelnum)

train = labelindex[:20]
test = labelindex[20:]
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



json.dump(trainevent, open('acetrain.json','w'))
json.dump(devevent, open('acedev.json','w'))
json.dump(testevent, open('acetest.json','w'))

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
json.dump(newtrain, open('acefiltertrain.json','w'))