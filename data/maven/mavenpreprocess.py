import json
import tqdm
import numpy as np
import copy
import random

'''
dataformate
inputfile format:
the original MAVEN train set and dev set: trian.data, dev.data
link: https://github.com/THU-KEG/MAVEN-dataset

outputfile format:

mavensentence.json: the data is a list [sentence1,sentence2,...]:
sentence = {
    "words":tokenlist,
    "event":eventlist
}
eventlist = [event1,event2,..]
event1 = [label,{"text":triggerword,"start":startindex,"end":endindex+1}]

mavenfiltertrain.json: the data is a list [sentenceid1, sentenceid2, ...], sentenceid is the sentence id for train set
mavendev.json: the data is a list [sentenceid1, sentenceid2, ...], sentenceid is the sentence id for dev set
maventest.json: the data is a list [sentenceid1, sentenceid2, ...], sentenceid is the sentence id for test set

'''

SEED = 2333
random.seed(SEED)

def process(file,datas,sentences):
    with open(file) as f:
        lines = f.readlines()
        for i in tqdm.tqdm(range(len(lines)//4)):
            words = lines[i*4].strip().split(' ')
            pos = lines[i*4+1].strip().split(' ')
            events  = lines[i*4+2].strip()
            sid = lines[i*4+3].strip()
            eventsset = []
            sentenceindex = len(sentences)
            if len(events) > 0:
                events = events.split(' | ')
                for event in events:
                    position,label = event.split(' ')
                    start,end = position.split(',')
                    start = int(start)
                    end = int(end)
                    text = words[start:end]
                    if label not in datas.keys():
                        datas[label] = [sentenceindex]
                    else:
                        if sentenceindex not in datas[label]:
                            datas[label].append(sentenceindex)
                    eventsset.append([label,{'text':" ".join(text),'start':start,'end':end}])
                newsentence = {
                    'words':words,
                    'pos':pos,
                    'event':eventsset,
                    'sid':sid
                }
                sentences.append(newsentence)
    return datas,sentences

train = 'train.data'
dev = 'dev.data'

events = {}
sentences = []

events,sentences = process(train,events,sentences)
events,sentences = process(dev,events,sentences)

# trigger stat
length = [0] * 5
for sentence in sentences:
    for event in sentence['event']:
        length[len(event[1]['text'].split(' '))-1] += 1

# label stat
labelnum = {}
for label in events.keys():
    if label != 'NA':
        labelnum[label] = len(events[label])

# co-occu
co = np.zeros((len(events.keys()),len(events.keys())))
label2id = {}
for label in events.keys():
    label2id[label] = len(label2id.keys())

for sentence in tqdm.tqdm(sentences):
    eventset = []
    for event in sentence['event']:
        if event[0] not in eventset:
            eventset.append(event[0])
    if len(eventset) > 1:
        for i in range(len(eventset)-1):
            eventi = label2id[eventset[i]]
            for j in range(i+1,len(eventset)):
                eventj = label2id[eventset[j]]
                if eventi < eventj:
                    co[eventi,eventj] += 1
                else:
                    co[eventj,eventi] += 1
clusters = []
num = len(events.keys())
clusters.append([[i] for i in range(num)])
newco = copy.copy(co)
while True:
    if np.max(newco) < 50:
        break
    start = np.max(newco,axis=1)
    start = np.argmax(start)
    end = np.argmax(newco[start])
    index1 = min(start,end)
    index2 = max(start,end)
    newcluster = copy.copy(clusters[-1])
    newco[index1,index2] = 0
    newcluster[index1] = copy.copy(newcluster[index1]) + copy.copy(newcluster[index2])
    newcluster[index2] = []
    clusters.append(newcluster)
    newco1 = newco[index1]
    newco2 = newco[index2]
    cat = np.max(np.array([newco1,newco2]),axis=0)
    newco[index1] = cat
    newco[index2] = 0


json.dump(sentences, open('mavensentence.json','w'))
json.dump(events, open('mavenfewshot.json','w'))

labelnum = np.array([len(events[i]) for i in events.keys()])
labelindex = np.argsort(-labelnum)

train = labelindex[:120]
test = labelindex[120:]
labellist = list(events.keys())
train = [labellist[i] for i in train]
test = [labellist[i] for i in test]


trainevent = {}
devevent = {}
testevent = {}

event = events

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



json.dump(trainevent, open('maventrain.json','w'))
json.dump(devevent, open('mavendev.json','w'))
json.dump(testevent, open('maventest.json','w'))

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
json.dump(newtrain, open('mavenfiltertrain.json','w'))

newdev = {}
for i in devevent.keys():
    print(len(devevent[i]))
    newdev[i] = []
    for j in devevent[i]:
        if j not in filtersentence:
            newdev[i].append(j)
    print(len(devevent[i]))
json.dump(newdev, open('mavenfilterdev.json','w'))

newtest = {}
for i in testevent.keys():
    print(len(testevent[i]))
    newtest[i] = []
    for j in testevent[i]:
        if j not in filtersentence:
            newtest[i].append(j)
    print(len(testevent[i]))
json.dump(newtest, open('mavenfiltertest.json','w'))


    
    

