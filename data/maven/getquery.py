import json
from nltk.stem.porter import PorterStemmer
import tqdm

def getlemma(word):
    word = word.replace('\n',' ')
    return lemma.stem(word)


def getquery(data,train=False,freq=0):
    sentencelist = getsentencelist(data)
    triggerset,triggernum,triggerdatas = gettriggerset(data)
    newtriggerset = {}
    for i in triggerset.keys():
        newtriggerset[i] = []
        for j in range(len(triggerset[i])):
            if triggernum[i][j] > freq:
                newtriggerset[i].append(triggerset[i][j])
    query1 = gethardquery(sentencelist,triggerset,train)
    return query1,triggerdatas

def getsentencelist(data):
    sentencelist = []
    for i in data.keys():
        for j in data[i]:
            if j not in sentencelist:
                sentencelist.append(j)
    print('get sentencelist')
    return sentencelist

def gettriggerset(data):
    triggersets = {}
    triggernums = {}
    triggerdatas = {}
    for i in data.keys():
        triggersetnum = []
        triggerset = []
        triggerdata = {}
        for j in data[i]:
            sentence = sentences[j]
            for event in sentence['event']:
                if event[0] == i:
                    triggertext = getlemma(event[1]['text'].lower())
                    if triggertext not in triggerset:
                        triggerset.append(triggertext)
                        triggersetnum.append(1)
                        triggerdata[triggertext] = [j]
                    else:
                        triggersetnum[triggerset.index(triggertext)] += 1
                        triggerdata[triggertext].append(j)
        triggersets[i] = triggerset
        triggernums[i] = triggersetnum
        triggerdatas[i] = triggerdata
    print('get triggerlist')
    return triggersets,triggernums,triggerdatas

# only event
def gethardquery(sentencelist,triggerset,train=False):
    print('begin to generate query')
    query = {}
    for i in tqdm.tqdm(sentencelist):
        sentence = sentences[i]
        lemmas = [getlemma(j.lower()) for j in sentence['words']]
        wordlabel = [0]*len(sentence['words'])
        for event in sentence['event']:
            start = event[1]['start']
            end = event[1]['end']
            for j in range(start,end):
                wordlabel[j] = event[0]
        for k in triggerset.keys():
            if k in wordlabel:
                continue
            else:
                for j in range(len(lemmas)):
                    if wordlabel[j] != k and wordlabel[j] != 0:
                        if lemmas[j] in triggerset[k] and lemmas[j] != 'it':
                            if train:
                                if k not in query.keys():
                                    query[k] = [[i,j,wordlabel[j]]]
                                else:
                                    query[k].append([i,j,wordlabel[j]])
                            else:
                                if k not in query.keys():
                                    query[k] = [i]
                                elif i not in query[k]:
                                    query[k].append(i)
    return query


lemma = PorterStemmer()
sentences = json.load(open('mavensentence.json'))
dev = json.load(open('mavendev.json'))

devquery,_ = getquery(dev)
json.dump(devquery,open('harddevquery.json','w'))