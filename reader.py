import torch
import torch.utils.data as data
import os
import numpy as np
import random
import json
from typing import Dict
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, ArrayField, ListField, MetadataField
from allennlp.data.tokenizers import Token
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.common import Tqdm, util
from tqdm import tqdm
from transformers import BertTokenizer
from pipelines import pipeline
from random import shuffle,choice
import copy
from nltk.stem.porter import PorterStemmer
import math


# reader for FS-base
class FewEventDetection_trainbaselinereader(DatasetReader):
    '''
    K : the number of instances in support set
    Q : the number of instances in query
    noise_lenth: the numebr of negative type
    maxlength: the max length of sentences
    instancenum: epoch number = instancenum * labelnum
    '''
    def __init__(self,
                 K, Q, noise_length=2,maxlength=60,sentence="data/ACE2005/acesentence.json",instancenum=1,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self.K = K
        self.Q = Q
        self.json_data = None
        self.noiselength = noise_length
        self.sentence = json.load(open(sentence))
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.maxlength = maxlength
        self.instancenum = instancenum
        print("train baseline reader")
        
    def convert_list_to_token(self,tokenlist):
        tokens = []
        tokenmap = []
        for i in tokenlist:
            tokenmap.append(len(tokens)+1)
            tokens += i
        tokenmap.append(len(tokens)+1)
        return tokens,tokenmap
    
    def compress(self,data,class_name):
        '''
        compress long sentence
        '''
        if len(data['words']) <= self.maxlength:
            return data
        else:
            newdata = {}
            eventset = []
            for i in data['event']:
                if i[0] == class_name:
                    eventset.append(i[1])
            event = random.choice(eventset)
            triggerindex = event['start']
            newstart1 = max(triggerindex-int(self.maxlength*(0.8)),0)
            newstart2 = max(triggerindex-int(self.maxlength*(0.2)),0)
            newstart = random.choice(list(range(newstart1,newstart2+1)))
            newdata['words'] = data['words'][newstart:newstart+self.maxlength]
            newdata['event'] = []
            for i in data['event']:
                if i[1]['start'] - newstart >= 0 and i[1]['end'] - newstart <= self.maxlength:
                    newevent = [i[0],{'text':i[1]['text'],'start':i[1]['start']-newstart,'end':i[1]['end']-newstart}]
                    newdata['event'].append(newevent)
            return newdata

    def gettokenizer(self, raw_data, eventtype, query = False):
        # token -> index
        raw_tokens = copy.copy(raw_data["words"])
        origin_tokens = []
        for token in raw_tokens:
            token = token.lower()
            origin_tokens.append(self.tokenizer.tokenize(token))
        length = len(origin_tokens)
        wordlabel = np.zeros((length))
        sp_tokens = copy.copy(origin_tokens)
        sp_tokens,tokenmap = self.convert_list_to_token(sp_tokens)
        sp_tokens = self.tokenizer.encode_plus(sp_tokens)
        mask = sp_tokens["attention_mask"]
        tokens = sp_tokens["input_ids"] 
        poslabel = []
        neglabel = []
        posindex = []
        negindex = []
        if query:
            for event in copy.copy(raw_data["event"]):
                start = event[1]['start']
                end = event[1]['end']
                if event[0] == eventtype:
                    wordlabel[start:end] = 1
            for i in range(length):
                if wordlabel[i] == 0:
                    negindex.append([tokenmap[i],tokenmap[i+1]])
                    neglabel.append(0)
                else: 
                    posindex.append([tokenmap[i],tokenmap[i+1]])
                    poslabel.append(1)
        else:
            for event in copy.copy(raw_data["event"]):
                start = event[1]['start']
                end = event[1]['end']
                if event[0] == eventtype:
                    posindex.append([tokenmap[start],tokenmap[end]])
                    wordlabel[start:end] = 1
                    poslabel.append(1)
            for i in range(length):
                if wordlabel[i] == 0:
                    negindex.append([tokenmap[i],tokenmap[i+1]])
                    neglabel.append(0)
        return tokens,mask,posindex,negindex,poslabel,neglabel

    def get_trainingdata(self,dataes,class_name, query = False):
        result = []
        for data in dataes:
            tokens,mask,posindex,negindex,poslabel,neglabel = self.gettokenizer(copy.copy(data),class_name,query)
            result.append([tokens,mask,posindex,negindex,poslabel,neglabel])
        return result

    def bert_data(self,class_name):
        support_set = {'data': [], 'word': [], 'mask' : [], 'index':[], 'label': [],'triggerlabel':[]}
        query_set = {'data': [], 'word':[], 'index': [], 'mask': [], 'label': [],'triggerlabel':[],'triggerset':[]}


        positive_set = copy.copy(self.json_data[class_name])
        negative_class =  random.sample(list(filter(lambda x:x!=class_name,self.classes)),self.noiselength)
        negative_list = []
        for i in negative_class:
            class_set = list(filter(lambda x:x not in positive_set, copy.copy(self.json_data[i])))
            sampleset = random.sample(class_set,min(self.Q,len(class_set)))
            for j in sampleset:
                instance = self.compress(copy.copy(self.sentence[j]),i)
                negative_list.append(instance)

        shuffle(positive_set)

        positive_set = copy.copy(positive_set[:self.K+self.Q])

        positive_list = []
        for i in positive_set:
            instance = self.compress(copy.copy(self.sentence[i]),class_name)
            positive_list.append(instance)

        support_list = positive_list[:self.K]
        query_list = positive_list[self.K:self.K+self.Q] 

        supports = self.get_trainingdata(support_list,class_name)

        
        for support in supports:
            tokens,mask,posindex,negindex,poslabel,neglabel = support
            index = posindex + negindex
            label = poslabel + neglabel
            support_set['word'].append(ArrayField(np.array(tokens)))
            support_set['mask'].append(ArrayField(np.array(mask)))
            support_set['index'].append(ArrayField(np.array(index)))
            support_set['label'].append(ListField([LabelField(i,skip_indexing=True) for i in label]))

        query_list += negative_list
        query_list = self.get_trainingdata(query_list,class_name,query=True)

        for query in query_list:
            tokens,mask,posindex,negindex,poslabel,neglabel = query
            index = posindex + negindex
            label = poslabel + neglabel
            query_set['index'].append(ArrayField(np.array(index)))
            query_set['word'].append(ArrayField(np.array(tokens)))
            query_set['mask'].append(ArrayField(np.array(mask)))
            query_set['label'].append(ListField([LabelField(i,skip_indexing=True) for i in label]))

        batch_data = {
            'support': support_set,
            'query': query_set,
        }
        return batch_data
    
    def text_to_instance(self, data) -> Instance:  # type: ignore
        fields = {
            "support_tokenid": ListField(data['support']['word']),
            "support_mask": ListField(data['support']['mask']),
            "support_index": ListField(data['support']['index']),
            "support_label": ListField(data['support']['label']),
            "query_tokenid": ListField(data['query']['word']),
            "query_mask": ListField(data['query']['mask']),
            "query_label": ListField(data['query']['label']),
            "query_index": ListField(data['query']['index']),
        }
        return Instance(fields)
    
    def initdata(self,filepath):
        # filepath = eval(filepath)
        if type(filepath) is dict:
            self.json_data = filepath
        else:
            if not os.path.exists(filepath):
                print("[ERROR] Data file does not exist!")
                assert(0)
            self.json_data = json.load(open(filepath))
        self.sentencelist = []
        for i in self.json_data.keys():
            for j in self.json_data[i]:
                if j not in self.sentencelist:
                    self.sentencelist.append(j)
        self.classes = list(self.json_data.keys())

    def _read(self, filepath):
        if self.json_data is None:
            self.initdata(filepath)
        shuffle(self.classes)
        for i in range(self.instancenum):
            for classname in self.classes:
                yield self.text_to_instance(self.bert_data(classname))


# reader for FS-causal
class FewEventDetection_traincausalreader(DatasetReader):
    '''
    K : the number of instances in support set
    Q : the number of instances in query
    noise_lenth: the numebr of negative type
    maxlength: the max length of sentences
    instancenum: epoch number = instancenum * labelnum
    device: the device for bert fill-mask pipeline
    originp: the probability of original trigger word
    '''
    def __init__(self,
                 K, Q, noise_length=2,maxlength=60,sentence="data/ACE2005/acesentence.json",instancenum=1,device=0,originp = 0.5,backdooruse = 'support',
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self.K = K
        self.Q = Q
        self.json_data = None
        self.noiselength = noise_length
        self.sentence = json.load(open(sentence))
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.originp = originp
        self.backdooruse = backdooruse
        try:
            self.pipe = pipeline('fill-mask',model='bert-base-uncased',tokenizer='bert-base-uncased',device=device,top_k=5)
        except:
            self.pipe = pipeline('fill-mask',model='bert-base-uncased',tokenizer='bert-base-uncased',device=-1,top_k=5)
        self.maxlength = maxlength
        self.instancenum = instancenum
        print("train causal reader")

    def convert_list_to_token(self,tokenlist):
        tokens = []
        tokenmap = []
        for i in tokenlist:
            tokenmap.append(len(tokens)+1)
            tokens += i
        tokenmap.append(len(tokens)+1)
        return tokens,tokenmap
    
    def compress(self,data,class_name):
        if len(data['words']) <= self.maxlength:
            return data
        else:
            newdata = {}
            eventset = []
            for i in data['event']:
                if i[0] == class_name:
                    eventset.append(i[1])
            event = random.choice(eventset)
            triggerindex = event['start']
            newstart1 = max(triggerindex-int(self.maxlength*(0.8)),0)
            newstart2 = max(triggerindex-int(self.maxlength*(0.2)),0)
            newstart = random.choice(list(range(newstart1,newstart2+1)))
            newdata['words'] = data['words'][newstart:newstart+self.maxlength]
            newdata['event'] = []
            for i in data['event']:
                if i[1]['start'] - newstart >= 0 and i[1]['end'] - newstart <= self.maxlength:
                    newevent = [i[0],{'text':i[1]['text'],'start':i[1]['start']-newstart,'end':i[1]['end']-newstart}]
                    newdata['event'].append(newevent)
            return newdata

    def gettokenizer(self, raw_data, eventtype, query = False):
        # token -> index
        raw_tokens = copy.copy(raw_data["words"])
        origin_tokens = []
        for token in raw_tokens:
            token = token.lower()
            origin_tokens.append(self.tokenizer.tokenize(token))
        length = len(origin_tokens)
        wordlabel = np.zeros((length))
        sp_tokens = copy.copy(origin_tokens)
        sp_tokens,tokenmap = self.convert_list_to_token(sp_tokens)
        sp_tokens = self.tokenizer.encode_plus(sp_tokens)
        mask = sp_tokens["attention_mask"]
        tokens = sp_tokens["input_ids"] 
        positive = []
        negative = []
        none = []
        triggerset = []
        poslabel = []
        neglabel = []
        index = []
        posindex = []
        negindex = []
        negevent = []
        posevent = []
        if query:
            for event in copy.copy(raw_data["event"]):
                start = event[1]['start']
                end = event[1]['end']
                if event[0] == eventtype:
                    wordlabel[start:end] = 2
                    label = 1
                else:
                    wordlabel[start:end] = 1
                    label = 0
                for i in range(start,end):
                    if label == 1:
                        posevent.append([tokenmap[i],tokenmap[i+1],i,i+1,event[0]])
                    else:
                        negevent.append([tokenmap[i],tokenmap[i+1],i,i+1,event[0]])
            negevent = random.sample(negevent,min(len(negevent),3))
            posindex = posevent + negevent
            poslabel = [1]*len(posevent) + [0] * len(negevent)
            for i in range(length):
                if wordlabel[i] == 0:
                    negindex.append([tokenmap[i],tokenmap[i+1]])
                    neglabel.append(0)
        else:
            for event in copy.copy(raw_data["event"]):
                start = event[1]['start']
                end = event[1]['end']
                if event[0] == eventtype:
                    posindex.append([tokenmap[start],tokenmap[end],start,end,event[0]])
                    wordlabel[start:end] = 1
                    poslabel.append(1)
            for i in range(length):
                if wordlabel[i] == 0:
                    negindex.append([tokenmap[i],tokenmap[i+1]])
                    neglabel.append(0)
        return tokens,mask,posindex,negindex,poslabel,neglabel,raw_tokens
    
    def addtrigger(self,rawtoken,start,end,originp,token,index1,index2,eventtype):
        newtoken = copy.copy(rawtoken)
        newtoken = newtoken[:start] +  ['[MASK]'] + newtoken[end:]
        newtoken = " ".join(newtoken)
        newtoken = self.pipe(newtoken)
        triggerset = [[[i['token']],i['score']] for i in newtoken]
        index = [index1,index2]
        return self.addtriggerset(token,index,triggerset,originp)
    
    def addtriggerset(self,token,index,triggerset,originp):
        newtokens = []
        newmasks = []
        newindexs = []
        score = []
        for i in triggerset:
            newtoken = token[:index[0]] + i[0] + token[index[1]:]
            newmask = [1]*len(newtoken)
            newindex = [index[0],index[0]+len(i[0])]
            newtokens.append(newtoken)
            newmasks.append(newmask)
            newindexs.append(newindex)
            score.append(i[1])
        score = torch.Tensor(score)
        score = torch.softmax(score,dim=-1)*(1-originp)
        score = score.tolist()
        triggermasks = [originp] + score
        return newtokens,newmasks,newindexs,triggermasks

    def get_trainingdata(self,dataes,class_name, query = False):
        result = []
        for data in dataes:
            tokens,mask,posindex,negindex,poslabel,neglabel,raw_tokens = self.gettokenizer(copy.copy(data),class_name,query)
            result.append([tokens,mask,posindex,negindex,poslabel,neglabel,raw_tokens])
        return result

    def bert_data(self,class_name):
        support_set = {'data': [], 'word': [], 'mask' : [], 'index':[], 'label': [],'posword':[],'posmask':[],'posindex':[],'poslabel':[],'triggerp':[]}
        query_set = {'data': [], 'word':[], 'index': [], 'mask': [], 'label': [],'eventword':[], 'eventindex': [], 'eventmask': [], 'eventlabel': [],'triggerp':[]}


        positive_set = copy.copy(self.json_data[class_name])
        negative_class =  random.sample(list(filter(lambda x:x!=class_name,self.classes)),self.noiselength)
        negative_list = []
        for i in negative_class:
            class_set = list(filter(lambda x:x not in positive_set, copy.copy(self.json_data[i])))
            sampleset = random.sample(class_set,min(self.Q,len(class_set)))
            for j in sampleset:
                instance = self.compress(copy.copy(self.sentence[j]),i)
                negative_list.append(instance)

        shuffle(positive_set)

        positive_set = copy.copy(positive_set[:self.K+self.Q])

        positive_list = []
        for i in positive_set:
            instance = self.compress(copy.copy(self.sentence[i]),class_name)
            positive_list.append(instance)

        support_list = positive_list[:self.K]
        query_list = positive_list[self.K:self.K+self.Q] 

        support_set['data'] = positive_list[:self.K]

        supports = self.get_trainingdata(support_list,class_name)

        
        for support in supports:
            tokens,mask,posindex,negindex,poslabel,neglabel,rawtoken = support
            support_set['word'].append(ArrayField(np.array(tokens)))
            support_set['mask'].append(ArrayField(np.array(mask)))
            support_set['index'].append(ArrayField(np.array(negindex)))
            support_set['label'].append(ListField([LabelField(i,skip_indexing=True) for i in neglabel]))
            originp = self.originp
            for i in range(len(posindex)):
                index1,index2,start,end,eventtype = posindex[i]
                if 'support' in self.backdooruse and originp < 1:
                    newtokens,newmasks,newindexs,triggermasks = self.addtrigger(rawtoken,start,end,originp,tokens,index1,index2,eventtype)
                else:
                    newtokens = []
                    newmasks = []
                    newindexs = []
                    triggermasks = [1]
                newtoken = [tokens] + newtokens
                newmask = [mask] + newmasks
                newindex = [[index1,index2]] + newindexs
                support_set['posword'].append(ListField([ArrayField(np.array(i)) for i in newtoken]))
                support_set['posmask'].append(ListField([ArrayField(np.array(i)) for i in newmask]))
                support_set['posindex'].append(ArrayField(np.array(newindex)))
                support_set['triggerp'].append(ArrayField(np.array(triggermasks)))

        query_list += negative_list

        query_set['data'] = query_list
        query_list = self.get_trainingdata(query_list,class_name,query=True)

        for query in query_list:
            tokens,mask,posindex,negindex,poslabel,neglabel,rawtoken = query
            query_set['index'].append(ArrayField(np.array(negindex)))
            query_set['word'].append(ArrayField(np.array(tokens)))
            query_set['mask'].append(ArrayField(np.array(mask)))
            query_set['label'].append(ListField([LabelField(i,skip_indexing=True) for i in neglabel]))
            originp = self.originp
            for i in range(len(posindex)):
                index1,index2,start,end,eventtype = posindex[i]
                if 'query' in self.backdooruse and originp < 1:
                    newtokens,newmasks,newindexs,triggermasks = self.addtrigger(rawtoken,start,end,originp,tokens,index1,index2,eventtype)
                else:
                    newtokens = []
                    newmasks = []
                    newindexs = []
                    triggermasks = [1]
                newtoken = [tokens] + newtokens
                newmask = [mask] + newmasks
                newindex = [[index1,index2]] + newindexs
                query_set['eventword'].append(ListField([ArrayField(np.array(i)) for i in newtoken]))
                query_set['eventmask'].append(ListField([ArrayField(np.array(i)) for i in newmask]))
                query_set['eventindex'].append(ArrayField(np.array(newindex)))
                query_set['triggerp'].append(ArrayField(np.array(triggermasks)))
                query_set['eventlabel'].append(LabelField(poslabel[i],skip_indexing=True))

        batch_data = {
            'support': support_set,
            'query': query_set,
        }
        return batch_data
    
    def text_to_instance(self, data) -> Instance:  # type: ignore
        fields = {
            "support_tokenid": ListField(data['support']['word']),
            "support_mask": ListField(data['support']['mask']),
            "support_index": ListField(data['support']['index']),
            "support_label": ListField(data['support']['label']),
            "support_postokenid": ListField(data['support']['posword']),
            "support_posmask": ListField(data['support']['posmask']),
            "support_posindex": ListField(data['support']['posindex']),
            "support_triggerp": ListField(data['support']['triggerp']),
            "query_tokenid": ListField(data['query']['word']),
            "query_mask": ListField(data['query']['mask']),
            "query_label": ListField(data['query']['label']),
            "query_index": ListField(data['query']['index']),
            "query_eventtokenid": ListField(data['query']['eventword']),
            "query_eventmask": ListField(data['query']['eventmask']),
            "query_eventlabel": ListField(data['query']['eventlabel']),
            "query_eventindex": ListField(data['query']['eventindex']),
            "query_triggerp": ListField(data['query']['triggerp']),
        }
        return Instance(fields)
    
    def initdata(self,filepath):
        if type(filepath) is dict:
            self.json_data = filepath
        else:
            if not os.path.exists(filepath):
                print("[ERROR] Data file does not exist!")
                assert(0)
            self.json_data = json.load(open(filepath))
        self.sentencelist = []
        for i in self.json_data.keys():
            for j in self.json_data[i]:
                if j not in self.sentencelist:
                    self.sentencelist.append(j)
        self.classes = list(self.json_data.keys())

    def _read(self, filepath):
        if self.json_data is None:
            self.initdata(filepath)
        shuffle(self.classes)
        for i in range(self.instancenum):
            for classname in self.classes:
                yield self.text_to_instance(self.bert_data(classname))

# reader for test 
class FewEventDetection_testbaselinereader(DatasetReader):
    '''
        K : the number of instances in support set
        instancenum: the number of times to sample the support set
    '''
    def __init__(self,
                 K,sentence="data/ACE2005/acesentence.json",instancenum=5 ,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self.K = K
        self.json_data = None
        self.sentence = json.load(open(sentence))
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.instancenum = instancenum
        print('test baseline reader')
        
    def convert_list_to_token(self,tokenlist):
        tokens = []
        tokenmap = []
        for i in tokenlist:
            tokenmap.append(len(tokens)+1)
            tokens += i
        tokenmap.append(len(tokens)+1)
        return tokens,tokenmap

    def gettokenizer(self, raw_data, eventtype, query = False):
        # token -> index
        raw_tokens = copy.copy(raw_data["words"])
        origin_tokens = []
        for token in raw_tokens:
            token = token.lower()
            origin_tokens.append(self.tokenizer.tokenize(token))
        length = len(origin_tokens)
        wordlabel = np.zeros((length))
        sp_tokens = copy.copy(origin_tokens)
        sp_tokens,tokenmap = self.convert_list_to_token(sp_tokens)
        sp_tokens = self.tokenizer.encode_plus(sp_tokens)
        mask = sp_tokens["attention_mask"]
        tokens = sp_tokens["input_ids"] 

        positive = []
        negative = []
        none = []
        label = []
        triggerset = []
        if not query:
            for event in copy.copy(raw_data["event"]):
                start = event[1]['start']
                end = event[1]['end']
                if event[0] == eventtype:
                    wordlabel[start:end] = 1
                    positive.append([tokenmap[i] for i in range(start,end+1)])
                else:
                    negative.append([tokenmap[start],tokenmap[end],1,0])
            posindex = random.sample(positive,1)[0]
            index = [[posindex[0],posindex[-1]]]
            label.append(1)
            for i in range(length):
                if wordlabel[i] == 0:
                    index.append([tokenmap[i],tokenmap[i+1]])
                    label.append(0)
        else:
            index = []
            for event in copy.copy(raw_data["event"]):
                start = event[1]['start']
                end = event[1]['end']
                triggerset.append([event[0],start,end-1])
            for i in range(length):
                index.append([tokenmap[i],tokenmap[i+1]])
                label.append(0)
        return tokens,mask,index,triggerset,label

    def get_testingdata(self,dataes,class_name,query):
        datalist = []
        for data in dataes:
            token,mask,index,triggerset,label = self.gettokenizer(copy.copy(self.sentence[data]),class_name, query = query)
            datalist.append([token,mask,index,triggerset,label])
        return datalist
    

    def bert_data(self,class_name):
        support_set = {'data': [], 'word': [], 'mask' : [], 'label': [],'triggerlabel':[],'index':[]}
        query_set = {'data': [], 'wordindex':[], 'word': [], 'mask': [], 'label': [],'triggerlabel':[],'triggerset':[],'index':[]}

        positive_list = copy.copy(self.json_data[class_name])
        negative_list = [i for i in self.sentencelist if i not in positive_list]

        shuffle(positive_list)
        support_list = positive_list[:self.K]

        query_list = positive_list[self.K:] 
        query_list += negative_list
        supportlist = self.get_testingdata(support_list,class_name, query = False)

        for support in supportlist:
            token,mask,index,_,label = support
            support_set['word'].append(ArrayField(np.array(token)))
            support_set['mask'].append(ArrayField(np.array(mask)))
            support_set['index'].append(ArrayField(np.array(index)))
            support_set['label'].append(ListField([LabelField(i,skip_indexing=True) for i in label]))
        
        for query in query_list:
            token,mask,index,triggerset,label = self.sentenceinstance[query]
            query_set['word'].append(ArrayField(np.array(token)))
            query_set['mask'].append(ArrayField(np.array(mask)))
            query_set['index'].append(ArrayField(np.array(index)))
            newtriggerset = []
            query_set['label'].append(ListField([LabelField(i,skip_indexing=True) for i in label]))
            for i in triggerset:
                if i[0] == class_name:
                    newtriggerset.append([1,i[1],i[2]])
            query_set['triggerset'].append(MetadataField(newtriggerset))

        batch_data = {
            'support': support_set,
            'query': query_set,
            'supportlist':support_list,
            'querylist':query_list,
        }
        return batch_data

    def text_to_instance(self, data) -> Instance:  # type: ignore
        fields = {
            "support_tokenid": ListField(data['support']['word']),
            "support_mask": ListField(data['support']['mask']),
            "support_index": ListField(data['support']['index']),
            "support_label": ListField(data['support']['label']),
            "query_tokenid": ListField(data['query']['word']),
            "query_mask": ListField(data['query']['mask']),
            "query_index": ListField(data['query']['index']),
            "query_label": ListField(data['query']['label']),
            "query_triggerset": ListField(data['query']['triggerset']),
        }
        return Instance(fields)
    
    def getinstance(self):
        sentenceinstance = [0]*(max(self.sentencelist)+1)
        for i in self.sentencelist:
            sentenceinstance[i] = self.gettokenizer(self.sentence[i], None, query = True)
        return sentenceinstance
    
    def initdata(self,filepath):
        # filepath = eval(filepath)
        if type(filepath) is dict:
            self.json_data = filepath
        else:
            if not os.path.exists(filepath):
                print("[ERROR] Data file does not exist!")
                assert(0)
            self.json_data = json.load(open(filepath))
        self.sentencelist = []
        for i in self.json_data.keys():
            for j in self.json_data[i]:
                if j not in self.sentencelist:
                    self.sentencelist.append(j)
        self.classes = list(self.json_data.keys())
        self.sentenceinstance = self.getinstance()

    def getnewdata(self,bertdata,index1,index2):
        newbert = {
            "support":bertdata['support']
        }
        newbert['query'] = {
            'word':bertdata['query']['word'][index1:index2],
            'index':bertdata['query']['index'][index1:index2],
            'mask':bertdata['query']['mask'][index1:index2],
            'label':bertdata['query']['label'][index1:index2],
            'triggerset':bertdata['query']['triggerset'][index1:index2],
        }
        return newbert
    
    def setclass(self,classname):
        self.nowclassname = classname

    def _read(self, filepath):
        if self.json_data is None:
            self.initdata(filepath)
        batch = 64
        for i in range(self.instancenum): 
            bertdata = self.bert_data(self.nowclassname)
            length = len(bertdata['query']['word'])
            for j in range(math.ceil(length/batch)):
                yield self.text_to_instance(self.getnewdata(bertdata,j*batch,(j+1)*batch))

class FewEventDetection_devquerybaselinereader(DatasetReader):
    '''
    K : the number of instances in support set
    posnum : the number of positive instances in query
    negativerate: the numebr of negative instance in query =  posnum * negativerate
    instancenum: epoch number = instancenum * labelnum
    query: the file for "ambiguous negative data"
    '''
    def __init__(self,
                 K, posnum, negativerate,sentence="data/ACE2005/acesentence.json",query=None,instancenum=5 ,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self.K = K
        self.json_data = None
        self.sentence = json.load(open(sentence))
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.posnum = posnum
        self.negativerate = negativerate
        self.query = json.load(open(query))
        self.instancenum = instancenum
        print('dev query reader')
        
    def convert_list_to_token(self,tokenlist):
        tokens = []
        tokenmap = []
        for i in tokenlist:
            tokenmap.append(len(tokens)+1)
            tokens += i
        tokenmap.append(len(tokens)+1)
        return tokens,tokenmap

    def gettokenizer(self, raw_data, eventtype, query = False):
        # token -> index
        raw_tokens = copy.copy(raw_data["words"])
        origin_tokens = []
        for token in raw_tokens:
            token = token.lower()
            origin_tokens.append(self.tokenizer.tokenize(token))
        length = len(origin_tokens)
        wordlabel = np.zeros((length))
        sp_tokens = copy.copy(origin_tokens)
        sp_tokens,tokenmap = self.convert_list_to_token(sp_tokens)
        sp_tokens = self.tokenizer.encode_plus(sp_tokens)
        mask = sp_tokens["attention_mask"]
        tokens = sp_tokens["input_ids"] 

        positive = []
        negative = []
        none = []
        triggerset = []
        triggerlabel = []
        label = []
        if not query:
            for event in copy.copy(raw_data["event"]):
                start = event[1]['start']
                end = event[1]['end']
                if event[0] == eventtype:
                    wordlabel[start:end] = 1
                    positive.append([tokenmap[i] for i in range(start,end+1)])
                else:
                    negative.append([tokenmap[start],tokenmap[end],1,0])
            posindex = random.sample(positive,1)[0]
            index = [[posindex[0],posindex[-1]]]
            label.append(1)
            for i in range(length):
                if wordlabel[i] == 0:
                    index.append([tokenmap[i],tokenmap[i+1]])
                    label.append(0)
        else:
            index = []
            for event in copy.copy(raw_data["event"]):
                start = event[1]['start']
                end = event[1]['end']
                if event[0] == eventtype:
                    wordlabel[start:end] = 2
                    triggerset.append([1,start,end-1])
                else:
                    wordlabel[start:end] = 1
            for i in range(length):
                index.append([tokenmap[i],tokenmap[i+1]])
                if wordlabel[i] == 0:
                    label.append(0)
                    triggerlabel.append(0)
                elif wordlabel[i] == 1: 
                    label.append(0)
                    triggerlabel.append(1)
                elif wordlabel[i] == 2: 
                    label.append(1)
                    triggerlabel.append(1)
        return tokens,mask,index,label,triggerlabel,triggerset

    def get_testingdata(self,dataes,class_name,query):
        datalist = []
        for data in dataes:
            token,mask,index,label,triggerlabel,triggerset = self.gettokenizer(copy.copy(self.sentence[data]),class_name, query = query)
            datalist.append([token,mask,index,label,triggerlabel,triggerset])
        return datalist

    def bert_data(self,class_name):
        support_set = {'data': [], 'word': [], 'mask' : [], 'label': [],'triggerlabel':[],'tokentype':[],'index':[]}
        query_set = {'data': [], 'wordindex':[], 'word': [], 'mask': [], 'label': [],'triggerlabel':[],'triggerset':[],'tokentype':[],'index':[]}

        positive_list = copy.copy(self.json_data[class_name])
        if class_name in self.query.keys():
            negative_list = copy.copy(self.query[class_name])
        else:
            negative_list = []
        negative_list_add = [i for i in self.sentencelist if i not in positive_list + negative_list]

        shuffle(positive_list)
        support_list = positive_list[:self.K]

        query_list = positive_list[self.K:self.K+self.posnum] 
        query_negative_list = random.sample(negative_list, min(math.ceil(self.posnum*self.negativerate/2),len(negative_list)))
        query_negative_list += random.sample(negative_list_add, min(self.posnum*self.negativerate-len(query_negative_list),len(negative_list_add)))
        query_list += query_negative_list
        supportlist = self.get_testingdata(support_list,class_name, query = False)
        querylist = self.get_testingdata(query_list,class_name, query = True)

        for support in supportlist:
            token,mask,index,label,_,_ = support
            support_set['word'].append(ArrayField(np.array(token)))
            support_set['mask'].append(ArrayField(np.array(mask)))
            support_set['index'].append(ArrayField(np.array(index)))
            support_set['label'].append(ListField([LabelField(i,skip_indexing=True) for i in label]))
        
        for query in querylist:
            token,mask,index,label,triggerlabel,triggerset = query
            query_set['word'].append(ArrayField(np.array(token)))
            query_set['mask'].append(ArrayField(np.array(mask)))
            query_set['index'].append(ArrayField(np.array(index)))
            query_set['triggerlabel'].append(ListField([LabelField(i,skip_indexing=True) for i in triggerlabel]))
            query_set['label'].append(ListField([LabelField(i,skip_indexing=True) for i in label]))
            query_set['triggerset'].append(MetadataField(triggerset))

        batch_data = {
            'support': support_set,
            'query': query_set,
            'supportlist':support_list,
            'querylist':query_list,
        }
        return batch_data

    def text_to_instance(self, data) -> Instance:  # type: ignore
        fields = {
            "support_tokenid": ListField(data['support']['word']),
            "support_mask": ListField(data['support']['mask']),
            "support_index": ListField(data['support']['index']),
            "support_label": ListField(data['support']['label']),
            "query_tokenid": ListField(data['query']['word']),
            "query_mask": ListField(data['query']['mask']),
            "query_label": ListField(data['query']['label']),
            "query_index": ListField(data['query']['index']),
            "query_triggerset": ListField(data['query']['triggerset']),
        }
        return Instance(fields)
    
    def initdata(self,filepath):
        # filepath = eval(filepath)
        if type(filepath) is dict:
            self.json_data = filepath
        else:
            if not os.path.exists(filepath):
                print("[ERROR] Data file does not exist!")
                assert(0)
            self.json_data = json.load(open(filepath))
        self.sentencelist = []
        for i in self.json_data.keys():
            for j in self.json_data[i]:
                if j not in self.sentencelist:
                    self.sentencelist.append(j)
        self.classes = list(self.json_data.keys())

    def _read(self, filepath):
        if self.json_data is None:
            self.initdata(filepath)
        for classname in self.classes:
            for i in range(self.instancenum): 
                yield self.text_to_instance(self.bert_data(classname))
        