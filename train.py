import torch
import numpy as np
from transformers import AdamW
import math
from allennlp.training.learning_rate_schedulers.polynomial_decay import PolynomialDecay
from allennlp.data.dataloader import PyTorchDataLoader
from allennlp.training.trainer import GradientDescentTrainer
import warnings
import sys
warnings.filterwarnings("ignore")


def baseline(config, test=False, k=5, tag="", model='causal', metric='proto',dataset='ace',dev=False, cuda=0):
    print("baseline bert")

    modelmetric = metric

    devreader = config.devreader(k, posnum = config.posnum, negativerate = config.negativerate, sentence=config.sentence,instancenum=config.devinstancenum,query=config.valquery)
    testreader = config.testreader(k, posnum = config.posnum, negativerate = config.negativerate,sentence=config.sentence,instancenum=config.testinstancenum,query=config.testquery)


    backdooruse = config.backdooruse[metric]
    
    if model == 'causal':
        print("model:causal")
        model = config.model(None, config.PRE_FILE,modelmetric)
        if not test:
            trainreader = config.trainreader(k, config.Q, noise_length=config.noiselength, nanoiselength = config.nanoiselength, maxlength = config.maxlength,sentence=config.sentence,instancenum = config.instancenum, backdooruse=backdooruse,device=cuda, lazy=True)
    elif model == 'base':
        print("model:baseline")
        if not test:
            model = config.model2(None, config.PRE_FILE,modelmetric)
            trainreader = config.trainreader2(k, config.Q, noise_length=config.noiselength, nanoiselength = config.nanoiselength, maxlength = config.maxlength,sentence=config.sentence,instancenum = config.instancenum, lazy=True)
        else:
            model = config.model(None, config.PRE_FILE,modelmetric)
    else:
        print("Sorry, the model you choose do not exist.")
        raise RuntimeError("modelError")

    parameters_to_optimize = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    parameters_to_optimize = [
            {'params': [p for n, p in parameters_to_optimize
                        if not any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.01},
            {'params': [p for n, p in parameters_to_optimize
                        if any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0},
        ]
    model = model.cuda(cuda)
    filepath = 'tmp/' + str(dataset) + '/' + model+'_'+str(k) +'_'+tag 
    if dev:
        testreader.initdata(config.devfile)
        model.load_state_dict(torch.load(filepath+'/best.th'))
        result = evalue_causal(model,testreader)
        with open(filepath+'/devresult.txt','w') as f:
            f.write(str(result)+"\n")
        return None
    if test:
        testreader.initdata(config.testfile)
        model.load_state_dict(torch.load(filepath+'/best.th'))
        result = evalue_causal(model,testreader)
        with open(filepath+'/result.txt','w') as f:
            f.write(str(result)+"\n")
        return None
    trainreader.initdata(config.trainfile)
    Trainset = trainreader.read(config.trainfile)
    devreader.initdata(config.devfile)
    Devset = devreader.read(config.devfile)

    optimizer = AdamW(parameters_to_optimize, lr=config.LR, correct_bias=False)

    data_loader = PyTorchDataLoader(Trainset,config.BATCH,batches_per_epoch=config.epochnum)
    valdata_loader = PyTorchDataLoader(Devset,config.BATCH,batches_per_epoch=config.devepochnum)
    learning_rate_scheduler = PolynomialDecay(optimizer,80,config.epochnum,1,config.epochnum,config.LR)
    trainer = GradientDescentTrainer(
                      model=model,
                      optimizer=optimizer,
                      data_loader=data_loader,
                      validation_data_loader=valdata_loader,
                      learning_rate_scheduler=learning_rate_scheduler,
                      patience=15,
                      num_epochs=80,
                      validation_metric='+microf1',
                      cuda_device=cuda_device,
                      serialization_dir=filepath,
                      use_amp=False)
    trainer.train()

def evalue_causal(model, testreader):
    model.eval()
    model.microf1.reset()
    batch_size = 1
    print("begin test!")
    result = {}
    resultp = []
    resultr = []
    resultf1 = []
    with torch.no_grad():
        for classname in testreader.classes:
            model.f1.reset()
            print("begin to test "+classname)
            testreader.setclass(classname)
            testset = testreader.read(None)
            for i in range(math.ceil(len(testset)/batch_size)):
                model.forward_on_instances(testset[i*batch_size:(i+1)*batch_size])
                d = model.get_metrics()
                sys.stdout.write('test step: {0}, p:{1:.4f}, r:{2:.4f}, f1: {3:.4f}, microF1: {4:.4f}'.format(
                    i,d['p'],d['r'], d['f1'], model.microf1.get_metric()['f1']) + '\r')
            sys.stdout.write('\n')
            print('\n')
            print(model.get_metrics()['f1'])
            result[classname] = model.get_metrics()
            resultp.append(model.get_metrics()['p'])
            resultr.append(model.get_metrics()['r'])
            resultf1.append(model.get_metrics()['f1'])
    p = np.mean(np.array(resultp))
    r = np.mean(np.array(resultr))
    f1 = np.mean(np.array(resultf1))
    result['macro'] = {
        'p':p,
        'r':r,
        'f1':f1
    }
    result['micro'] = model.microf1.get_metric()
    print("macro f1")
    print(result['macro'])
    print("micro f1")
    print(result['micro'])
    return result
