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


def baseline(config, test=False, k=5, tag="", model='FSCausal', metric='proto',dataset='ace',dev=False, cuda=0):
    print("baseline bert")
    modelmetric = metric

    backdooruse = config.backdooruse[metric]

    filepath = 'tmp/' + str(dataset) + '/' + model + '_' + metric +'_'+str(k) +'_'+tag 
    if model == 'FSCausal':
        print("model:FS-Causal")
        if (not test) and (not dev):
            trainreader = config.trainreader(k, config.Q, noise_length=config.noiselength, maxlength = config.maxlength,sentence=config.sentence,instancenum = config.instancenum, backdooruse=backdooruse,device=cuda, lazy=True)
            model = config.model(None, config.PRE_FILE,modelmetric)
        else:
            model = config.model2(None, config.PRE_FILE,modelmetric)
    elif model == 'FSBase':
        print("model:FS-base")
        if (not test) and (not dev):
            model = config.model2(None, config.PRE_FILE,modelmetric)
            trainreader = config.trainreader2(k, config.Q, noise_length=config.noiselength, maxlength = config.maxlength,sentence=config.sentence,instancenum = config.instancenum, lazy=True)
        else:
            model = config.model2(None, config.PRE_FILE,modelmetric)
    else:
        print("Sorry, the model you choose do not exist.")
        raise RuntimeError("modelError")

    model = model.cuda(cuda)
    if dev:
        testreader = config.testreader(k,sentence=config.sentence,instancenum=config.testinstancenum)
        testreader.initdata(config.devfile)
        model.load_state_dict(torch.load(filepath+'/best.th',map_location='cuda:'+str(cuda)))
        result = evalue_causal(model,testreader)
        with open(filepath+'/devresult.txt','w') as f:
            f.write(str(result)+"\n")
        return None
    if test:
        testreader = config.testreader(k,sentence=config.sentence,instancenum=config.testinstancenum)
        testreader.initdata(config.testfile)
        model.load_state_dict(torch.load(filepath+'/best.th',map_location='cuda:'+str(cuda)))
        result = evalue_causal(model,testreader)
        with open(filepath+'/result.txt','w') as f:
            f.write(str(result)+"\n")
        return None
    devreader = config.devreader(k, posnum = config.posnum, negativerate = config.negativerate, sentence=config.sentence,instancenum=config.devinstancenum,query=config.valquery)
    trainreader.initdata(config.trainfile)
    Trainset = trainreader.read(config.trainfile)
    devreader.initdata(config.devfile)
    Devset = devreader.read(config.devfile)

    parameters_to_optimize = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    parameters_to_optimize = [
            {'params': [p for n, p in parameters_to_optimize
                        if not any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.01},
            {'params': [p for n, p in parameters_to_optimize
                        if any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0},
        ]
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
                      cuda_device=cuda,
                      serialization_dir=filepath,
                      use_amp=False)
    trainer.train()

def evalue_causal(model, testreader):
    model.eval()
    model.f1.reset()
    batch_size = 1
    print("begin test!")
    result = {}
    resultp = []
    resultr = []
    resultf1 = []
    with torch.no_grad():
        for classname in testreader.classes:
            model.typef1.reset()
            print("begin to test "+classname)
            testreader.setclass(classname)
            testset = testreader.read(None)
            for i in range(math.ceil(len(testset)/batch_size)):
                model.forward_on_instances(testset[i*batch_size:(i+1)*batch_size])
                d = model.typef1.get_metric()
                sys.stdout.write('test step: {0}, typef1: {1:.4f}, microF1: {2:.4f}'.format(i, d['f1'], model.get_metrics()['microf1']) + '\r')
            sys.stdout.write('\n')
            print('\n')
            result[classname] = model.typef1.get_metric()
            print(result[classname])
            resultp.append(model.typef1.get_metric()['precision'])
            resultr.append(model.typef1.get_metric()['recall'])
            resultf1.append(model.typef1.get_metric()['f1'])
    p = np.mean(np.array(resultp))
    r = np.mean(np.array(resultr))
    f1 = np.mean(np.array(resultf1))
    result['macro'] = {
        'p':p,
        'r':r,
        'f1':f1
    }
    result['micro'] = model.get_metrics()
    print("macro f1")
    print(result['macro'])
    print("micro f1")
    print(result['micro'])
    return result
