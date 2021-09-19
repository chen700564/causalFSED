import sys
sys.path.append("..")
import argparse
import torch
import numpy as np
import random
from train import finetuning,trainadv,baseline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--k", default = 5,type=int, help='the number of instances in support set, default is 5')
    parser.add_argument("-t", "--test", action="store_true")
    parser.add_argument("-d", "--dev", action="store_true")
    parser.add_argument('--tag', default="", type=str, help='tag of model file')
    parser.add_argument("--model", default = 'causal',type=int, help='base:FS-Base,causal:FS-Causal, default is causal')
    parser.add_argument("--metric", default = "proto",type=str,help='metric: proto or relation, default is proto')
    parser.add_argument("--dataset", default = "ace",type=str,help='metric:ace, maven, kbp or custom, default is ace')
    parser.add_argument("--cuda", default = 0,type=int, help='gpu id for training')

    args = parser.parse_args()
    
    k = args.k
    test = args.test
    dev = args.dev
    model = args.model
    tag = args.tag
    metric = args.metric
    dataset = args.dataset
    cuda= args.cuda
    
    
    SEED = int(np.random.uniform(0,1)*100000)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic=True
    print('seed: ', SEED)

    if test:
        # fix seed in test
        random.seed(5422)

    if dataset == 'maven':
        from config.maven.baselineconfig import config
    elif dataset == 'ace':
        from config.ace.baselineconfig import config
    elif dataset == 'kbp':
        from config.kbp.baselineconfig import config
    elif dataset == 'custom':
        from config.custom.baselineconfig import config
    baseline(config, test=test, k=k, tag=tag, model=model, metric=metric,  dataset=dataset, dev=dev, cuda=cuda)
