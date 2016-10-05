import os
import sys

# Tuning parameters:
dims = ['64'] #, '15', '20']
tasks = ['imdb'] # []??
nnet_models = ['highway'] # ['dense', 'highway']
seq_models = ['lstm'] # ['gru', 'lstm', 'rnn']
regs = ['inphid'] # ['x', 'inp', 'hid', 'inphid'] # 'x' means no dropout
pretrains = ['x'] # ['x', 'finetune', 'fixed']
            # 'x' means no pretraining,
            # 'finetune' means embedding matrix is initialized by pretrained parameters
            # 'fixed' means using pretrained embedding matrix as input features
pools = ['mean'] # ['mean', 'max', 'last']

if sys.argv[1] == 'gpu':
    flag = 'THEANO_FLAGS=\'floatX=float32,device=gpu\' '
else:
    flag = ''

# Running script:
for s in tasks:
    for nnet in nnet_models:
        for seq in seq_models:
            for dim in dims:
                for reg in regs:
                    for pretrain in pretrains:
                        for pool in pools:
                            cmd = 'python training.py -data ' + s + ' -nnetM ' + nnet + ' -seqM ' + seq + ' -dim ' + dim + \
                                  ' -reg ' + reg + ' -pretrain ' + pretrain + ' -pool ' + pool
                            cmd += ' -saving ' + s + '_' + seq + '_' + nnet + '_dim' + dim + \
                                   '_reg' + reg + '_pre' + pretrain + '_pool' + pool

                        print cmd
                        os.system(cmd)