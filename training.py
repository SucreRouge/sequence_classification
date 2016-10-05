import numpy
import prepare_data
import sys

args = prepare_data.arg_passing(sys.argv)
numpy.random.seed(args['-seed'])

from keras.optimizers import *
from keras.objectives import *
from create_model import *

################################# LOAD DATA #######################################################
dataset = 'data/' + args['-data'] + '.pkl.gz'
seq_model = args['-seqM']
nnet_model = args['-nnetM']
saving = args['-saving']
hid_dim = args['-dim']
vocab_size = args['-vocab']
pool = args['-pool']
pretrain = args['-pretrain']

if 'lm' in pretrain:
    lm = 'lm'
else: lm = ''

pretrain_path = 'NCE' + lm + '_' + args['-data'] + '_dim' + str(hid_dim)
if pretrain == 'x':
    emb_weight = None
else: emb_weight = prepare_data.load_weight(pretrain_path)

if 'inp' in args['-reg']: dropout_inp = True
else: dropout_inp = False
if 'hid' in args['-reg']: dropout_hid = True
else: dropout_hid = False

train_x, train_y, valid_x, valid_y, test_x, test_y = prepare_data.load(dataset)
train_x, train_mask = prepare_data.prepare_data(train_x, vocab_size)
valid_x, valid_mask = prepare_data.prepare_data(valid_x, vocab_size)
test_x, test_mask = prepare_data.prepare_data(test_x, vocab_size)

print ('ntrain: %d, n_valid: %d, n_test: %d' % (len(train_y), len(valid_y), len(test_y)))

if train_y.dtype == 'float32':
    n_classes = -1
    loss = mean_squared_error
elif max(train_y) > 1:
    n_classes = max(train_y) + 1
    loss = sparse_categorical_crossentropy
else:
    n_classes = 1
    loss = binary_crossentropy

###################################### BUILD MODEL##################################################
print 'Building model...'

if 'fixed' in pretrain:
    train_x, valid_x, test_x = prepare_data.to_features([train_x, valid_x, test_x], emb_weight)
    # n_classes, inp_len, emb_dim,
    # seq_model='lstm', nnet_model='highway', pool_mode='mean',
    # dropout_inp=False, dropout_hid=True):

    model = create_fixed(n_classes=n_classes, inp_len=train_x.shape[1], emb_dim=hid_dim,
                         seq_model=seq_model, nnet_model=nnet_model, pool_mode=pool,
                         dropout_inp=dropout_inp, dropout_hid=dropout_hid)
else:
    # n_classes, vocab_size, inp_len, emb_dim,
    # seq_model='lstm', nnet_model='highway', pool_mode='mean',
    # dropout_inp=False, dropout_hid=True
    model = create_model(n_classes=n_classes, vocab_size=vocab_size + 1, inp_len=train_x.shape[-1], emb_dim=hid_dim,
                         seq_model=seq_model, nnet_model=nnet_model, pool_mode=pool,
                         dropout_inp=dropout_inp, dropout_hid=dropout_hid, emb_weight=emb_weight)

model.summary()
json_string = model.to_json()
fModel = open('models/' + saving + '.json', 'w')
fModel.write(json_string)

opt = RMSprop(lr=0.01)
model.compile(optimizer=opt, loss=loss)

train_y = numpy.expand_dims(train_y, -1)

fParams = 'bestModels/' + saving + '.hdf5'
fResult = 'log/' + saving + '.txt'

if n_classes == -1: type = 'linear'
elif n_classes == 1: type = 'binary'
else: type = 'multi'

saveResult = SaveResult([[valid_x, valid_mask], valid_y,
                         [test_x, test_mask], test_y],
                        metric_type=type, fileResult=fResult, fileParams=fParams)

callbacks = [saveResult, NanStopping()]
his = model.fit([train_x, train_mask], train_y,
                validation_data=([valid_x, valid_mask], numpy.expand_dims(valid_y, -1)),
                nb_epoch=1000, batch_size=100, callbacks=callbacks)