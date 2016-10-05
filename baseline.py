import numpy
import prepare_data
import sys

import numpy
import prepare_data
import sys

args = prepare_data.arg_passing(sys.argv)
numpy.random.seed(args['-seed'])

from keras.optimizers import *
from keras.objectives import *
from create_model import *

################################# LOAD DATA #######################################################
dataset = '../data/' + args['-data'] + '.pkl.gz'
nnet_model = args['-nnetM']
saving = args['-saving']
hid_dim = args['-dim']
vocab_size = args['-vocab']

if 'hid' in args['-reg']: dropout_hid = True
else: dropout_hid = False

train_t, train_d, train_y, valid_t, valid_d, valid_y, test_t, test_d, test_y = prepare_data.load(dataset)
train_x = prepare_data.prepare_BoW(train_t, train_d, vocab_size)
valid_x = prepare_data.prepare_BoW(valid_t, valid_d, vocab_size)
test_x = prepare_data.prepare_BoW(test_t, test_d, vocab_size)

print ('ntrain: %d, n_valid: %d, n_test: %d' % (len(train_y), len(valid_y), len(test_y)))

print train_y.dtype
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

# n_classes, vocab_size, inp_len, emb_dim,
# seq_model='lstm', nnet_model='highway', pool_mode='mean',
# dropout_inp=False, dropout_hid=True

model = create_BoW(n_classes=n_classes, vocab_size=vocab_size, hid_dim=hid_dim,
                     nnet_model=nnet_model,
                     dropout=dropout_hid)

model.summary()
opt = RMSprop(lr=0.01)
model.compile(optimizer=opt, loss=loss)

train_y = numpy.expand_dims(train_y, -1)

fParams = 'bestModels/' + saving + '.hdf5'
fResult = 'log/' + saving + '.txt'

if n_classes == -1: type = 'linear'
elif n_classes == 1: type = 'binary'
else: type = 'multi'

saveResult = SaveResult([valid_x, valid_y, test_x, test_y],
                        metric_type=type, fileResult=fResult, fileParams=fParams)

callbacks = [saveResult, NanStopping()]
his = model.fit(train_x, train_y,
                validation_data=(valid_x, numpy.expand_dims(valid_y, -1)),
                nb_epoch=1000, batch_size=100, callbacks=callbacks)