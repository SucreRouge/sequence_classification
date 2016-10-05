from keras.models import Model
from keras.layers import *
from sklearn import metrics
from keras.callbacks import *
import numpy

class SaveResult(Callback):
    '''
    Compute result after each epoch. Return a log of result
    Arguments:
        data_x, data_y, metrics
    '''

    def __init__(self, data=None, metric_type='binary', fileResult='', fileParams='', minPatience=5, maxPatience=30):
        # metric type can be: binary, multi
        super(SaveResult, self).__init__()

        self.valid_x = None
        self.valid_y = None
        self.test_x = None
        self.test_y = None
        self.do_test = False

        f = open(fileResult, 'w')

        if 'binary' in metric_type:
            fm = ['auc', 'f1', 'pre', 'rec']
        elif 'multi' in metric_type:
            fm = ['ma_f1', 'mi_f1', 'pre', 'rec']
        else:
            fm = ['mae', 'pr25', 'mae', 'pr25']

        if len(data) >= 2:
            self.valid_x, self.valid_y = data[0], data[1]
            f.write('epoch\tloss\tv_loss\t|\tv_' + fm[0] + '\tv_' + fm[1] + '\tv_' + fm[2] + '\tv_' + fm[3] + '\t|')
        if len(data) == 4:
            self.test_x, self.test_y = data[2], data[3]
            f.write('\tt_' + fm[0] + '\tt_' + fm[1] + '\tt_' + fm[2] + '\tt_' + fm[3])
            self.do_test = True
        f.write('\n')
        f.close()

        self.bestResult = 1000.0 if 'linear' in metric_type else 0.0
        self.bestEpoch = 0
        # wait to divide the learning rate. if reach the maxPatience -> stop learning
        self.wait = 0
        self.patience = minPatience
        self.maxPatience = maxPatience

        self.metric_type = metric_type
        self.fileResult = fileResult
        self.fileParams = fileParams

    def _compute_result(self, x, y_true):
        def pr25(y_true, y_pred):
            y_true = numpy.expand_dims(y_true, -1)
            mre = abs(y_true - y_pred) / y_true
            m = (mre <= 0.25).astype('float32')
            return 100.0 * numpy.sum(m) / len(y_true)

        y_pred = self.model.predict(x, batch_size=x[0].shape[0])
        if numpy.isnan(y_pred).any(): return 0.0, 0.0, 0.0, 0.0

        if 'binary' in self.metric_type:
            fp, tp, thresholds = metrics.roc_curve(y_true, y_pred)
            auc = metrics.auc(fp, tp)
            y_pred = numpy.round(y_pred)
            f1 = metrics.f1_score(y_true, y_pred)
            pre = metrics.precision_score(y_true, y_pred)
            rec = metrics.recall_score(y_true, y_pred)
        elif 'multi' in self.metric_type:
            y_pred = numpy.argmax(y_pred, axis=1)
            auc = metrics.f1_score(y_true, y_pred, average='macro')
            f1 = metrics.f1_score(y_true, y_pred, average='micro')
            pre = metrics.precision_score(y_true, y_pred, average='micro')
            rec = metrics.recall_score(y_true, y_pred, average='micro')
        else:
            auc = metrics.mean_absolute_error(y_true, y_pred)
            f1 = pr25(y_true, y_pred)
            pre, rec = auc, f1

        return auc, f1, pre, rec

    def better(self, a, b, sign):
        if sign == -1:
            return a < b
        else: return a > b

    def on_epoch_end(self, epoch, logs={}):
        v_auc, v_f1, v_pre, v_rec = self._compute_result(self.valid_x, self.valid_y)

        f = open(self.fileResult, 'a')
        f.write('%d\t%.4f\t%.4f\t|' % (epoch, logs['loss'], logs['val_loss']))
        f.write('\t%.4f\t%.4f\t%.4f\t%.4f\t|' % (v_auc, v_f1, v_pre, v_rec))
        if self.do_test:
            t_auc, t_f1, t_pre, t_rec = self._compute_result(self.test_x, self.test_y)
            f.write('\t%.4f\t%.4f\t%.4f\t%.4f\t|' % (t_auc, t_f1, t_pre, t_rec))

        if 'linear' in self.metric_type:
            compare_val = v_pre
            sign = -1
            #print (compare_val, self.bestResult, self.better(compare_val, self.bestResult, -1))
        else:
            compare_val = v_f1
            sign = 1

        if self.better(compare_val, self.bestResult, sign):
            self.bestResult = compare_val
            self.bestEpoch = epoch
            self.model.save_weights(self.fileParams, overwrite=True)
            self.wait = 0
        f.write('  Best result at epoch %d\n' % self.bestEpoch)
        f.close()

        if not self.better(compare_val, self.bestResult, sign):
            self.wait += 1
            if self.wait == self.patience:
                self.wait = 0
                self.patience += 5

                lr = self.model.optimizer.lr / 2.0
                self.model.optimizer.lr = lr
                if self.patience > self.maxPatience:
                    self.model.stop_training = True

class NanStopping(Callback):
    def __init__(self):
        super(NanStopping, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        for k in logs.values():
            if numpy.isnan(k):
                self.model.stop_training = True


class PoolingSeq(Layer):
    # pooling a sequence of vector to a vector
    # mode can be: mean, max or last (the last vector of the sequence)
    def __init__(self, mode='mean', **kwargs):
        self.mode = mode

        super(PoolingSeq, self).__init__(**kwargs)

    def compute_mask(self, input, mask=None):
        return None

    def get_output_shape_for(self, input_shape):
        return (None, input_shape[0][-1])

    def call(self, inputs, mask=None):
        seqs = inputs[0]
        mask = inputs[1]

        if self.mode == 'mean':
            seqs = seqs * mask[:, :, None]
            pooled_state = K.sum(seqs, axis=1) / K.sum(mask, axis=1)[:, None]
        elif self.mode == 'max':
            seqs = seqs * mask[:, :, None]
            pooled_state = K.max(seqs, axis=1)
        else: # self.mode = last
            pooled_state = seqs[:, -1]

        return pooled_state

def create_highway(below_layer, out_dim):
    shared_highway = Highway(activation='relu', init='glorot_normal', transform_bias=-1)
    hidd = below_layer
    for i in range(10):
        hidd = shared_highway(hidd)

    return hidd

def create_dense(below_layer, out_dim):
    hidd = Dense(output_dim=out_dim, activation='relu', init='glorot_normal')(below_layer)

    return hidd

def create_model(n_classes, vocab_size, inp_len, emb_dim,
                 seq_model='lstm', nnet_model='highway', pool_mode='mean',
                 dropout_inp=False, dropout_hid=True, emb_weight=None):
    if emb_weight is not None:
        emb_weight = [emb_weight[:vocab_size]]
    seq_dict = {'lstm': LSTM, 'gru': GRU, 'rnn': SimpleRNN}
    nnet_dict = {'highway': create_highway, 'dense': create_dense}
    if n_classes == -1:
        top_act = 'linear'
    elif n_classes == 1:
        top_act = 'sigmoid'
    else: top_act = 'softmax'

    inp = Input(shape=(inp_len,), dtype='int64', name='title_inp')
    mask = Input(shape=(inp_len,), dtype='float32', name='title_mask')

    if dropout_inp:
        drop_rate = 0.2
    else:
        drop_rate = 0.0

    embedding = Embedding(output_dim=emb_dim, input_dim=vocab_size, input_length=inp_len,
                          mask_zero=True, weights=emb_weight,
                          dropout=drop_rate)(inp)
    seq_hid = seq_dict[seq_model](input_dim=emb_dim, output_dim=emb_dim,
                     return_sequences=True, dropout_U=drop_rate, dropout_W=drop_rate)(embedding)

    hidd = PoolingSeq(mode=pool_mode)([seq_hid, mask])

    if dropout_hid:
        hidd = Dropout(0.5)(hidd)

    hidd = nnet_dict[nnet_model](hidd, emb_dim)
    hidd = Dropout(0.5)(hidd)
    top_hidd = Dense(output_dim=abs(n_classes), activation=top_act)(hidd)

    model = Model(input=[inp, mask], output=top_hidd)

    return model

def create_fixed(n_classes, inp_len, emb_dim,
                 seq_model='lstm', nnet_model='highway', pool_mode='mean',
                 dropout_inp=False, dropout_hid=True):
    seq_dict = {'lstm': LSTM, 'gru': GRU, 'rnn': SimpleRNN}
    nnet_dict = {'highway': create_highway, 'dense': create_dense}
    if n_classes == -1:
        top_act = 'linear'
    elif n_classes == 1:
        top_act = 'sigmoid'
    else: top_act = 'softmax'

    inp = Input(shape=(inp_len, emb_dim), dtype='float32', name='title_inp')
    mask = Input(shape=(inp_len,), dtype='float32', name='title_mask')

    if dropout_inp:
        drop_rate = 0.2
    else:
        drop_rate = 0.0

    seq_hid = seq_dict[seq_model](input_dim=emb_dim, output_dim=emb_dim,
                     return_sequences=True, dropout_U=drop_rate, dropout_W=drop_rate)(inp)

    hidd = PoolingSeq(mode=pool_mode)([seq_hid, mask])

    if dropout_hid:
        hidd = Dropout(0.5)(hidd)

    hidd = nnet_dict[nnet_model](hidd, emb_dim)
    hidd = Dropout(0.5)(hidd)
    top_hidd = Dense(output_dim=abs(n_classes), activation=top_act)(hidd)

    model = Model(input=[inp, mask], output=top_hidd)

    return model


def create_BoW(n_classes, vocab_size, hid_dim,
               nnet_model='highway', dropout=True):
    nnet_dict = {'highway': create_highway, 'dense': create_dense}
    if n_classes == -1:
        top_act = 'linear'
    elif n_classes == 1:
        top_act = 'sigmoid'
    else: top_act = 'softmax'

    input = Input(shape=(vocab_size,), dtype='float32', name='input')
    hidd = Dense(output_dim=hid_dim, activation='relu', init='glorot_normal')(input)
    hidd = Dropout(0.5)(hidd)
    hidd = nnet_dict[nnet_model](hidd, hid_dim)
    hidd = Dropout(0.5)(hidd)
    top_hidd = Dense(output_dim=abs(n_classes), activation=top_act)(hidd)

    model = Model(input=input, output=top_hidd)

    return model
