from __future__ import division
from optparse import OptionParser
  
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from gat_layer import GraphAttention
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow import keras
import matplotlib.pyplot as plt
import time
from scipy.stats import spearmanr
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from transformer_posenc import Encoder_Posenc
from transformer import Encoder

def main():
    usage = 'usage: %prog [options]'
    parser = OptionParser(usage)
    parser.add_option('-c', dest='cell_line',
        default='K562', type='str')
    parser.add_option('-o', dest='organism',
        default='human', type='str')
    parser.add_option('-v', dest='valid_chr',
        default='1,11', type='str')
    parser.add_option('-t', dest='test_chr',
        default='2,12', type='str')
    parser.add_option('-p', dest='data_path',
        default='/Users/ashwini/Desktop/SuperUROP', type='str')
    parser.add_option('-a', dest='assay_type',
        default='HiChIP', type='str')
    parser.add_option('-q', dest='qval',
        default=0.1, type='float')
    parser.add_option('-n', dest='n_gat_layers',
        default=2, type='int')
    parser.add_option('-g', dest='generalizable',
        default=0, type='int')
    parser.add_option('--att_heads', dest='att_heads',
	default=4, type='int')
    parser.add_option('--att_layers', dest='att_layers',
	default=1, type='int')
    parser.add_option('--lr', dest='lr',
	default=0.0002, type='float') 
    parser.add_option('--epochs', dest='epochs',
	default=30, type='int')
    parser.add_option('--posenc', dest='posenc',
	default=0, type='int')
    parser.add_option('--chr_tfr_path', dest='chr_tfr_path',
        default='/data/tfrecords', type='str')

    (options, args) = parser.parse_args()
    valid_chr_str = options.valid_chr.split(',')
    valid_chr = [int(valid_chr_str[i]) for i in range(len(valid_chr_str))]
    test_chr_str = options.test_chr.split(',')
    test_chr = [int(test_chr_str[i]) for i in range(len(test_chr_str))]

    data_path = options.data_path
    chr_tfr_path = options.chr_tfr_path
    assay_type = options.assay_type
    qval = options.qval

    if qval == 0.1:
        fdr = '1'
    elif qval == 0.01:
        fdr = '01'
    elif qval == 0.001:
        fdr = '001'

    print('organism:', options.organism)
    print('cell type:', options.cell_line)
    print('valid chrs: ', valid_chr)
    print('test chrs: ', test_chr)
    print('data path: ', options.data_path)
    print('3D assay type: ', options.assay_type)
    print('HiCDCPlus FDR: ', options.qval)
    print('number of GAT layers: ', options.n_gat_layers)
    print('generalizables: ', options.generalizable)
    print('number of attention heads: ', options.att_heads)
    print('number of attention layers: ', options.att_layers)
    print('learning rate: ', options.lr)
    print('epochs: ', options.epochs)
    print('posenc: ', options.posenc)
    print('chr_tfr_path: ', options.chr_tfr_path)

    def poisson_loss(y_true, mu_pred):
        nll = tf.reduce_mean(tf.math.lgamma(y_true + 1) + mu_pred - y_true * tf.math.log(mu_pred))
        return nll

    def parse_proto(example_protos):
        features = {
            'last_batch': tf.io.FixedLenFeature([1], tf.int64),
            'adj': tf.io.FixedLenFeature([], tf.string),
            #'adj_real': tf.io.FixedLenFeature([], tf.string),
            'tss_idx': tf.io.FixedLenFeature([], tf.string),
            'X_1d': tf.io.FixedLenFeature([], tf.string),
            'Y': tf.io.FixedLenFeature([], tf.string)
        }
        parsed_features = tf.io.parse_example(example_protos, features=features)
        last_batch = parsed_features['last_batch']

        adj = tf.io.decode_raw(parsed_features['adj'], tf.float16)
        adj = tf.cast(adj, tf.float32)

        tss_idx = tf.io.decode_raw(parsed_features['tss_idx'], tf.float16)
        tss_idx = tf.cast(tss_idx, tf.float32)

        X_epi = tf.io.decode_raw(parsed_features['X_1d'], tf.float16)
        X_epi = tf.cast(X_epi, tf.float32)

        Y = tf.io.decode_raw(parsed_features['Y'], tf.float16)
        Y = tf.cast(Y, tf.float32)

        return {'last_batch': last_batch, 'X_epi': X_epi, 'Y': Y, 'adj': adj, 'tss_idx': tss_idx}

    def file_to_records(filename):
            return tf.data.TFRecordDataset(filename, compression_type='ZLIB')

    def dataset_iterator(file_name, batch_size):
        dataset = tf.data.TFRecordDataset(file_name, compression_type="ZLIB")
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(parse_proto)
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        return iterator

    def read_tf_record_1shot(iterator):
        try:
            next_datum = iterator.get_next()
            data_exist = True
        except tf.errors.OutOfRangeError:
            data_exist = False
        if data_exist:
            T = 400       # number of 5kb bins inside middle 2Mb region 
            b = 50        # number of 100bp bins inside 5Kb region
            F = 3         # number of Epigenomic tracks used in model
            X_epi = next_datum['X_epi']
            batch_size = tf.shape(X_epi)[0]
            X_epi = tf.reshape(X_epi, [batch_size, 3*T*b, F])
            adj = next_datum['adj']
            adj = tf.reshape(adj, [batch_size, 3*T, 3*T])

            #last_batch = next_datum['last_batch']
            tss_idx = next_datum['tss_idx']
            tss_idx = tf.reshape(tss_idx, [3*T])
            idx = tf.range(T, 2*T)

            Y = next_datum['Y']
            Y = tf.reshape(Y, [batch_size, 3*T])

        else:
            X_epi = 0
            Y = 0
            adj = 0
            tss_idx = 0
            idx = 0
        return data_exist, X_epi, Y, adj, idx, tss_idx

    def calculate_loss(model_gat, chr_list, cell_lines, batch_size, assay_type, fdr):
        loss_gat_all = np.array([])
        rho_gat_all = np.array([])
        Y_hat_all = np.array([])
        Y_all = np.array([])
        for num, cell_line in enumerate(cell_lines):
            for i in chr_list:
                if options.generalizable == 0:
                    file_name = data_path+chr_tfr_path+'/chr' +str(i)+'.tfr'
                else:
                    file_name = data_path+chr_tfr_path+'/chr' +str(i)+'.tfr'

                iterator = dataset_iterator(file_name, batch_size)
                while True:
                    data_exist, X_epi, Y, adj, idx, tss_idx = read_tf_record_1shot(iterator)
                    if data_exist:
                        if tf.reduce_sum(tf.gather(tss_idx, idx)) > 0:
                            Y_hat, _ = model_gat([X_epi, adj])
                            Y_hat_idx = tf.gather(Y_hat, idx, axis=1)
                            Y_idx = tf.gather(Y, idx, axis=1)

                            loss = poisson_loss(Y_idx, Y_hat_idx)
                            loss_gat_all = np.append(loss_gat_all, loss.numpy())
                            e1 = np.random.normal(0,1e-6,size=len(Y_idx.numpy().ravel()))
                            e2 = np.random.normal(0,1e-6,size=len(Y_idx.numpy().ravel()))

                            rho_gat_all = np.append(rho_gat_all, np.corrcoef(np.log2(Y_idx.numpy().ravel()+1)+e1,np.log2(Y_hat_idx.numpy().ravel()+1)+e2)[0,1])
                            Y_hat_all = np.append(Y_hat_all, Y_hat_idx.numpy().ravel())
                            Y_all = np.append(Y_all, Y_idx.numpy().ravel())
                    else:
                        break

        print('len of test/valid Y: ', len(Y_all))
        valid_loss = np.mean(loss_gat_all)
        rho = np.mean(rho_gat_all)

        return valid_loss, rho

    # Parameters
    T = 400
    b = 50
    N = 3*T                       # number of 5Kb bins inside 6Mb region
    F = 3                         # feature dimension
    F_ = 32                       # output size of GraphAttention layer
    n_attn_heads = 4              # number of attention heads in GAT layers
    dropout_rate = 0.5            # dropout rate
    l2_reg = 0.0                  # factor for l2 regularization
    re_load = False

    # Model definition
    if re_load:
        model_name = 'model_name.h5'
        model = tf.keras.models.load_model(model_name, custom_objects={'GraphAttention': GraphAttention})
        model.summary()
    else:
        tf.keras.backend.clear_session()
        X_in = Input(shape=(3*T*b,F))
        A_in = Input(shape=(N,N))

        # Full Transformer implementation
        x = layers.Reshape((N,b*F))(X_in)
        att_layers = options.att_layers
        att_heads = options.att_heads
        
        if options.posenc == 0:
            print("Not using positional encoder")
            sample_encoder = Encoder(num_layers=att_layers,
                         d_model=150,
                         num_heads=att_heads,
                         dff=2048)
        else:
            print("Using positional encoder")
            sample_encoder = Encoder_Posenc(num_layers=att_layers,
                         d_model=150,
                         num_heads=att_heads,
                         dff=2048, seq_len=N)

        x = sample_encoder(x)
        # Output of Transformer encoder still has size (1200, 150)
        # so we need to get it down to (1200, 128)
        x = layers.Conv1D(128, 3 , activation='relu', padding='same')(x)
	    
        att=[]
        for i in range(options.n_gat_layers):
            x, att_ = GraphAttention(F_,
                        attn_heads=n_attn_heads,
                        attn_heads_reduction='concat',
                        dropout_rate=dropout_rate,
                        activation='elu',
                        kernel_regularizer=l2(l2_reg),
                        attn_kernel_regularizer=l2(l2_reg))([x, A_in])
            x = layers.BatchNormalization()(x)
            att.append(att_)

        x = Dropout(dropout_rate)(x)
        x = layers.Conv1D(64, 1, activation='relu', padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)

        mu_cage = layers.Conv1D(1, 1, activation='exponential', padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg))(x)
        mu_cage = layers.Reshape([3*T])(mu_cage)

        # Build model
        model_gat = Model(inputs=[X_in, A_in], outputs=[mu_cage, att])
        model_gat._name = 'Epi-GraphReg'
        model_gat.summary()
        #print(len(model_gat.trainable_variables))
        #keras.utils.plot_model(model, 'GAT.png', show_shapes = True)


    ########## training ##########
    print("Training")
    cell_line = options.cell_line
    cell_lines = [cell_line]
    if options.generalizable == 0:
        model_name_gat = data_path+'/models/'+cell_line+'/Epi-GraphReg_'+cell_line+'_'+options.assay_type+'_FDR_'+fdr+'_valid_chr_'+options.valid_chr+'_test_chr_'+options.test_chr+'.h5'
    else:
        #model_name_gat = data_path+'/models/'+cell_line+'/Epi-GraphReg_generalizable_'+cell_line+'_'+options.assay_type+'_FDR_'+fdr+'_valid_chr_'+options.valid_chr+'_test_chr_'+options.test_chr+'.h5'
        model_name_gat = data_path+'/models/'+cell_line+'/Epi-GraphReg_RPGC_'+cell_line+'_'+options.assay_type+'_FDR_'+fdr+'_valid_chr_'+options.valid_chr+'_test_chr_'+options.test_chr+'.h5'

    if options.organism == 'mouse':
        train_chr_list = [c for c in range(1,1+19)]
        valid_chr_list = valid_chr
        test_chr_list = test_chr
        vt = valid_chr_list + test_chr_list
        for j in range(len(vt)):
            train_chr_list.remove(vt[j])
    else:
        train_chr_list = [c for c in range(1,1+22)]
        valid_chr_list = valid_chr
        test_chr_list = test_chr
        vt = valid_chr_list + test_chr_list
        for j in range(len(vt)):
            train_chr_list.remove(vt[j])

    print("train_chr_list: ", train_chr_list)
    print("valid_chr_list: ", valid_chr_list)
    print("test_chr_list: ", test_chr_list)

    best_loss = 1e20
    max_early_stopping = 10
    early_stopping_counter=1
    ## n_epochs = 200
    n_epochs = options.epochs
    lr = options.lr
    print("lr: ", lr)
    opt = tf.keras.optimizers.Adam(learning_rate=lr, decay=1e-6)
    batch_size = 1
    t0 = time.time()
    for epoch in range(1,n_epochs+1):
        loss_gat_all = np.array([])
        rho_gat_all = np.array([])
        Y_hat_all = np.array([])
        Y_all = np.array([])
        for num, cell_line in enumerate(cell_lines):
            for i in train_chr_list:
                if options.generalizable == 0:
                    file_name_train = data_path+chr_tfr_path+'/chr'+str(i)+'.tfr'
                else:
                    file_name_train = data_path+chr_tfr_path+'/chr'+str(i)+'.tfr'

                iterator_train = dataset_iterator(file_name_train, batch_size)
                train_count = 0
                while True:
                    data_exist, X_epi, Y, adj, idx, tss_idx = read_tf_record_1shot(iterator_train)
                    if data_exist:
                        if tf.reduce_sum(tf.gather(tss_idx, idx)) > 0:
                            with tf.GradientTape() as tape:
                                Y_hat, _ = model_gat([X_epi, adj])
                                Y_hat_idx = tf.gather(Y_hat, idx, axis=1)
                                Y_idx = tf.gather(Y, idx, axis=1)
                                loss = poisson_loss(Y_idx, Y_hat_idx)

                            grads = tape.gradient(loss, model_gat.trainable_variables)
                            opt.apply_gradients(zip(grads, model_gat.trainable_variables))

                            loss_gat_all = np.append(loss_gat_all, loss.numpy())
                            e1 = np.random.normal(0,1e-6,size=len(Y_idx.numpy().ravel()))
                            e2 = np.random.normal(0,1e-6,size=len(Y_idx.numpy().ravel()))

                            rho_gat_all = np.append(rho_gat_all, np.corrcoef(np.log2(Y_idx.numpy().ravel()+1)+e1,np.log2(Y_hat_idx.numpy().ravel()+1)+e2)[0,1])
                            Y_hat_all = np.append(Y_hat_all, Y_hat_idx.numpy().ravel())
                            Y_all = np.append(Y_all, Y_idx.numpy().ravel())
                            train_count+=1
                    else:
                        break
                ## print("train_count: ", train_count)
        if epoch == 1:
            print(Y_all)
            print("train_count: ", train_count)
            print("batch size: ", batch_size)
            print('len of train Y: ', len(Y_all))

        print("rho_gat_all: ", rho_gat_all)
        print("rho_gat_all.shape: ", rho_gat_all.shape)
        train_loss = np.mean(loss_gat_all)
        rho = np.mean(rho_gat_all)
        print('epoch: ', epoch, ', train loss: ', train_loss, ', train rho: ', rho, ', time passed: ', (time.time() - t0), ' sec')
        valid_loss,  valid_rho = calculate_loss(model_gat, valid_chr_list, cell_lines, batch_size, assay_type, fdr)
        print('epoch: ', epoch, ', valid loss: ', valid_loss, ', valid rho: ', valid_rho, ', time passed: ', (time.time() - t0), ' sec')
        test_loss,  test_rho = calculate_loss(model_gat, test_chr_list, cell_lines, batch_size, assay_type, fdr)
        print('epoch: ', epoch, ', test loss: ', test_loss, ', test rho: ', test_rho, ', time passed: ', (time.time() - t0), ' sec')

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
