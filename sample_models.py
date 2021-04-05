from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, Dropout)

def simple_rnn_model(input_dim, output_dim=29, mode = 'GRU'):
    """ Build a recurrent network for speech 
    Input >> RNN_1 >> Softmax >> Output
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    if mode == 'GRU':
        simp_rnn = GRU(output_dim, return_sequences=True, 
                         implementation=2, name='rnn')(input_data)
    elif mode == 'SimpleRNN':
        simp_rnn = SimpleRNN(output_dim, return_sequences=True, name='rnn')(input_data)
        
    elif mode == 'LSTM':
        simp_rnn = LSTM(output_dim, return_sequences=True, 
                         implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29, mode = 'GRU'):
    """ Build a recurrent network for speech
    Input >> RNN_1 >> Batch_Normalization_1 >> Time Distribution >> Softmax >> Output
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    if mode == 'GRU':
        simp_rnn = GRU(units, activation=activation,
            return_sequences=True, implementation=2, name='rnn')(input_data)
    elif mode == 'SimpleRNN':
        simp_rnn = SimpleRNN(units, activation=activation,
            return_sequences=True, name='rnn')(input_data)
    elif mode == 'LSTM':
        simp_rnn = LSTM(units, activation=activation,
            return_sequences=True, implementation=2, name='rnn')(input_data)
    # Add batch normalization 
    bn_rnn = BatchNormalization()(simp_rnn)
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29, mode = 'GRU'):
    """ Build a recurrent + convolutional network for speech
    Input >> CNN 1-D >> RNN_1 >> Batch Normalizer_1 >> Time Distribution >> Softmax >> Output
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    if mode == 'SimpleRNN':
        simp_rnn = SimpleRNN(units, activation='relu',
            return_sequences=True, name='rnn')(bn_cnn)
    elif mode == 'GRU':
        simp_rnn = GRU(units, activation='relu',
            return_sequences=True, name='rnn')(bn_cnn)
    elif mode == 'LSTM':
        simp_rnn = LSTM(units, activation='relu',
            return_sequences=True, name='rnn')(bn_cnn)
    # Add batch normalization
    bn_rnn = BatchNormalization()(simp_rnn)
    # Add a TimeDistributed layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29, mode = 'GRU'):
    """ Build a deep recurrent network for speech
    Input >> RNN_1 >> Batch Normalizer_1 >> ... >> RNN_L >> Batch Normalizer_L >> Time Distribution >> Softmax >> Output
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    inputs = [input_data]
    # Add recurrent layers, each with batch normalization
    for i in range(recur_layers):
        if mode == 'SimpleRNN':
            simp_rnn = SimpleRNN(units[i], activation='relu',
                return_sequences=True, name='rnn_' + str(i))(inputs[i])
        elif mode == 'GRU':
            simp_rnn = GRU(units[i], activation='relu',
               return_sequences=True, name='rnn_' + str(i))(inputs[i])
        elif mode == 'LSTM':
            simp_rnn = LSTM(units[i], activation='relu',
                return_sequences=True, name='rnn_' + str(i))(inputs[i])
            
        bn_rnn = BatchNormalization(name = 'bn_rnn_{}'.format(i))(simp_rnn)
        inputs.append(bn_rnn)
    
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29, mode = 'GRU'):
    """ Build a bidirectional recurrent network for speech
    Input >> Bidirectional_RNN_1 >> Batch Normalizer_1 >> Time Distribution >> Softmax >> Output
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    if mode == 'GRU':
        bidir_rnn = Bidirectional(GRU(units, activation='relu',
                return_sequences=True, name='rnn'))(input_data)
    elif mode == 'LSTM':
        bidir_rnn = Bidirectional(GRU(units, activation='relu',
                return_sequences=True, name='rnn'))(input_data)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def final_model(input_dim, units, recur_layers, filters, kernel_size, conv_stride,
    conv_border_mode, output_dim=29, mode = 'GRU'):
    """ Build a deep network for speech 
    Input >> CNN 1-D >> Bi_RNN_1 >> Batch Normalizer_1 >> ... >> Bi_RNN_L >> Batch Normalizer_L >> Time_Dense >> Softmax >> Output
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    inputs = [bn_cnn]
    # Add recurrent layers, each with batch normalization
    for i in range(recur_layers):
        if mode == 'SimpleRNN':
            simp_rnn = Bidirectional(SimpleRNN(units[i], activation='relu',
                dropout_W = 0.2, dropout_U = 0.1,
                return_sequences=True, name='rnn_' + str(i)))(inputs[i])
        elif mode == 'GRU':
            simp_rnn = Bidirectional(GRU(units[i], activation='relu',
               dropout_W = 0.2, dropout_U = 0.1,                          
               return_sequences=True, name='rnn_' + str(i)))(inputs[i])
        elif mode == 'LSTM':
            simp_rnn = Bidirectional(LSTM(units[i], activation='relu',
                dropout_W = 0.2, dropout_U = 0.1, 
                return_sequences=True, name='rnn_' + str(i)))(inputs[i])
            
        bn_rnn = BatchNormalization(name = 'bn_rnn_{}'.format(i))(simp_rnn)
        inputs.append(bn_rnn)
    ...
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # Specify model.output_length
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride