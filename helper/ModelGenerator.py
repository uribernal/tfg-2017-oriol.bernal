"""
This assistant returns different Keras models (architectures)
that may need to be compiled and trained.
"""
from keras.layers import LSTM, BatchNormalization, Dense, Dropout, Input, TimeDistributed
from keras.models import Model


def lstm_alberto_tfg_c3d(batch_size=32, time_steps=1, dropout_probability=.5, summary=False):

    input_features = Input(batch_shape=(batch_size, time_steps, 4096,), name='features')
    input_normalized = BatchNormalization(name='normalization')(input_features)
    input_dropout = Dropout(dropout_probability)(input_normalized)
    lstm = LSTM(512, return_sequences=True, stateful=False, name='lsmt1')(input_dropout)
    output_dropout = Dropout(dropout_probability)(lstm)
    output = TimeDistributed(Dense(1, activation='tanh'), name='fc')(output_dropout)

    model = Model(inputs=input_features, outputs=output)

    if summary:
        model.summary()
    return model


def lstm_alberto_tfg_c3d_val_ar(batch_size=32, time_steps=1, dropout_probability=.5, summary=False):

    input_features = Input(batch_shape=(batch_size, time_steps, 7168,), name='features')
    input_normalized = BatchNormalization(name='normalization')(input_features)
    input_dropout = Dropout(dropout_probability)(input_normalized)
    lstm = LSTM(512, return_sequences=True, stateful=False, name='lsmt1')(input_dropout)
    output_dropout = Dropout(dropout_probability)(lstm)
    output = TimeDistributed(Dense(2, activation='tanh'), name='fc')(output_dropout)

    model = Model(inputs=input_features, outputs=output)

    if summary:
        model.summary()
    return model


def lstm_alberto_tfg_activities(batch_size=32, dropout_probability=.5, summary=False):

    input_features = Input(batch_shape=(batch_size, 1, 201,), name='features')
    input_normalized = BatchNormalization(name='normalization')(input_features)
    input_dropout = Dropout(dropout_probability)(input_normalized)
    lstm = LSTM(512, return_sequences=True, stateful=False, name='lsmt1')(input_dropout)
    output_dropout = Dropout(dropout_probability)(lstm)
    output = TimeDistributed(Dense(1, activation='tanh'), name='fc')(output_dropout)

    model = Model(inputs=input_features, outputs=output)

    if summary:
        model.summary()
    return model


def two_layers_lstm(s1=2048, s3=512, batch_size=32, time_steps=1, dropout_probability=.5, summary=False):

    input_features = Input(batch_shape=(batch_size, time_steps, 4096), name='features')
    input_normalized = BatchNormalization(name='normalization')(input_features)
    input_dropout = Dropout(dropout_probability)(input_normalized)
    lstm = LSTM(s1, return_sequences=True, stateful=True, name='lsmt1')(input_dropout)
    output_dropout = Dropout(dropout_probability)(lstm)
    output_dropout2 = Dropout(dropout_probability)(output_dropout)
    lstm2 = LSTM(s3, return_sequences=True, stateful=True, name='lsmt3')(output_dropout2)
    output_dropout3 = Dropout(dropout_probability)(lstm2)
    output = TimeDistributed(Dense(1, activation='tanh'), name='fc')(output_dropout3)

    model = Model(inputs=input_features, outputs=output)

    if summary:
        model.summary()
    return model


def three_layers_lstm(s1=2048, s2=1024, s3=512, batch_size=32, time_steps=1, dropout_probability=.5, summary=False):

    input_features = Input(batch_shape=(batch_size, time_steps, 4096), name='features')
    input_normalized = BatchNormalization(name='normalization')(input_features)
    input_dropout = Dropout(dropout_probability)(input_normalized)
    lstm = LSTM(s1, return_sequences=True, stateful=True, name='lsmt1')(input_dropout)
    output_dropout = Dropout(dropout_probability)(lstm)
    lstm2 = LSTM(s2, return_sequences=True, stateful=True, name='lsmt2')(output_dropout)
    output_dropout2 = Dropout(dropout_probability)(lstm2)
    lstm3 = LSTM(s3, return_sequences=True, stateful=True, name='lsmt3')(output_dropout2)
    output_dropout3 = Dropout(dropout_probability)(lstm3)
    output = TimeDistributed(Dense(1, activation='tanh'), name='fc')(output_dropout3)

    model = Model(inputs=input_features, outputs=output)

    if summary:
        model.summary()
    return model


def lstm_raw_audio(batch_size=32, dropout_probability=.5, summary=False):

    input_features = Input(batch_shape=(batch_size, 1, 5*98*64), name='features')
    input_normalized = BatchNormalization(name='normalization')(input_features)
    input_dropout = Dropout(dropout_probability)(input_normalized)
    lstm = LSTM(512, return_sequences=True, stateful=False, name='lsmt')(input_dropout)
    output_dropout = Dropout(dropout_probability)(lstm)
    output = TimeDistributed(Dense(2, activation='tanh'), name='fc')(output_dropout)

    model = Model(inputs=input_features, outputs=output)

    if summary:
        model.summary()
    return model


def lstm_audio_features(batch_size=32, time_steps=1, dropout_probability=.5, summary=False):

    input_features = Input(batch_shape=(batch_size, time_steps, 3*2*512,), name='features')
    input_normalized = BatchNormalization(name='normalization')(input_features)
    input_dropout = Dropout(dropout_probability)(input_normalized)
    lstm = LSTM(512, return_sequences=True, stateful=False, name='lsmt1')(input_dropout)
    output_dropout = Dropout(dropout_probability)(lstm)
    output = TimeDistributed(Dense(2, activation='tanh'), name='fc')(output_dropout)

    model = Model(inputs=input_features, outputs=output)

    if summary:
        model.summary()
    return model
