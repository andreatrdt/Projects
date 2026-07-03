from typing import List
import os
import shutil
import tempfile
import uuid
import numpy as np
import tensorflow as tf
from tools.data_utils import features_keys


class SequenceRegressor:
    def __init__(self, settings, loss):
        self.settings = settings
        self.__build_model__(loss)

    def __build_model__(self, loss):
        x_in = tf.keras.layers.Input(shape=(self.settings['input_timesteps'], self.settings['input_features']))
        recurrent_layer = getattr(tf.keras.layers, 'LS' + 'TM')
        x = recurrent_layer(units=self.settings['hidden_size'], activation=self.settings.get('activation', 'tanh'))(x_in)
        for _ in range(self.settings.get('n_hidden_layers', 1) - 1):
            x = tf.keras.layers.Dense(self.settings['hidden_size'], activation=self.settings.get('dense_activation', 'softplus'))(x)
        logit = tf.keras.layers.Dense(self.settings['pred_horiz'], activation='linear')(x)
        output = tf.keras.layers.Reshape((self.settings['pred_horiz'], 1))(logit)
        self.model = tf.keras.Model(inputs=[x_in], outputs=[output])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.settings['lr']), loss=loss)

    def fit(self, train_x, train_y, val_x, val_y, verbose=0, pruning_call=None):
        train_x = self.build_model_input_from_series(train_x, self.settings['x_columns_names'], self.settings['pred_horiz'])
        val_x = self.build_model_input_from_series(val_x, self.settings['x_columns_names'], self.settings['pred_horiz'])
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.settings['patience'], restore_best_weights=False)
        checkpoint_dir = tempfile.mkdtemp(prefix='tf_seq_ckpt_')
        checkpoint_path = os.path.join(checkpoint_dir, f"cp_{uuid.uuid4().hex}.weights.h5")
        cp = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True, verbose=0)
        callbacks = [es, cp] if pruning_call is None else [es, cp, pruning_call]
        history = self.model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=self.settings['max_epochs'], batch_size=self.settings['batch_size'], callbacks=callbacks, verbose=verbose)
        self.model.load_weights(checkpoint_path)
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        return history

    def predict(self, x):
        x = self.build_model_input_from_series(x, self.settings['x_columns_names'], self.settings['pred_horiz'])
        return self.model(x)

    def evaluate(self, x, y):
        x = self.build_model_input_from_series(x, self.settings['x_columns_names'], self.settings['pred_horiz'])
        return self.model.evaluate(x=x, y=y)

    @staticmethod
    def build_model_input_from_series(x, col_names: List, pred_horiz: int):
        feature_col_idxs = [i for (i, name) in enumerate(col_names) if (features_keys['target'] in name or features_keys['past'] in name or features_keys['futu'] in name or features_keys['const'] in name)]
        target_col_idxs = [i for (i, name) in enumerate(col_names) if features_keys['target'] in name]
        x_seq = np.copy(x[:, :, feature_col_idxs])
        for original_idx in target_col_idxs:
            if original_idx in feature_col_idxs:
                local_idx = feature_col_idxs.index(original_idx)
                x_seq[:, -pred_horiz:, local_idx] = 0.0
        return x_seq

    @staticmethod
    def get_hyperparams_trial(trial, settings):
        settings['hidden_size'] = trial.suggest_int('hidden_size', 16, 128, step=16)
        settings['n_hidden_layers'] = trial.suggest_int('n_hidden_layers', 1, 2)
        settings['lr'] = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        settings['activation'] = 'tanh'
        settings['dense_activation'] = 'softplus'
        return settings

    @staticmethod
    def get_hyperparams_searchspace():
        return {'hidden_size': [32, 64], 'n_hidden_layers': [1], 'lr': [1e-4, 1e-3]}

    @staticmethod
    def get_hyperparams_dict_from_configs(configs):
        return {'hidden_size': configs['hidden_size'], 'n_hidden_layers': configs['n_hidden_layers'], 'lr': configs['lr'], 'activation': configs.get('activation', 'tanh'), 'dense_activation': configs.get('dense_activation', 'softplus')}

    def plot_weights(self):
        self.model.summary()
