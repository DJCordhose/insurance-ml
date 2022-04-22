# https://github.com/AndreasMadsen/python-lrcurve
from lrcurve import KerasLearningCurve
from viz import plot_metric_over_time, plot_decision_boundaries
from data import RiskGroup, DataType
import tensorflow as tf

class Model:
    def __init__(self, model, data, batch_size=32, keras_format=True):
        self.model = model
        self.data = data
        self.batch_size = batch_size
        self.history = None
        self.keras_format = keras_format

    def train(self, epochs=50, plot_curve=True):
        if not self.keras_format:
            # https://docs.python.org/3/library/exceptions.html#exception-hierarchy
            raise RuntimeError('Keras format is required for training')
        self.model.compile(loss='sparse_categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

        if plot_curve:
            callbacks = [KerasLearningCurve()]
        else:
            callbacks = []

        X_train, X_val, y_train, y_val = self.data.get_split()
        history = self.model.fit(X_train, y_train,
                                 validation_data=(X_val, y_val),
                                 epochs=epochs,
                                 batch_size=self.batch_size,
                                 callbacks=callbacks,
                                 verbose=0 if plot_curve else 1)
        if not self.history:
            self.history = history
        else:
            self.history.history['val_loss'] += history.history['val_loss']
            self.history.history['val_acc'] += history.history['val_acc']
            self.history.history['loss'] += history.history['loss']
            self.history.history['acc'] += history.history['acc']
        return self.history

    def evaluate(self):
        X_train, X_val, y_train, y_val = self.data.get_split()
        train_loss, train_metric = self.model.evaluate(
            X_train, y_train, batch_size=self.batch_size, verbose=0)
        test_loss, test_metric = self.model.evaluate(
            X_val, y_val, batch_size=self.batch_size, verbose=0)
        return ((train_loss, train_metric), (test_loss, test_metric))

    def plot_loss(self):
        if not self.history:
            return
        plot_metric_over_time('Loss over epochs',
                              self.history.history['loss'], self.history.history['val_loss'],
                              y_label='loss')

    def plot_accuracy(self):
        if not self.history:
            return
        plot_metric_over_time('Accuracy over epochs',
                              self.history.history['accuracy'], self.history.history['val_accuracy'],
                              y_label='accuracy')

    # https://keras.io/guides/serialization_and_saving/
    def load_model(self, model_path='classifier', keras_format=None):
        if keras_format is None:
            keras_format = self.keras_format

        if keras_format:
            self.model = tf.keras.models.load_model(f'{model_path}.h5')
        else:
            self.model = tf.saved_model.load(model_path)
        return self.model

    def save_model(self, model_path='classifier', keras_format=None):
        if keras_format is None:
            keras_format = self.keras_format

        if keras_format:
            self.model.save(f'{model_path}.h5', save_format='h5')
        else:
            self.model.save(model_path, save_format='tf')

    def predict(self, age, max_speed):
        probas = self.predict_proba(age, max_speed)
        return probas.argmax()

    def predict_proba(self, age, max_speed):
        X = [[age, max_speed]]
        return self.model.predict(X)

    # should better be a unit test
    def check_model_invariants(self):
        assert self.predict(48, 100) == RiskGroup.LOW.value
        assert self.predict(30, 150) == RiskGroup.HIGH.value
        
    def plot_decision_boundaries(self, plot_extended=False):
        X, y = self.data.get_data()
        if plot_extended:
            plot_decision_boundaries(self.model, X, y, x1_range=(10, 150), x2_range=(50, 250))
        else:
            plot_decision_boundaries(self.model, X, y)        
