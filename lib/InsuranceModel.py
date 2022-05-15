import tensorflow as tf

class InsuranceModel:
    def __init__(self, model=None, data=None, batch_size=32, age_range=(10, 150), max_speed_range=(50, 250), keras_format=True):
        self.model = model
        self.data = data
        self.batch_size = batch_size
        self.history = None
        self.keras_format = keras_format
        self.age_range = age_range
        self.max_speed_range = max_speed_range

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

    def check_range(self, age, speed, throw=False):
        min_age, max_age = self.age_range 
        min_speed, max_speed = self.max_speed_range 
        valid = age >= min_age and age <= max_age and speed >= min_speed and speed <= max_speed
        if throw and not valid:
            raise Exception(f'age {age} and/or max_speed {max_speed} outside of valid range')
        return valid

    def predict(self, age, max_speed):
        probas = self.predict_proba(age, max_speed)
        return probas[0].tolist(), int(probas.argmax())

    def predict_proba(self, age, max_speed):
        self.check_range(age, max_speed, throw=True)
        X = [[age, max_speed]]
        return self.model.predict(X)
