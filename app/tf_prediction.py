import tensorflow as tf

# loading models only once on startup
# https://www.tensorflow.org/guide/keras/save_and_serialize

model_path = "/home/olli/insurance-ml/app/keras_model.h5"
# model_path = "/python_server/keras_model"

model = tf.keras.models.load_model(model_path)
# model = tf.keras.models.load_model("C:/Users/olive/Development/insurance-ml/app/classifier")
# model = tf.keras.models.load_model("C:/Users/oliver.zeigermann/Development/insurance-ml/app/classifier")

def predict(speed, age, miles):
    sample = [[age, speed]]

    prediction = model.predict(sample)
    result = prediction.argmax()

    return int(result), prediction[0].tolist()