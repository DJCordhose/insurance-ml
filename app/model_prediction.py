from InsuranceModel import InsuranceModel

insurance_model = InsuranceModel()

# model_path = "/home/olli/insurance-ml/app/classifier"
# model_path = "/home/olli/insurance-ml/app/keras_model"
model_path = "INSERT PATH TO YOUR MODEL HERE"
# model_path = "/python_server/classifier"
insurance_model.load_model(model_path, keras_format=True)

def predict(speed, age, _):
    probas, result = insurance_model.predict(age, speed)
    return result, probas