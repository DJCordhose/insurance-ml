import pytest
import sys 
sys.path.append('lib')    

from TrainableInsuranceModel import TrainableInsuranceModel
from data import RiskGroup

insurance_model = TrainableInsuranceModel(None)
model_path = "/home/olli/insurance-ml/app/keras_model"
insurance_model.load_model(model_path, keras_format=True)

def test_known_low_category():
    _, category = insurance_model.predict(48, 100)
    assert category == RiskGroup.LOW.value

def test_known_high_category():
    _, category = insurance_model.predict(30, 150)
    assert category == RiskGroup.HIGH.value    

def test_invalid_range():
    with pytest.raises(ValueError):
        insurance_model.predict(500, 500)