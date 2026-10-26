
from train_model import create_model

def test_model():
    model = create_model()
    prediction = model.predict([[4]])
    assert abs(prediction[0] - 8) < 0.01
