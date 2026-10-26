
from train_model import create_model

def test_model():
    model = create_model()
    assert abs(model.predict([[4]])[0] - 8) < 0.01
