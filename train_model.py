
from sklearn.linear_model import LinearRegression
import pickle

def create_model():
    model = LinearRegression()
    model.fit([[1], [2], [3]], [2, 4, 6])
    return model

if __name__ == "__main__":
    model = create_model()
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model trained successfully!")
