import pickle

with open("weights.pkl", "rb") as f:
    state = pickle.load(f)

print(state)