
import pickle

def load_data():
    # load
    with open('./train-data-job-1/data-labels.pkl', 'rb') as f:
        data = pickle.load(f)
    return data
