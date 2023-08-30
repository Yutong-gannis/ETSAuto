import pickle


def save_pkl(path, data):
    file = open(path, 'wb')
    pickle.dump(data, file)
    file.close()


def load_pkl(path):
    try:
        file = open(path, 'rb')
        data = pickle.load(file)
        file.close()
        return data
    except EOFError:
        print('EOFError')
        return None
