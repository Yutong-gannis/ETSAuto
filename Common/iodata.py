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
    except pickle.UnpicklingError:
        print('UnpicklingError: ' + path)
        return None
    except EOFError:
        print('EOFError: ' + path)
        return None
