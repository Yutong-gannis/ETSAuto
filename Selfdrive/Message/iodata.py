import pickle


def save_pkl(path, data):
    """This is function to save data to .pkl

    :param path: The path to save
    :type path: str
    :param data: The saving data
    :type data: dict
    """
    file = open(path, 'wb')
    pickle.dump(data, file)
    file.close()


def load_pkl(path):
    """This is function to load data from .pkl file

    :param path: The path to load
    :type path: str
    :return: The loading data
    :rtype: dict
    """
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
