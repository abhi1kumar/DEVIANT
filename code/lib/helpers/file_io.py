

"""
    File Input Output Operations for different kinds of files.
"""
import os
import numpy as np
import pandas as pd
import cv2
import pickle
import logging
import warnings
import json

def imread(path):

    return cv2.imread(path)


def imwrite(im, path):

    cv2.imwrite(path, im)


def read_csv(path, delimiter= " ", ignore_warnings= False, use_pandas= False):
    try:
        if ignore_warnings:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if use_pandas:
                    data = pd.read_csv(path, delimiter= delimiter, header=None).values
                else:
                    data = np.genfromtxt(path, delimiter= delimiter)
        else:
            if use_pandas:
                data = pd.read_csv(path, delimiter=delimiter, header=None).values
            else:
                data = np.genfromtxt(path, delimiter=delimiter)
    except:
        data = None

    return data


def write_csv(path, numpy_variable, delimiter= " ", save_folder= None, float_format= None):

    if save_folder is not None:
        path = os.path.join(save_folder, path)

    pd.DataFrame(numpy_variable).to_csv(path, sep= delimiter, header=None, index=None, float_format= float_format)


def read_lines(path, strip= True):
    with open(path) as f:
        lines = f.readlines()

    if strip:
        # you may also want to remove whitespace characters like `\n` at the end of each line
        lines = [x.strip() for x in lines]

    return lines


def write_lines(path, lines_with_return_character):
    with open(path, 'w') as f:
        f.writelines(lines_with_return_character)


def read_numpy(path, folder= None, show_message= True):
    if folder is not None:
        path = os.path.join(folder, path)

    if show_message:
        logging.info("=> Reading {}".format(path))

    return np.load(path)

def save_numpy(path, numpy_variable, save_folder= None, show_message= True):

    if save_folder is not None:
        path = os.path.join(save_folder, path)

    if show_message:
        logging.info("=> Saving to {}".format(path))
    np.save(path, numpy_variable)


def pickle_read(file_path):
    """
    De-serialize an object from a provided file_path
    """
    print("=> Loading pickle {}".format(file_path))
    with open(file_path, 'rb') as file:
        return pickle.load(file)


def pickle_write(file_path, obj):
    """
    Serialize an object to a provided file_path
    """
    logging.info("=> Saving pickle to {}".format(file_path))
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)

def read_json(json_path):
    print("=> Loading JSON {}".format(json_path))
    with open(json_path, 'rb') as file:
        return json.load(file)
