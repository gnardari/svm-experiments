import gzip
import string
import os
import csv
import numpy as np
import pandas as pd
import cv2

'''

MNIST DATASET READER
adapted from: https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/contrib/learn/python/learn/datasets/mnist.py

'''

def _read32(bytestream):
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def extract_images(f):
  """Extract the images into a 4D uint8 np array [index, y, x, depth].
  Args:
    f: A file object that can be passed into a gzip reader.
  Returns:
    data: A 4D uint8 np array [index, y, x, depth].
  Raises:
    ValueError: If the bytestream does not start with 2051.
  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                       (magic, f.name))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = np.frombuffer(buf, dtype=np.uint8)
    data = data.reshape(num_images, rows, cols)
    return data

def extract_labels(f, one_hot=False, num_classes=10):
  """Extract the labels into a 1D uint8 np array [index].
  Args:
    f: A file object that can be passed into a gzip reader.
    one_hot: Does one hot encoding for the result.
    num_classes: Number of classes for the one hot encoding.
  Returns:
    labels: a 1D uint8 np array.
  Raises:
    ValueError: If the bystream doesn't start with 2049.
  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                       (magic, f.name))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = np.frombuffer(buf, dtype=np.uint8)
    if one_hot:
      return dense_to_one_hot(labels, num_classes)
    return labels

def read_mnist(base_dir='../datasets/mnist', test=True):
    mnist = {'train': {'data': None, 'labels': None},
             'test': {'data': None, 'labels': None}}

    file_path = os.path.join(base_dir, 'train-images-idx3-ubyte.gz')
    with open(file_path, 'rb') as f:
        mnist['train']['data'] = extract_images(f)

    file_path = os.path.join(base_dir, 'train-labels-idx1-ubyte.gz')
    with open(file_path, 'rb') as f:
        mnist['train']['labels'] = extract_labels(f)

    if not test:
        return mnist

    file_path = os.path.join(base_dir, 't10k-images-idx3-ubyte.gz')
    with open(file_path, 'rb') as f:
        mnist['test']['data'] = extract_images(f)

    file_path = os.path.join(base_dir, 't10k-labels-idx1-ubyte.gz')
    with open(file_path, 'rb') as f:
        mnist['test']['labels'] = extract_labels(f)

    return mnist

'''

    IMAGENET

'''

def read_imagenet(base_dir='../datasets/imagenet/animals', split=0.8):
    split = 0.9

    lbs = {'dogs': 1, 'cats': 2, 'fish': 3, 'birds': 4}

    imagenet = {'train': {'data': None, 'labels': None},
                'test': {'data': None, 'labels': None}}

    data = []
    labels = []
    for d in os.listdir(base_dir):
        class_dir = os.path.join(base_dir, d)
        dir_content = os.listdir(class_dir)

        data += [cv2.imread(os.path.join(class_dir, img))
                 for img in dir_content]
        labels += [lbs[d]] * len(dir_content)

    idx = np.arange(len(data))
    np.random.shuffle(idx)

    tr_pct = int(len(idx)*split)
    tr_idx = idx[:tr_pct]
    te_idx = idx[tr_pct:]

    data = np.array(data)
    labels = np.array(labels)

    imagenet['train']['data'] = data[tr_idx]
    imagenet['train']['labels'] = labels[tr_idx]
    imagenet['test']['data'] = data[te_idx]
    imagenet['test']['labels'] = labels[te_idx]

    return imagenet

'''

    MILLION SONG DATASET GENRE RECOGNITION

'''

def read_msd(base_dir='../datasets/musicas'):
    msd = {'train': {'data': None, 'labels': None},
           'test': {'data': None, 'labels': None}}

    train = np.load(os.path.join(base_dir, 'msd-25-genre-train.npz'))
    tis = train['inputs'].shape
    msd['train']['data'] = np.reshape(train['inputs'], (tis[0], tis[1]*tis[2]))
    msd['train']['labels'] = np.array(train['targets'])

    test = np.load(os.path.join(base_dir, 'msd-25-genre-valid.npz'))
    tis = test['inputs'].shape
    msd['test']['data'] = np.reshape(test['inputs'], (tis[0], tis[1]*tis[2]))
    msd['test']['labels'] = test['targets']

    return msd

'''

    MOVIE REVIEW SENTIMENT ANALYSIS

'''

def read_movie_reviews(base_dir='../datasets/movie-sentiment'):
    reduce_labels = lambda x: 0 if x < 2 else (1 if x == 2 else 2)

    reviews = {'train': {'data': None, 'labels': None},
               'test': {'data': None, 'labels': None}}


    path = os.path.join(base_dir,
                        'train.tsv')
    data = []
    labels = []
    last_id = None

    with open(path, 'rb') as f:
        tsv = csv.reader(f, delimiter='\t')

        # skip headers
        next(tsv)

        for i, row in enumerate(tsv):
            if last_id == row[1]:
                continue

            last_id = row[1]

            # lower case and no punctuation
            data.append(row[2].lower().translate(None,
                                                 string.punctuation))
            labels.append(int(row[3]))

    split = int(0.9 * len(labels))
    labels = map(reduce_labels, labels)

    reviews['train']['data'] = np.array(data[:split])
    reviews['train']['labels'] = np.array(labels[:split])

    reviews['test']['data'] = np.array(data[split:])
    reviews['test']['labels'] = np.array(labels[split:])

    return reviews

'''

    SANTANDER CUSTOMER SATISFACTION

'''

def read_santander(base_dir='../datasets/santander'):
    dataset = pd.read_csv(os.path.join(base_dir, 'train.csv'))

    san = {'train': {'data': None, 'labels': None},
           'test': {'data': None, 'labels': None}}
    # remove constant columns (std = 0)
    remove = []
    for col in dataset.columns:
        if dataset[col].std() == 0:
            remove.append(col)

    dataset.drop(remove, axis=1, inplace=True)

    # remove duplicated columns
    remove = []
    cols = dataset.columns
    for i in range(len(cols)-1):
        v = dataset[cols[i]].values
        for j in range(i+1,len(cols)):
            if np.array_equal(v,dataset[cols[j]].values):
                remove.append(cols[j])

    dataset.drop(remove, axis=1, inplace=True)

    data = dataset.drop(["TARGET","ID"],axis=1)
    labels = dataset.TARGET.values

    split = int(0.9 * len(labels))
    san['train']['data'] = np.array(data[:split])
    san['train']['labels'] = np.array(labels[:split])

    san['test']['data'] = np.array(data[split:])
    san['test']['labels'] = np.array(labels[split:])

    return san
