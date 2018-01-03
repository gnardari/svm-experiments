from sklearn.feature_extraction.text import TfidfVectorizer
import skimage.feature as ski
import cv2
import mahotas as mh
import numpy as np

# img2 = cv2.drawKeypoints(img, corners[-1], dummy, color=(255,0,0))
# cv2.imshow('asdd', img2)
# cv2.waitKey(0)

def haralick(dataset):
    data = []
    for img in dataset:
        features = np.hstack(mh.features.haralick(img))
        data.append(features)
    data /= np.max(np.abs(data), axis=0)
    return np.array(data)

def hog(dataset):
    return np.array([ski.hog(img) for img
            in dataset])

def gabor(dataset, ksize=(3,3), sigma=1.0, theta=np.pi/4, lamb=2.0, gamma=0.5):
    data = []
    gkernel = cv2.getGaborKernel(ksize, sigma, theta, lamb, gamma)
    for img in dataset:
        filtered_img = cv2.filter2D(img, cv2.CV_8UC3, gkernel)
        image = np.hstack(filtered_img)
        data.append(image)
    return np.array(data)

def fastCorner(dataset):
    fast = cv2.FastFeatureDetector_create()
    corners = []
    dummy = np.zeros((1,1))
    for img in dataset:
        corners.append(fast.detect(img, None))
    return corners

def fourrier(dataset):
    for img in dataset:
        rows, cols = img.shape
        crow,ccol = rows/2 , cols/2

        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)

        magnitude_spectrum = 20*np.log(np.abs(fshift))
        fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        cv2.imshow('fourrier', img_back)
        cv2.waitKey(0)
        break

def surf(dataset, threshold=400):
    features = []
    surf = cv2.xfeatures2d.SURF_create()
    for img in dataset:
        print(img.shape)
        kp, desc = surf.detectAndCompute(img, None)
        print(desc.shape)
        print(desc[0].shape)
        features.append(desc)
        exit()
    features = np.array(features)
    print(features.shape)
    return features

def vgg_convolutions(dataset):
    # avoiding tf message on import of this module
    from keras.applications.vgg16 import VGG16
    from keras.preprocessing import image
    from keras.applications.vgg16 import preprocess_input

    model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    features = []
    for img in dataset:
        x = np.array(cv2.resize(img, (224,224)), dtype=np.float64)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features.append(model.predict(x).flatten())
    return np.array(features)

def tfidf(tr_data, te_data):
    vect = TfidfVectorizer(ngram_range=(1,3),
                           max_df=0.7)

    tr_data = vect.fit_transform(tr_data)
    te_data = vect.transform(te_data)

    return tr_data.todense(), te_data
