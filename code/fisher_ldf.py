from __future__ import division
from utils import *
import numpy as np
from confusion_matrix import plotConfusionMatrix, confusionMatrix, class_accuracy
import matplotlib.pyplot as plt


def prepareTrainingset():
    '''
    loads dataset and converts all the image array to grayscale images
    :param file:
    :return: array of images and array of classes
    '''
    images, cls = load_training_data(first_batch_only = True)
    images = images / 255
    images = np.average(images, axis=3)
    return images, cls

def prepareTestset():
    '''
    loads dataset and converts all the image array to grayscale images
    :param file:
    :return: array of images and array of classes
    '''
    images, cls = load_test_data()
    images = images / 255
    images = np.average(images, axis=3)
    return images, cls

def get_dominant_eig(mat, n):
    '''
    returns n number of dominant eigenvectors
    :param mat:
    :param n:
    :return:
    '''
    eigvals, eigvecs = np.linalg.eig(mat)
    eiglist = [(eigvals[i], eigvecs[:, i]) for i in range(len(eigvals))]

    # sort the eigvals in decreasing order
    eiglist = sorted(eiglist, key=lambda x: x[0], reverse=True)

    # take the first n eigvectors
    w = np.array([eiglist[i][1] for i in range(n)])

    return w

def fisher_ldf_classifier(training_data):
    scatter_mat = 0
    overall_mean = training_data['overall_mean']
    sum_cov_mat = 0
    for _class, value in training_data['data'].items():
        images = np.asarray(value['images'])
        training_data['data'][str(_class)]['images'] = images

        # calculate class mean
        class_mean = np.average(images, axis = 0)
        training_data['data'][str(_class)]['class_mean'] = class_mean.reshape(1,-1)

        #calculate class covariance
        sum_mat = 0
        for image in images:
            diff = (image - class_mean).reshape(-1, 1)
            result = np.matmul(diff, np.transpose(diff))
            sum_mat = sum_mat + result
        class_cov_mat = (sum_mat)/len(images)
        training_data['data'][str(_class)]['class_cov_mat'] = class_cov_mat

        # compute scatter matrix
        diff_mean = (class_mean - overall_mean).reshape(1,-1)
        scatter_mat = scatter_mat + np.matmul(np.transpose(diff_mean), diff_mean)

        #compute sum of covariance matrix
        sum_cov_mat = sum_cov_mat + class_cov_mat

        del training_data['data'][str(_class)]['images']

    training_data['scatter_mat'] = scatter_mat
    training_data['sum_cov_mat'] = sum_cov_mat

    mat = np.matmul( np.linalg.pinv(sum_cov_mat), scatter_mat)

    dominant_eig_vec_transposed = get_dominant_eig(mat, 9)
    dominant_eig_vec = np.transpose(dominant_eig_vec_transposed)
    training_data['h'] = dominant_eig_vec

    for _class, value in training_data['data'].items():
        class_mean = value['class_mean'].reshape(-1, 1)
        class_cov_mat = value['class_cov_mat']
        #computing parameter m
        m = np.matmul(dominant_eig_vec_transposed, class_mean)
        training_data['data'][_class]["m"] = m

        #computing parameter s
        tmp = np.matmul(class_cov_mat, dominant_eig_vec)
        s = np.matmul(dominant_eig_vec_transposed, tmp)
        training_data['data'][_class]["s"] = s

    return training_data

def mahalanobis_distance(f, m, s):
    '''
    given feature vectors f, class mean m, and class covariance matrix s, it returns the mahanobis distance
    :param f: a numpy array of feature vector
    :param m: a numpy array of class mean
    :param s: a numpy array of class covariance matrix
    :return:
    '''
    diff = (f-m)
    tmp = np.matmul(np.transpose(diff), np.linalg.inv(s))
    distance = np.matmul(tmp, diff)
    return distance

def selectLeastDistance(classes, distances):
    '''
    selects the class with least distance
    :param classes: list of classes
    :param distances: list of distances
    :return: returns class with min class distance and min-distance
    '''
    assert len(classes)==len(distances)
    min_dist = min(distances)
    min_dist_index = distances.index(min_dist)
    best_class = classes[min_dist_index]
    return best_class, min_dist

def fisher_predict(X, fisher_parameters):
    '''
    returns a list of predictions for the given test samples X using the fisher parameters
    :param X: a numpy array of test samples with dimention N*d, where N is number of samples
    :param fisher_parameters: a dictionary object containing fisher parameters
    :return:
    '''
    predictions = []
    h = fisher_parameters['h']
    for image in X:
        image = image.flatten().reshape(-1,1)
        f = np.matmul(np.transpose(h), image)

        distances = []
        labels = []
        for i in range(len(fisher_parameters['data'].keys())):
            m = fisher_parameters['data'][str(i)]['m']
            s = fisher_parameters['data'][str(i)]['s']
            d = mahalanobis_distance(f, m, s)
            distances.append(d[0][0])

            labels.append(str(i))

        prediction, _ = selectLeastDistance(labels, distances)

        predictions.append(int(prediction))

    return predictions

def seggregate_class(X, Y):
    '''
    takes features and labels as input, and returns a special dictionary object
    For example: dataset_dict =
                {
                    data:{
                            class1: {
                                        images: [ array of images ],
                                        class_mean:
                                        class_cov_mat:
                                    }
                            class1: {
                                        images: [ array of images ],
                                        class_mean:
                                        class_cov_mat:
                                    }
                    },
                    overall_mean:
                    class_scatter_mat:
                }
    :param X: features
    :param Y:labels
    :return:
    '''

    sum_images = 0
    seggregated_data = {'data':{}}
    for i, _class in enumerate(Y):
        if str(_class) not in seggregated_data['data']:
            seggregated_data['data'][str(_class)] = {
                                                    'images': [],
                                                    'class_mean': None,
                                                    'class_cov_mat': None
                                                    }
        image_flattened = X[i].flatten()
        seggregated_data['data'][str(_class)]['images'].append(image_flattened)
        sum_images = sum_images + image_flattened

    # seggregated_data[str(_class)]['images'] = np.asarray(seggregated_data[str(_class)]['images'])
    seggregated_data['overall_mean'] = (sum_images/ len(X)).reshape(1,-1)

    return seggregated_data


def getAccuracy(true_labels, predictions):
    '''
    calculates accuracy
    :param true_labels:
    :param predictions:
    :return: returns accuracy
    '''
    true_labels = list(true_labels)
    result = list(map(lambda x,y: (1 if int(x)==int(y) else 0), predictions, true_labels))
    accuracy = sum(result)/(len(result))
    return accuracy


def main():
    X_train, Y_train = prepareTrainingset()
    X_test, Y_test = prepareTestset()

    seggregated_data = seggregate_class(X_train, Y_train)
    # seggregated_data = seggregate_class(X_train[:1000], Y_train[:1000])

    fisher_parameters = fisher_ldf_classifier(seggregated_data)

    # training accuracy
    print("=================Training Summary=======================")
    predictions = fisher_predict(X_train, fisher_parameters)
    accuracy = getAccuracy(Y_train, predictions)
    print("Training accuracy: ", accuracy)
    cm, classes = confusionMatrix(Y_train, predictions)
    plotConfusionMatrix(cm, classes)

    cls_accuracy, classes = class_accuracy(cm, classes)
    cls_error_rate = 1 - cls_accuracy

    print("Classes: ", classes)
    print("Class Accuracy: \n", np.round(cls_accuracy, 3))
    print("Class error rate: \n", np.round(cls_error_rate, 3))



    # test accuracy
    print("=================Testing Summary=======================")
    predictions = fisher_predict(X_test, fisher_parameters)
    accuracy = getAccuracy(Y_test, predictions)
    print("Testing accuracy: ", accuracy)
    cm, classes = confusionMatrix(Y_test, predictions)
    plotConfusionMatrix(cm, classes)

    cls_accuracy, classes = class_accuracy(cm, classes)
    cls_error_rate = 1 - cls_accuracy

    print("Classes: ", classes)
    print("Class Accuracy: \n", np.round(cls_accuracy, 3))
    print("Class error rate: \n", np.round(cls_error_rate, 3))


if __name__ == '__main__':
    main()