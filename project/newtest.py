import io
import math
import matplotlib.pyplot as plt
import matplotlib.mlab
import numpy as np


def getdictionary():
    vec=[]
    with io.open("test.txt", "r", encoding="utf-8") as df:
        for line in df.readlines():
            words = line.split()
            dict_obj.key=words[0]
            dict_obj.value = words[1:]
            dict_obj.add(dict_obj.key, dict_obj.value)

        return dict_obj


def load_words(obj):

    # word1 = input('Please insert word: ')
    # word2 = input('Please insert word: ')
    word1='president'
    word2='there'
    if obj.get(word1) != None:
        xco = obj.get(word1)

    if obj.get(word2) != None:
        yco= obj.get(word2)



    return xco,yco



def plotattempt(x_axis,y_axis):
    x1 = np.c_[x_axis]
    y1 = np.c_[y_axis]
    plt.scatter(x1, y1,color='k',s=500 ,marker='*',alpha=0.5)
    plt.figure()
    plt.plot(x1,y1)
    plt.show()


# def calc_distance(x_axis,y_axis):
#     a=x_axis
#     b=y_axis
#     dst=math.sqrt(sum([(x-y)**2 for x,y in zip(a,b)]))
#     print(dst)
#     return dst

def euclidean5(vector1, vector2):
    ''' use matplotlib.mlab to calculate the euclidean distance. '''
    vector1=np.array(vector1)
    vector2 = np.array(vector2)
    dist = plt.mlab.dist(vector1, vector2)
    return dist






class my_dictionary(dict):

    # __init__ function
    def __init__(self):
        self = dict()

    # Function to add key:value
    def add(self, key, value):
        self[key] = value


# Main Function

dict_obj = my_dictionary()
dict_obj = getdictionary()


# dict_obj.add(dict_obj.key, dict_obj.value)

k1, k2 = load_words(dict_obj)
# print(k1, k2)
# plotattempt(k1,k2)
euclidean5(k1,k2)
#plotattempt(k2)



# print(dict_obj.keys())
