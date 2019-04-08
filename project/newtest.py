import io
import math
import matplotlib.pyplot as plt
import numpy as np
from numpy import dot
from numpy.linalg import norm


#give a number k nearest (progress bar to choose number of neighbours
#tie both display in case of a tie

def getdictionary():

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
    word1='government'
    word2='president'

    if obj.get(word1)!= obj.get(word2):
        if obj.get(word1) != None:
            xco = obj.get(word1)

        if obj.get(word2) != None:
            yco= obj.get(word2)
    else:
        print("words are the same")



    return xco,yco



def plotattempt(x_axis,y_axis):
    x1= np.float32(x_axis)
    average1 = np.average(x1)
    y1 = np.float32(y_axis)
    z=np.arange(0,100)
    z1=np.float32(z)
    # print(z1)
    # print(x1,y1)
    average2 = np.average(y1)
    plt.scatter(z1, x1,c='green')
    plt.scatter(z1, y1,c='red')
    plt.title("Similarity")
    plt.xlabel('nananan')
    # plt.xlim(-0.1,0.1)
    # plt.ylim(-0.1,0.1)
    # plt.text(average1,average1,"word1")
    # plt.text(average2, average2, "word2")
    # print(average1,average2)
    # plt.figure()
    plt.plot(average1)

    plt.show()

def k_nearest(obj):
    # word1 = input('Please insert word: ')
    word1 = 'there'
    neighbours=10
    distance=[]
    n1=obj.get(word1)
    for i in dict_obj.values():
        distance.append(euclidean(n1,i))
    print(distance)




def euclidean(x_axis,y_axis):
    a=np.float32(x_axis)
    b=np.float32(y_axis)
    print(a,b)
    cos_sin = dot(a, b)/(norm(a)*norm(b))
    print(cos_sin)

    dst=math.sqrt(sum([(x-y)**2 for x,y in zip(a,b)]))
    print(dst)
    return dst

def angle(x_axis,y_axis):
    a=np.float32(x_axis)
    b=np.float32(y_axis)
    print(a,b)
    cos_sin = dot(a, b)/(norm(a)*norm(b))
    print(cos_sin)

    return cos_sin


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




k1, k2 = load_words(dict_obj)
plotattempt(k1,k2)
# euclidean(k1,k2)
# k_nearest(dict_obj)
angle(k1,k2)





# print(dict_obj.keys())
