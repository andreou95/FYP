import numpy as np
from scipy.spatial import distance







def load_english_dictionary():
    with open('words_alpha.txt') as word_file:
        valid_words = set(word_file.read().split())

    return valid_words

def euclidean_distance():
	
	a = (1, 2, 3)
	b = (4, 5, 6)
	dst = distance.euclidean(a, b)
	return dst

def load_words():
	
	word1=[]
	word1=input('Please insert word: ')
	word2=[]
	word2=input('Please insert word: ')

	return(word1,word2)

if __name__ == '__main__':
    english_words = load_english_dictionary()
    a=english_words
    distance=euclidean_distance()
    b=distance
    word1,word2 = load_words()
    print(word1,word2)
