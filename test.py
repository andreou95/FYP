import pandas as pd
import sys 
import io
import numpy as np
import matplotlib.pyplot as plt

words=[]
vectors=[]
vectorkeys=[]
vec={}
line=0


def dictionary():
	
	

	with io.open("test.txt","r",encoding="utf-8") as dictionary: 
		for line in dictionary.readlines():
			Line=line.split()
			key=Line[0]
			
			value=line[len(key):]

			if key in vec:
				vec[key].append(value)
			else:
				vec[key]=[value]
			vectors=vec.values()
			vectorkeys=vec.keys()
			lista=list(vec.values())
			print("key",vec.get('end'))
			
			
		#print(lista)
		#print(value)
		#print(vectors[](0))
		#print(vectorkeys)
		
		
		
	return lista

def test():
	vectorkeys=vec.keys()
	vectorvalues=vec.values()
	#for key,val in vec.items():
		#print(key,"=>",val)

	return vectorkeys,vectorvalues

def load_words():
	#if radio button not clicked implement later
	search=[]
	word1=[]
	word2=[]

	word1=input('Please insert word: ')
	word2=input('Please insert word: ')

	search=vec.get(word1)
	print(search)



	return(search)

def plotting():
	search=vec.get('the')
	a,b=([1,2,3]),([1,2,2])
	x=np.array(a)
	y=np.array(b)
	print(x,y)
	plt.plot(x,y)
	plt.ylabel('testing')
	plt.title('title')
	plt.show()
	return a 

def turntovectors():
	search=vec.get('the')
	b=([1,2,3],[1,2,2,3])
	a=vector(b)
	print(a)
	plt.plot([1,2,3,],[1,2,2])
	plt.ylabel('testing')
	plt.title('title')
	plt.show()
	
	print(a)
	return a


	
dictionary()
test()	
#turntovectors()
plotting()


