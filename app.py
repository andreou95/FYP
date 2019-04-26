import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output,Input
import pandas as pd
import plotly.graph_objs as go
import io
import math
import matplotlib.pyplot as plt
import numpy as np
from numpy import dot
from numpy.linalg import norm

#give a number k nearest (progress bar to choose number of neighbours
#tie both display in case of a tie
#rasio buttons anfle and distance calculated re apearing word input buttons
#create requirements.txt file run with pip install -r requirements.txt

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)



# df = pd.read_csv('test.txt')






def getdictionary():

    with io.open("test.txt", "r", encoding="utf-8") as df:
        for line in df.readlines():
            words = line.split()
            dict_obj.key=words[0]
            dict_obj.value = words[1:]
            dict_obj.add(dict_obj.key, dict_obj.value)
            # print(dict_obj)

        return dict_obj





def load_words(obj):

    # word1 = input('Please insert word: ')
    # word2 = input('Please insert word: ')
    word1='him'
    word2='her'

    if obj.get(word1)!= obj.get(word2):
        if obj.get(word1) != None:
            xco = obj.get(word1)

        if obj.get(word2) != None:
            yco= obj.get(word2)
    else:
        print("words are the same")




    return xco,yco

def convToFloat(x_axis):
    x1 =np.float32(x_axis)
    return x1




def plotattempt(x_axis,y_axis):
    x1= np.float32(x_axis)
    y1 = np.float32(y_axis)
    z=np.arange(0,100)
    z1=np.float32(z)
    # print(z1)
    # print(x1,y1)
    average1 = np.average(x1)
    average2 = np.average(y1)
    # plt.scatter(z1, x1,c='green')
    # plt.scatter(z1, y1,c='red')
    # plt.title("Similarity")
    # plt.xlabel('nananan')
    # plt.xlim(-0.1,0.1)
    # plt.ylim(-0.1,0.1)
    # plt.text(average1,average1,"word1")
    # plt.text(average2, average2, "word2")
    # print(average1,average2)
    # plt.figure()
    # plt.plot(average1)
    # plt.plot(average2)
    #
    # plt.show()
    return x1


def k_nearestangle(k):
    count=k
    dict_obj1 = my_dictionary()
    for i,j in dict_obj.items():
        value=angle(j,k1)
        dict_obj1.add(i,value)
    sortedDict=(sorted(dict_obj1.items(), key=lambda x: x[1], reverse=True))# This returns the list of key-value pairs in the dictionary, sorted by value from highest to lowest
    return sortedDict[1:count]

def k_nearestdistance(k):
    count = k
    dict_obj1 = my_dictionary()

    for i,j in dict_obj.items():
        value=euclidean(j,k1)
        dict_obj1.add(i,value)
    sortedDict= sorted(dict_obj1.items(), key=lambda x: x[1] , reverse=False)# This returns the list of key-value pairs in the dictionary, sorted by value from  lowest to highest

    return sortedDict[1:count+1]





def euclidean(x_axis,y_axis):
    a=np.float32(x_axis)
    b=np.float32(y_axis)
    # print(a,b)

    dst=math.sqrt(sum([(x-y)**2 for x,y in zip(a,b)]))
    # print(dst)
    return dst

def angle(x_axis,y_axis):
    a=np.float32(x_axis)
    b=np.float32(y_axis)
    # print(a,b)
    cos_sin = dot(a, b)/(norm(a)*norm(b))
    # print(cos_sin)
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


# print(dict_obj1)






# euclidean(k1,k2)
# angle(k1,k2)
# k_nearestangle(10)
# k_nearestdistance(10)

app.layout = html.Div([
html.Div([
    html.H1("Words 2 Vec",style={'backgroundColor': '#FDEBD0'}),
    html.Label('First Word'),
    dcc.Input(id='input-1', value='', type='text',spellCheck='False'),#we set this to true so it doesnt spell check for now
    html.Label('Second Word'),
    dcc.Input(id='input-2', type='text'),
    html.Div(id='input-div'),
    html.Button('submit',id='submit-button',style={'display': 'inline-block'})
    # ,    dcc.Graph(
    #     id='word embeddings',
    #     figure={
    #         'data': [
    #             go.Scatter(
    #                 x=convToFloat(k1),
    #                 y=convToFloat1(k2),
    #                 mode='markers',
    #                 opacity=0.8,
    #                 marker={
    #                     'size': 15,
    #                     'line': {'width': 0.5, 'color': 'white'}
    #                 },
    #
    #             )
    #         ],
    #         'layout': go.Layout(
    #
    #             yaxis={'title': 'Words'},
    #             margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
    #
    #             hovermode='closest'
    #         )
    #     }
    # )
])])

@app.callback(Output('input-div', 'children'),
              [Input('input-1', 'value'),
               Input('input-2', 'value')])
def update_output(input1, input2):
    return u'Input 1 is "{}" and Input 2 is "{}"'.format(input1, input2)

@app.callback(Output('submit-button','children'),
              [Input('input-1', 'value'),
               Input('input-2', 'value')])
def search_button(input1,input2):
    w1 = dict_obj.get(input1)
    w2 = dict_obj.get(input2)
    z = np.arange(0, 100)
    # if w1 is None:
    #     if w2 is None:
    #         print("ERROR")
    return html.Div([html.Div([
        dcc.Graph(
        id='word embeddings',
        figure={
            'data': [
                go.Scatter(
                    x=convToFloat(w1),
                    y=convToFloat(z),
                    mode='markers',
                    opacity=0.8,
                    marker={
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'white'}
                    },

                )
    ,go.Scatter(
                    x1=convToFloat(w2),
                    y1=convToFloat(z),
                    mode='markers',
                    opacity=0.8,
                    marker={
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'blue'}
                    },

                )
            ],
            'layout': go.Layout(
                showlegend=False,
                xaxis=dict(zeroline=False),
                yaxis=dict(hoverformat='2.f'),
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                hovermode='closest'
            )
        }
    )])])







if __name__ == '__main__':
    app.run_server(debug=True)