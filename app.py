import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output,Input
import plotly.plotly as plt
import pandas as pd
import plotly.graph_objs as go
import io
import math

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
    x1 = np.float32(x_axis)
    return x1






def k_nearestangle(k):
    count = k
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
        value = euclidean(j, k1)
        dict_obj1.add(i, value)
    sortedDict = sorted(dict_obj1.items(), key=lambda x: x[1], reverse=False)# This returns the list of key-value pairs in the dictionary, sorted by value from  lowest to highest

    return sortedDict[1:count+1]





def euclidean(x_axis,y_axis):
    a = np.float32(x_axis)
    b = np.float32(y_axis)
    dst = math.sqrt(sum([(x-y)**2 for x, y in zip(a, b)]))
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
    html.H1("Words 2 Vec",style={'backgroundColor': '#FDEBD0',
                                 'border':'0px'}),
    html.Label('First Word'),
    dcc.Input(id='input-1', value='', type='text',spellCheck='False'),#we set this to true so it doesnt spell check for now
    html.Label('Second Word'),
    dcc.Input(id='input-2', type='text'),
    html.Div(id='input-div',),
    html.Button('submit',id='submit-button',style={'display': 'inline-block'}),
    html.Div([dcc.Dropdown(
        id='my-dropdown',
        options=[
            {'label': 'Euclidean', 'value': 'euclidean'},
            {'label': 'Cosine', 'value': 'cosine'}
        ],
        searchable=False,
        placeholder='Select Distance function to be used',
        value='euclidean',

    )]),html.Div([
    dcc.Graph(id='my-graph')
], style={'width': '500','display': 'none'})
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
    z = np.arange(0,len(w1))
    # if w1 is None:
    #     if w2 is None:
    #         print("ERROR")
    return html.Div([html.Div([
        dcc.Graph(
        id='word embeddings',
        figure={
            'data': [
                go.Scatter(
                    x=np.mean(convToFloat(w1)),
                    y=np.mean(convToFloat(z)),
                    mode='markers + text',
                    opacity=0.8,
                    name='Word 1',
                    marker={
                        'maxdisplayed': 90,
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'white'}
                    },

                )
    ,go.Scatter(
                    x1=np.mean(convToFloat(w2)),
                    y1=np.mean(convToFloat(z)),
                    mode='markers',
                    opacity=0.8,
                    name='Word 2',
                    marker={
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'blue'}
                    },

                )
            ],
            'layout': go.Layout(
                yaxis={'title': 'Life Expectancy'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'x':0,'y':1},
                hovermode='closest',

            )
        }
    )])])
@app.callback(Output('my-graph', 'figure'), [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown_value):

    if selected_dropdown_value!= None:
        if selected_dropdown_value == 'euclidean':
            df = k_nearestdistance(10)
        else:
            df = k_nearestangle(10)


    return html.H3( {df})

        #     'data': [{
        #         'x': df.keys(),
        #         'y': df.Close
        #     }],
        #     'layout': {'margin': {'l': 40, 'r': 0, 't': 20, 'b': 30}}
        #








if __name__ == '__main__':
    app.run_server(debug=True)