import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output,Input
import plotly.plotly as plt
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objs as go
import io
import math
import dash_table

import numpy as np
from numpy import dot
from numpy.linalg import norm

#give a number k nearest (progress bar to choose number of neighbours
#tie both display in case of a tie
#rasio buttons anfle and distance calculated re apearing word input buttons
#create requirements.txt file run with pip install -r requirements.txt
# talk about pretrained methods used by glove

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# def generate_table(dataframe, max_rows):
#     return html.Table(
#         # Header
#         [html.Tr([html.Th(col) for col in dataframe.columns])] +
#
#         # Body
#         [html.Tr([
#             html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
#         ]) for i in range(min(len(dataframe), max_rows))]
#     )

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
    sortedDict = sorted(dict_obj1.items(), key=lambda x: round(x[1],2), reverse=False)# This returns the list of key-value pairs in the dictionary, sorted by value from  lowest to highest

    return sortedDict[1:count+1]

def euclidean(x_axis,y_axis):
    a = np.float32(x_axis)
    b = np.float32(y_axis)
    dst = math.sqrt(sum([(x-y)**2 for x, y in zip(a, b)]))
    return dst

def angle(x_axis,y_axis):
    a=np.float32(x_axis)
    b=np.float32(y_axis)
    cos_sin = dot(a, b)/(norm(a)*norm(b))
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

df=pd.DataFrame(k_nearestdistance(10),columns={'Words','Distance'})




app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([

        html.Div(
            [
                html.H1(children='Words2GO',
                        className='eight columns',
                        style={'text-align': 'center','letter-spacing': '0.5cm','text-shadow':' 0.08cm 0.08cm'},),
                html.Img(
                    src="https://www.ithehouse.org/wp-content/uploads/2015/05/060209-black-ink-grunge-stamp-textures-icon-people-things-speech.png",
                    className='four columns',
                    style={
                        'height': '9%',
                        'width': '9%',
                        'float': 'right',
                        'position': 'relative',
                        'margin-top': 10,

                    },
                ),

            ], className="row",
            style={'backgroundColor': '#FDEBD0'},
        ),

        html.Div(
                [
                    dcc.Tabs(id='tabs',children=[
                        dcc.Tab(label='Compare Words',children=[
                            html.P('Insert First Word'),
                            dcc.Input(id='compare-input-box1', type='text', placeholder='Enter word', value='him'),
                            html.P('Insert Second Word'),
                            dcc.Input(id='compare-input-box2',type='text',placeholder='Enter Word',value='her'),
                            html.Br(),
                            html.Button('Update graph',id='Compare=button')
                    ]),

                        dcc.Tab(label='Closest neighbours', children=[
                        html.P('Insert word to calculate closest neighbours'),
                        dcc.Input(id='input-box',type='text',placeholder='Enter word',value='him'),
                        html.P('Choose method for calculating distance:'),


                        dcc.RadioItems(
                                id = 'distance',
                                options=[
                                    {'label': 'Euclidean', 'value': 'euclidean'},
                                    {'label': 'Cosine', 'value': 'cosine'}
                                ],
                                value=['euclidean'],
                                labelStyle={'display': 'inline-block'}
                        ),html.P('Choose number of neighbours')
                            ,dcc.Slider(
                                        id='my-slider',
                                        min=0,
                                        max=10,
                                        step=1,
                                        marks={i:'no: {}'.format(i) for i in range(10)},
                                        value=10,
                                    ),html.Br(),


                    ],
                    className='six columns',
                    style={'margin-top': '10'}
                )
                ], colors={
                "border" : "white",
                "primary" : "gold",
                "background" : "cornsilk"
            })

            ], className="row"
        ),



        html.Div([
        html.Div([


                dcc.Graph(id='example-graph')
                ], className='six columns'
                )
                ,html.Div([
                html.Br(),
                    dash_table.DataTable(
                    id='table',
                    columns=[{"name": i, "id": i} for i in df.columns],
                    data=df.to_dict("rows"),
                    style_cell={'width': '80px'},
                )
                ], className= 'six columns'
                )
            ], className="row"
        )
    ], className='twelve columns')

@app.callback(
    Output('example-graph', 'figure'),
    [Input('compare-input-box1', 'value'),
     Input('compare-input-box2', 'value')])
def update_image_src(selector1, selector2):
    data = []
    # if 'euclidean' in selector:
    #     data.append({'x': [1, 2, 3], 'y': [4, 1, 2], 'mode': 'markers', 'name': 'SF'})
    # if 'cosine' in selector:
    #     data.append({'x': [1, 2, 3], 'y': [2, 4, 5], 'mode': 'markers', 'name': u'Montréal'})
    w1 = dict_obj.get(selector1)
    w2 = dict_obj.get(selector2)
    w3 = np.arange(0, 100)


    # Create a trace
    trace1 = go.Scatter(
        x=convToFloat(w1),
        y=convToFloat(w3),
        mode='markers+text')

    # Create a trace2
    trace2 = go.Scatter(
        x=convToFloat(w2),
        y=convToFloat(w3),
        mode='markers+text')

    data.append(trace1)
    data.append(trace2)


    figure = {
        'data': data,
        'layout': {
            'title':selector1 +' vs '+selector2 ,
            'showlegend' : True,
            'xaxis' : dict(
                visible=False,
                titlefont=dict(
                family='Courier New, monospace',
                size=20,
                color='#7f7f7f'
            )),
            'yaxis' : dict(

                titlefont=dict(
                family='Helvetica, monospace',
                size=20,
                color='#7f7f7f'
            ))
        }
    }
    return figure

# @app.callback(Output('table-container', 'children')
#     , [Input('my-slider', 'value')])
# def update_table(value):
#     dff = eucli_dic# update with your own logic
#     return generate_table(dff,value)
# def generate_table(dataframe, max_rows):
#     return html.Table(
#         # Header
#         [html.Tr([html.Th(col) for col in dataframe.columns])] +
#
#         # Body
#         [html.Tr([
#             html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
#         ]) for i in range(min(len(dataframe), max_rows))]
#     )
#
# @app.callback(
#     Output('hidden-form', 'style'),
#     [Input('toggle', 'value')])
# def toggle_closest(toggle_value):
#     if toggle_value == 'Closest neighbours':
#         return {'display': 'block'}
#     else:
#         return{'display':'none'}
# @app.callback(
#     Output('hidden-graph', 'style'),
#     [Input('toggle', 'value')])
# def toggle_compare(toggle_value):
#     if toggle_value == 'Compare Words':
#         return {'display': 'none'}
#     else:
#         return{'display':'block'}


# @app.callback(
#     Output('table', 'children'),
#     [Input('distance', 'value')])
# def update_image_src(selector):
#
#     if 'euclidean' in selector:
#         # data.append({'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'markers', 'name': 'SF'})
#         table = ff.create_table(k_nearestdistance(10))
#     if 'cosine' in selector:
#         # data.append({'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'markers', 'name': u'Montréal'})
#         table=ff.create_table(k_nearestangle(10))
#
#
#
#     return table

if __name__ == '__main__':
    app.run_server(debug=True)




