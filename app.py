import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output,Input
import pandas as pd
import plotly.graph_objs as go
import io
import math
import dash_table
import numpy as np
from numpy import dot
from numpy.linalg import norm


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config['suppress_callback_exceptions']= True



def getdictionary():

    with io.open("test.txt", "r", encoding="utf-8") as df:
        for line in df.readlines():
            words = line.split()
            dict_obj.key=words[0]
            dict_obj.value = words[1:]
            dict_obj.add(dict_obj.key, dict_obj.value)
            # print(dict_obj)

    return dict_obj



def convToFloat(x_axis):
    x1 = np.float32(x_axis)
    return x1

def k_nearestangle(input,k):
    count = k
    dict_obj1 = my_dictionary()
    for i,j in dict_obj.items():
        value=angle(j,dict_obj.get(input))
        dict_obj1.add(i,round(value,4))
    sortedDict=(sorted(dict_obj1.items(), key=lambda x: x[1], reverse=True))# This returns the list of key-value pairs in the dictionary, sorted by value from highest to lowest
    return sortedDict[1:count]

def k_nearestdistance(input,k):

    count = k
    dict_obj1 = my_dictionary()

    for i,j in dict_obj.items():
        value = round(euclidean(j, dict_obj.get(input)),4)
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


app.layout = html.Div([html.Div([

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
                    dcc.Tabs(id='tabs',value='tab-1',children=[
                        dcc.Tab(label='Compare Words',value='tab-1')

                    ,

                        dcc.Tab(label='Closest neighbours',value='tab-2')



                    ],
                    className='row',
                    style={'margin-top': '10'},
                    colors={
                        "border" : "white",
                        "primary" : "gold",
                        "background" : "cornsilk"
                    }
                ),html.Div(id='tabs-hidden')
                ])

            ], className="row"
        ),



    html.Div([
        html.Div(id='hidden-graph-div',style={'display':'none'},children=[html.Div(id='hidden-graph',children=[dcc.Graph(id='example-graph')])

    ], className='six columns')
            ,html.Div(id='hidden-table-div',style={'display' : 'none'},children=[
             html.Br()
            ,html.Div(id='hidden-table',children=[])], className= 'six columns')


        ], className="row"
        )
], className='twelve columns')








@app.callback(Output('tabs-hidden', 'children'),
    [Input('tabs', 'value')])
def update_tabs(t1):
    if t1== 'tab-1':
        return html.Div([   html.Br(),
                            html.P('Instructions: In the section Compare Words The user is asked to insert 2 words to be compared and visualised in a plot'),
                            html.P('The plot will visualise the representation of the two word embeddingsas vectors of numbers.'),
                            html.P('Once the update graph button is clicked the values of these word embeddings are displayed'),
                            html.Br(),
                            html.P('Insert First Word'),
                            dcc.Input(id='compare-input-box1', type='text', placeholder='Enter word', value='him'),
                            html.P('Insert Second Word'),
                            dcc.Input(id='compare-input-box2',type='text',placeholder='Enter Word',value='her'),
                            html.Br(),
                            html.Button('Update graph',id='compare-button',n_clicks=0)
                      ],className='row')

    else:
        return html.Div([   html.Br(),
                            html.P('Instructions: In the section Closest neighbours The user is asked to insert a word he wishes'),
                            html.P('Step 1 choose the method of calculating the distance/angle from the radio buttons'),
                            html.P('Step 2 choose from the slider component the number of neighbouring words to be displayed'),
                            html.P('As a result the most similar words in the pre trained word embeddings file will be displayed in a table visualisation'),
                            html.Br(),
                            html.P('Insert word to calculate closest neighbours'),
                            dcc.Input(id='closest-input-box',type='text',placeholder='Enter word',value='him'),
                            html.P('Choose method for calculating distance:'),


                            dcc.RadioItems(
                                    id = 'distance',
                                    options=[
                                        {'label': 'Euclidean Distance', 'value': 'Euclidean'},
                                        {'label': 'Cosine Angle', 'value': 'Cosine'}
                                    ],

                                    labelStyle={'display': 'inline-block'}
                            ),html.P('Choose number of nearest neighbours')
                                ,dcc.Slider(
                                            id='my-slider',
                                            min=0,
                                            max=50,

                                            step=1,
                                            # marks={i:'=>{}'.format(i) for i in range(10)},
                                            value=15,
                                        ),html.Br(),
                            ],className='row')


@app.callback(
    Output('example-graph', 'figure'),
    [Input('compare-input-box1', 'value'),
     Input('compare-input-box2', 'value'),
     Input('compare-button','n_clicks')])
def update_image_src(selector1, selector2,n):
    
    trace=[]

    w1 = dict_obj.get(selector1)
    w2 = dict_obj.get(selector2)
    w3 = np.arange(0, 100)




    if n!=0:
        # Create a trace
        trace.append( go.Scatter(
            x=convToFloat(w1),
            y=convToFloat(w3),
            mode='markers',
            name=selector1))

        # Create a trace2
        trace.append(go.Scatter(
            x=convToFloat(w2),
            y=convToFloat(w3),
            mode='markers',
            name=selector2))






        data = trace



        figure = {
            'data': data,
            'layout': {
                'title':selector1 +' vs '+selector2 ,
                'showlegend' : True,
                'xaxis' : dict(
                    # visible=False,
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





@app.callback(
    Output('hidden-table-div', 'style'),
    [Input('tabs', 'value')])
def table_div_style(toggle_value):
    if toggle_value == 'tab-2':

        return {'display': 'block'}
    else:
        return {'display': 'none'}





@app.callback(
    Output('hidden-table', 'children'),
    [Input('closest-input-box','value')
        ,Input('my-slider','value')
        ,Input('distance','value')])
def table_style(in_value ,slider_value,distance_value):
    df=create_table_df(in_value,slider_value, distance_value)
    return  dash_table.DataTable(
                    id='table',
                    columns=[{"name": i, "id": i} for i in df.columns],
                    data=df.to_dict("rows"),
                    style_cell={'width': '80px'})

def create_table_df(in_value,n, flag):
    if flag == 'Euclidean':
        df = pd.DataFrame(k_nearestdistance(in_value,n), columns={ flag + ' Distance','Words'})

    else:
        df = pd.DataFrame(k_nearestangle(in_value,n), columns={ flag + ' Angle','Words'})
    return df

@app.callback(
        Output('hidden-graph-div', 'style'),
        [Input('tabs', 'value')])
def graph_div_style(tab_value):
    if tab_value == 'tab-2':
        return {'display': 'none'}

    else:
        return {'display': 'block'}



if __name__ == '__main__':
    app.run_server(debug=True)




