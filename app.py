#!/usr/bin/env python
"""
Author: Codegression
This file is responsible for creating a live demo webapp that segements user uploaded images
It should be run via server.py in a production environment.
"""

import datetime
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import PIL
import base64
import dash_bootstrap_components as dbc
import io
import datetime
import inference



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Water segmentation Using Deep Learning"
server = app.server

app.layout = html.Div([
    
   html.H1(children='Water Segmentation Using Deep Learning',
           style={'text-align': 'center', "margin-top": "50px"}),
   html.Br(),
   html.Div(children=[
       html.Img(src=app.get_asset_url('AI.png')),
       html.Img(src=app.get_asset_url('header.png'), style={
                "margin-left": "80px", "margin-bottom": "5px"})
   ], style={'text-align': 'center'}),
   html.Br(),
   html.Br(),
   html.Div(id='alpha-output', children='Water opacity level', style={'text-align': 'center'}),
        html.Div(children=[
            dcc.Slider(
                id='alpha-selector',
                min=0,
                max=255,
                step=1,
                value=190,
                marks={
                    0: '0%',
                    64: '25%',
                    128: '50%',
                    192: '75%',
                    255: '100%',
                },
            )],
            style={'align-items': 'center', 'margin-left':'180px', 'margin-right':'180px', 'text-align': 'center'}
        ),

   html.Br(),
   html.Br(),
   html.Br(),
   html.Div('Drag and drop one or more images onto the box below for a live demo.', style={'text-align': 'center'}),

    
    html.Div(children=dcc.Upload(
            id='upload-image',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '2px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'text-align': 'center'           
    
            },
            multiple=True
        ),
    style={'margin':'50px', 'text-align':'center'}),
    
    dcc.Loading(
               id="loading-1",
               type="'cube",    
               fullscreen=False,
               children=[    
                  html.Div(id='output-image-upload')])
], style={"border": "solid 1px #555",
          "background-color": "white",
          "box-shadow": "10px -10px 5px  rgba(0,0,0,0.6)",
          "-moz-box-shadow": "10px -10px 5px  rgba(0,0,0,0.6)",
          "-webkit-box-shadow": "10px -10px 5px  rgba(0,0,0,0.6)",
          "-o-box-shadow": "10px -10px 5px  rgba(0,0,0,0.6)",
          "border-radius": "25px",
          "margin-top": "25px",
          "margin-bottom": "25px",
          "margin-right": "50px",
          "margin-left": "50px"
          })

def parse_contents(contents, filename, date, alpha):
    """This function gets decodes base64-encoded input image, performs inference, and returns html output

    Args:
        contents (_type_): base64 encoded input image
        filename (_type_): original name of the uploaded file
        date (_type_): uploaded date and time
        alpha (_type_): opacity level for water

    Returns:
        dash.html: html output
    """
    base64_decoded = base64.b64decode(contents.split(',')[1])
    image = PIL.Image.open(io.BytesIO(base64_decoded))    
    
    outputimage, timeelapsed = inference.infer(image, alpha, interpolate=False)
    
    buff = io.BytesIO()
    outputimage.save(buff, format="PNG")
    new_image_string = "data:image/png;base64," + base64.b64encode(buff.getvalue()).decode("utf-8")
       
    inferencetime = int(timeelapsed.total_seconds() * 1000)
    fps = round(1 / timeelapsed.total_seconds(), 2)
    
    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=new_image_string, style={'max-width':'800px'}),

        
        html.Div('Inference time: ' + str(inferencetime) + 'ms'),
        html.Div('Frame rate: ' + str(fps) + ' fps'),
        html.Hr()
    ], style={'margin':'50px', 'text-align':'center'})

@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              Input('alpha-selector', 'value'),
              State('upload-image', 'filename'),
              State('upload-image', 'last_modified'))
def update_output(list_of_contents, alpha, list_of_names, list_of_dates):
    """This function gets called whenever user uploads new images or changes the opacity slider bar

    Args:
        list_of_contents (_type_): list of base64 encoded uploaded images
        alpha (_type_): opacity level for water (comes from the slider)
        list_of_names (_type_): list of names of uploaded files
        list_of_dates (_type_): list of uploaded dates/times

    Returns:
        dash.html : html output
    """
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d, alpha) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

if __name__ == '__main__':
    print("Running server")
    app.run_server(debug=False, port=8040, host= '0.0.0.0')
    #app.run_server()