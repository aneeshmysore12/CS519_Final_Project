import plotly.graph_objs as go
import numpy as np
import matplotlib.cm as cm
from scipy.spatial import Delaunay
import pandas as pd
import plotly.express as px  # (version 4.7.0 or higher)
import plotly.graph_objects as go
import plotly.figure_factory as ff
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
from functools import reduce

app = dash.Dash(__name__)

app.layout = html.Div([

    html.H1("Triangulated California Temperature Map For Year 2010", style={'text-align': 'center'}),

    dcc.Dropdown(id="slct_month",
                 options=[
                     {"label": "January", "value": "January"},
                     {"label": "February", "value": "February"},
                     {"label": "March", "value": "March"},
                     {"label": "April", "value": "April"},  
                     {"label": "May", "value": "May"},
                     {"label": "June", "value": "June"},  
                     {"label": "July", "value": "July"},  
                     {"label": "August", "value": "August"},  
                     {"label": "September", "value": "September"},
                     {"label": "October", "value": "October"},  
                     {"label": "November", "value": "November"},  
                     {"label": "December", "value": "December"}],
                 multi=False,
                 value="January",
                 style={'width': "40%"}
                 ),

    html.Div(id='output_container', children=[]),
    html.Br(),

    dcc.Graph(id='my_3d_obj', figure={})

])

df_raw = pd.read_csv("2790101.csv")

df_new = df_raw[df_raw['DATE'] == 1]
df_new.reset_index(inplace=True)
x_cal_long = df_new['LONGITUDE'].tolist()
y_cal_lat = df_new['LATITUDE'].tolist()
z_cal = [0] * len(x_cal_long)
z_cal_temp = [item for item in (df_new['MLY-DUTR-NORMAL'].tolist())]
temp_mean=round(np.mean([item for item in z_cal_temp if np.isnan(item) == False]),1)
z_cal_temp = [ temp_mean if np.isnan(x) else x for x in z_cal_temp]


#define 2D points, as input data for the Delaunay triangulation of U

points2D_cal=np.vstack([x_cal_long,y_cal_lat]).T
tri_cal = Delaunay(points2D_cal)    #triangulate the rectangle U

sp1_index=np.where(tri_cal.points == [-120.1574,41.8715])[0][0]  
sp2_index=np.where(tri_cal.points == [-116.8669,36.4622])[0][0] 
sp3_index=np.where(tri_cal.points == [-115.5311,35.45917])[0][0] 
sp4_index=np.where(tri_cal.points == [-114.6188,34.7675])[0][0]  
sp5_index=np.where(tri_cal.points == [-114.1708,34.2903])[0][0]  
sp6_index=np.where(tri_cal.points == [-119.0142,38.2119])[0][0]
s_1 = {sp1_index,sp2_index,sp3_index}
s_2 = {sp1_index,sp3_index,sp4_index}
s_3 = {sp1_index,sp4_index,sp5_index}
s_4 = {sp1_index,sp2_index,sp6_index}

tri_cal.simplices = np.array([item for item in tri_cal.simplices 
                                if (set(item) == s_1 or 
                                     set(item) == s_2 or 
                                     set(item) == s_3 or 
                                     set(item) == s_4) == False])




def getTemperature(month):
    if month == 'January':
        scope = 1
    elif month == 'February':
        scope = 2
    elif month == 'March':
        scope = 3
    elif month == 'April':
        scope = 4
    elif month == 'May':
        scope = 5
    elif month == 'June':
        scope = 6
    elif month == 'July':
        scope = 7
    elif month == 'August':
        scope = 8
    elif month == 'September':
        scope = 9
    elif month == 'October':
        scope = 10
    elif month == 'November':
        scope = 11
    else:
        scope = 12
         
    df_new = df_raw[df_raw['DATE'] == scope]
    df_new.reset_index(inplace=True)
    z_cal_temp = [item for item in (df_new['MLY-DUTR-NORMAL'].tolist())]
    temp_mean=round(np.mean([item for item in z_cal_temp if np.isnan(item) == False]),1)
    z_cal_temp = [ temp_mean if np.isnan(x) else x for x in z_cal_temp]

    return z_cal_temp

def map_z2color(zval, colormap, vmin, vmax):
    #map the normalized value zval to a corresponding color in the colormap

    if vmin>vmax:
        raise ValueError('incorrect relation between vmin and vmax')
    t=(zval-vmin)/float((vmax-vmin))#normalize val

    R, G, B, alpha=colormap(t)
    return 'rgb('+'{:d}'.format(int(R*255+0.5))+','+'{:d}'.format(int(G*255+0.5))+\
           ','+'{:d}'.format(int(B*255+0.5))+')'


def tri_indices(simplices):
    #simplices is a numpy array defining the simplices of the triangularization
    #returns the lists of indices i, j, k

    return ([triplet[c] for triplet in simplices] for c in range(3))




def plotly_trisurf_cal(x, y, z, z_temperature, simplices, colormap=cm.RdBu, plot_edges=None):
    #x, y, z are lists of coordinates of the triangle vertices 
    #simplices are the simplices that define the triangularization;
    #simplices  is a numpy array of shape (no_triangles, 3)
    #insert here the  type check for input data

    points3D=np.vstack((x,y,z_temperature)).T
    tri_vertices=list(map(lambda index: points3D[index], simplices))# vertices of the surface triangles


    zmean=[np.mean(tri[:,2]) for tri in tri_vertices]  # mean values of z-coordinates of triangle vertices

    min_zmean=np.min(zmean)
    max_zmean=np.max(zmean)



    facecolor=[map_z2color(zz,  colormap, min_zmean, max_zmean) for zz in zmean]
    #print(facecolor)
    I,J,K=tri_indices(simplices)

    triangles=go.Mesh3d(x=x,
                     y=y,
                     z=z,
                     facecolor=facecolor,
                     i=I,
                     j=J,
                     k=K,
                     name=''
                    )

    if plot_edges is None:# the triangle sides are not plotted 
        return [triangles]
    else:
        #define the lists Xe, Ye, Ze, of x, y, resp z coordinates of edge end points for each triangle
        #None separates data corresponding to two consecutive triangles
        points3D=np.vstack((x,y,z)).T
        tri_vertices=list(map(lambda index: points3D[index], simplices))# vertices of the surface triangles

        lists_coord=[[[T[k%3][c] for k in range(4)]+[ None]   for T in tri_vertices]  for c in range(3)]
        Xe, Ye, Ze=[reduce(lambda x,y: x+y, lists_coord[k]) for k in range(3)]

        #define the lines to be plotted
        lines=go.Scatter3d(x=Xe,
                        y=Ye,
                        z=Ze,
                        mode='lines',
                        line=dict(color= 'rgb(50,50,50)', width=1.5)
               )

        return [triangles, lines]


axis = dict(
showbackground=True,
backgroundcolor="rgb(230, 230,230)",
gridcolor="rgb(255, 255, 255)",
zerolinecolor="rgb(255, 255, 255)",
    )
noaxis=dict(showbackground=False,
            showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title=''
          )
layout = go.Layout(
         #title='Triangulated California Temperature Map',
         width=1000,
         height=1000,
         scene=dict(xaxis=noaxis,
                    yaxis=noaxis,
                    zaxis=noaxis,
                    aspectratio=dict(x=10, y=10, z=0),
                    camera=dict(eye=dict(x=0.1,
                                     y=-1.5,
                                     z= 5)
                            )
                )
        )

@app.callback(
    [Output(component_id='output_container', component_property='children'),
     Output(component_id='my_3d_obj', component_property='figure')],
    [Input(component_id='slct_month', component_property='value')]
)

def update_graph(option_slctd):

    container = "The selected month is: {}".format(option_slctd)

    z_cal_temp = getTemperature(option_slctd)
    data2=plotly_trisurf_cal(x_cal_long,y_cal_lat, z_cal, z_cal_temp, tri_cal.simplices, colormap=cm.cubehelix, plot_edges=True)
    fig = go.Figure(data=data2, layout=layout)

    return container, fig


if __name__ == "__main__":
    app.run_server(debug=True)