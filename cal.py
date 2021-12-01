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
import dash_daq as daq

external_stylesheets = [
    {
        "href": "https://fonts.googleapis.com/css2?"
        "family=Lato:wght@400;700&display=swap",
        "rel": "stylesheet",
    },
]
cal = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = cal.server

cal.layout = html.Div(
    children=[
        html.Div(
            children=[
                #html.P(children="🥑", className="header-emoji"),
                #html.P(children="🥑", className="header-emoji"),
                html.H1(
                    children="Triangulated California Weather Map For Years 2010-2020", className="header-title"
                ),
                html.P(
                    children="Color mapping California using temperatures recorded at Calfornia weather collection stations from year 2010 to 2020",
                    className="header-description",
                ),
            ],
            className="header",
        ),
        html.Div(
            children=[
               html.Div(
                    children=[
                        html.Div(children="Weather Type", className="menu-title"),
                        dcc.Dropdown(
                            id="weather-type-filter",
                            options=[
                                {"label": "Temperature", "value": "Temperature"}],
                                # {"label": "Precipitation", "value": "Precipitation"}],
                            clearable=False,
                            searchable=False,
                            value="Temperature",                       
                            className="dropdown",
                        ),
                    ],
                ),
                dcc.Interval(id='auto-stepper-year',
                            interval=1*1000, # in milliseconds
                            n_intervals=0,
                            max_intervals=-1, #10,
                            disabled=True
                ),
                html.Div(
                    children=[
                        html.Div(
                            children="Year",
                            className="menu-title"
                            ),
                        dcc.Slider(
                            id='year-slider-filter',
                            min=2010,
                            max=2020,
                            step=1,
                            value=2010,
                            className="myslider",
                                marks={
                                2010: '2010',
                                2011: '2011',
                                2012: '2012',
                                2013: '2013',
                                2014: '2014',
                                2015: '2015',                               
                                2016: '2016',
                                2017: '2017',
                                2018: '2018',
                                2019: '2019',
                                2020: '2020',
                            },
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                        daq.PowerButton(
                            id='year-power-button',
                            on=False,
                            color = "#FF5E5E",
                            size = 40,
                            label = dict(label="Slide Show",style=dict(color="#079A82")),
                            labelPosition = "bottom"
                        ),
                    ]
                ),
                html.Div(
                    children=[
                        html.Div(
                            children="Month",
                            className="menu-title"
                            ),
                        dcc.Interval(id='auto-stepper-month',
                                    interval=1*1000, # in milliseconds
                                    n_intervals=0,
                                    max_intervals=-1, #11,
                                    disabled=False,
                        ),    
                        dcc.Slider(
                            id='month-slider-filter',
                            min=1,
                            max=12,
                            step=1,
                            value=1,
                            className="myslider",
                                marks={
                                1: 'JAN',
                                2: 'FEB',
                                3: 'MAR',
                                4: 'APR',
                                5: 'MAY',
                                6: 'JUN',                               
                                7: 'JUL',
                                8: 'AUG',
                                9: 'SEP',
                                10: 'OCT',
                                11: 'NOV',
                                12: 'DEC'
                            },
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                        daq.PowerButton(
                            id='month-power-button',
                            on=False,
                            color = "#FF5E5E",
                            size = 40,
                            label = dict(label="Slide Show",style=dict(color="#079A82")),
                            labelPosition = "bottom"
                        ),
                    ]
                ),
            ],
            className="menu",
        ),
        html.Div(
            dcc.Graph(id='cal_3d_obj', 
                        figure={'data':[],
                                'layout': go.Layout(
                                    xaxis={
                                        'showticklabels': False,
                                        'ticks': '',
                                        'showgrid': False,
                                        'zeroline': False
                                    },
                                    yaxis={
                                        'showticklabels': False,
                                        'ticks': '',
                                        'showgrid': False,
                                        'zeroline': False
                                    }
                                )}
            ),
            className="diagram"
        ),
    ]
)




#  The following commented code is used to read raw data and clean up missing data using the following logic:
#  1) Skip station that has 132 month of data (11 years)
#  2) Skip station that has more than 30 missing TAVG values
#  3) For each station, calculate the average value for the same month from other years which have values for that same year. 
#      using this average value to fill other missing data for this station for the same month
#  Finally write the data into csv file cal_2010_2020.csv


# years = ["2010","2011","2012","2013","2014","2015",
#             "2016","23017","2018","2019","2020"]
# months = ["-01","-02","-03","-04","-05","-06","-07","-08","-09","-10","-11","-12"]
# path = os.getcwd()
# csv_files = glob.glob(os.path.join("weather_data/2010_2020", "*.csv"))

# li = []
# mylen = 0
# for f in csv_files:
#     df_temp = pd.read_csv(f)
#     li.append(df_temp)

# df_master = pd.concat(li, ignore_index=True)

# df_master_stations = list(set(df_master["STATION"]))

# station_132m = []
# station_not_132m=[]
# for station in df_master_stations:
#     mystation_data = df_master[df_master["STATION"] == station]
#     if len(mystation_data) == 132:
#         station_132m.append(station)
#     else:
#         station_not_132m.append(station)

# mylist = []
# for goodstation in station_132m:
#     goodstation_data = (df_master[df_master["STATION"] == goodstation]).sort_values(by = "DATE")
#     goodstation_data.reset_index(inplace=True)
#     where_list = np.where(np.isnan(goodstation_data["TAVG"]))
#     if (len(where_list[0]) <= 30):
#         date_data = list(goodstation_data["DATE"])
#         tavg_data = goodstation_data["TAVG"]
#         station_monthly_tavg = [round(np.mean([item for item in tavg_data[np.where(np.char.find(date_data,month)>=0)[0]] 
#                             if np.isnan(item) == False]),1) for month in months]
#         for pos in where_list[0]:
#             goodstation_data["TAVG"][pos] = station_monthly_tavg[pos % 12]
#         mylist.append(goodstation_data)

# df_raw_10y=pd.concat(mylist,ignore_index=True)
# df_raw_10y.reset_index(inplace=True)
# df_raw_10y.to_csv('cal_2010_2020.csv', index=False)


months = ["-01","-02","-03","-04","-05","-06","-07","-08","-09","-10","-11","-12"]
months_str = ["January","February","March","April","May", "June","July","August","September","October","November","December"]

df_raw = pd.read_csv("cal_2010_2020.csv")
df_jan = df_raw[df_raw["DATE"] == "2010-01"]
x_cal_long = df_jan['LONGITUDE'].tolist()
y_cal_lat = df_jan['LATITUDE'].tolist()
z_cal = [0] * len(y_cal_lat)
temperature_list=list(df_raw["TAVG"])
tmin = np.min(temperature_list)
tmax = np.max(temperature_list)

points2D_cal=np.vstack([x_cal_long,y_cal_lat]).T
tri_cal = Delaunay(points2D_cal)


sp1_index=np.where(tri_cal.points == [-120.18,41.99])[0][1]  # 354
sp2_index=np.where(tri_cal.points == [-118.358,37.3711])[0][1] #202
sp3_index=np.where(tri_cal.points == [-117.1449,36.602])[0][1] # 80
sp4_index=np.where(tri_cal.points == [-116.8672,36.4626])[0][1]  # 130
sp5_index=np.where(tri_cal.points == [-115.9092,35.7706])[0][1]  # 389
sp6_index=np.where(tri_cal.points == [-114.6188,34.7675])[0][1]  #66
sp7_index=np.where(tri_cal.points == [-119.37,38.44])[0][1]  #213
s_1 = {sp1_index,sp2_index,sp3_index}
s_2 = {sp1_index,sp3_index,sp4_index}
s_3 = {sp1_index,sp4_index,sp5_index}
s_4 = {sp1_index,sp5_index,sp6_index}
s_5 = {sp1_index,sp2_index,sp7_index}

tri_cal.simplices = np.array([item for item in tri_cal.simplices 
                                if (set(item) == s_1 or 
                                     set(item) == s_2 or 
                                     set(item) == s_3 or 
                                     set(item) == s_4 or 
                                     set(item) == s_5) == False])

s_year_global = 2010

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
    #facecolor=[map_z2color(zz,  colormap, min_zmean, max_zmean) for zz in zmean]
    facecolor=[map_z2color(zz,  colormap, tmin, tmax) for zz in zmean]

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
        myheatmap=go.Heatmap(
                    x=Xe,
                    y=Ye,
                    z=zmean,
                    colorscale='rainbow',
                    opacity=0,
                    zmin =tmin,
                    zmax=tmax,
                    colorbar=dict(title=dict(text='In °F')) 
        )
        return [triangles, lines, myheatmap]


noaxis=dict(showbackground=False,
            showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title=''
          )
layout = go.Layout(
         title='Triangulated California Weather Map',
         width=1200,
         height=1200,
         yaxis={'visible': False,'showticklabels':False},
         xaxis={'visible': False,'showticklabels':False},
         scene=dict(xaxis=noaxis,
                    yaxis=noaxis,
                    zaxis=noaxis,
                    aspectratio=dict(x=10, y=10, z=0),
                    camera=dict(eye=dict(x=0.001,
                                     y=-0.015,
                                     z= 1.4)
                            )
                )
        )


@cal.callback(
    Output('auto-stepper-year', 'disabled'),
    [Input('year-power-button', 'on'),]
)
def update_output(on):
    if (on == True):
        return False
    else:
        return True

@cal.callback(
    Output('auto-stepper-month', 'disabled'),
    [Input('month-power-button', 'on')]
)
def update_output(on):
    if (on == True):
        return False
    else:
        return True


@cal.callback(
    Output('year-slider-filter', 'value'),
    [Input('auto-stepper-year', 'n_intervals')]
)
def on_click(n_intervals):
    new_intervals = n_intervals%11
    if n_intervals is None:
        return 0
    else:
        return (new_intervals+2010)%2021


@cal.callback(
    Output('month-slider-filter', 'value'),
    [Input('auto-stepper-month', 'n_intervals')]
)
def on_click(n_intervals):
    if n_intervals is None:
        return 0
    else:
        return (n_intervals+1)%13


@cal.callback(
    Output(component_id='cal_3d_obj', component_property='figure'),
    [
        Input("weather-type-filter", "value"),
        Input("year-slider-filter", "value"),
        Input("month-slider-filter", "value"),
    ],


)

def update_graph(weather_type,s_year,s_month):
    z_cal_temp = (df_raw[df_raw["DATE"] == str(s_year)+str(months[s_month-1])])["TAVG"]
    #graph_data=plotly_trisurf_cal(x_cal_long,y_cal_lat, z_cal, z_cal_temp, tri_cal.simplices, colormap=cm.cubehelix, plot_edges=True)
    graph_data=plotly_trisurf_cal(x_cal_long,y_cal_lat, z_cal, z_cal_temp, tri_cal.simplices, colormap=cm.rainbow, plot_edges=True)
    fig = go.Figure(data=graph_data, layout=layout)
    fig.update_layout(title_text="Triangulated California Weather Map: " + str(s_year) + ", " + months_str[s_month-1])

    return fig

if __name__ == "__main__":
    cal.run_server(debug=True)