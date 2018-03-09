

```python
import plotly.plotly as py
import pandas as pd
from config import *
mapbox_access_token = api_key
```


```python
# df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2011_us_ag_exports.csv')
df = pd.read_csv("Resources/crimes.csv")

```


```python
for col in df.columns:
    df[col] = df[col].astype(str)
```


```python
scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]
```


```python
df['text'] = df['state_abbr'] + '<br>' +\
    'year '+df['year']+' robbery rate '+df['robbery rate']
```


```python
data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = True,
        locations = df['state_abbr'],
        z = df['robbery rate'].astype(float),
        locationmode = 'USA-states',
        text = df['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            )
        ),
        colorbar = dict(
            title = "Robbery-rate"
        )
    ) ]
```


```python
layout = dict(
        title = 'Crime rate over the years in USA',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'aqua',
        ),
    )
```


```python
fig = dict( data=data, layout=layout )
```


```python
url = py.iplot( fig, filename='Crime_rate-map' )
url
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~anjvenus/4.embed" height="525px" width="100%"></iframe>




```python
py.image.save_as(fig, filename='Crime_rate-map.png')
```
