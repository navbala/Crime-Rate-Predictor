

```python
import pandas as pd
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib as mpl

import matplotlib.pyplot as plt
%matplotlib inline
```


```python
crimes_raw = pd.read_csv("Resources/crimes.csv")
```


```python
crimes_raw["state_abbr"] = pd.Categorical(crimes_raw["state_abbr"], crimes_raw.state_abbr.unique())
crimes_raw.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state_abbr</th>
      <th>year</th>
      <th>robbery rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AK</td>
      <td>1995</td>
      <td>155.132450</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AK</td>
      <td>1996</td>
      <td>116.968699</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AK</td>
      <td>1997</td>
      <td>106.403941</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AK</td>
      <td>1998</td>
      <td>86.644951</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AK</td>
      <td>1999</td>
      <td>91.364003</td>
    </tr>
  </tbody>
</table>
</div>




```python
crime_matrix = crimes_raw.pivot("state_abbr", "year", "robbery rate")
crime_matrix.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>year</th>
      <th>1995</th>
      <th>1996</th>
      <th>1997</th>
      <th>1998</th>
      <th>1999</th>
      <th>2000</th>
      <th>2001</th>
      <th>2002</th>
      <th>2003</th>
      <th>2004</th>
      <th>...</th>
      <th>2007</th>
      <th>2008</th>
      <th>2009</th>
      <th>2010</th>
      <th>2011</th>
      <th>2012</th>
      <th>2013</th>
      <th>2014</th>
      <th>2015</th>
      <th>2016</th>
    </tr>
    <tr>
      <th>state_abbr</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AK</th>
      <td>155.132450</td>
      <td>116.968699</td>
      <td>106.403941</td>
      <td>86.644951</td>
      <td>91.364003</td>
      <td>78.158397</td>
      <td>81.119896</td>
      <td>76.229731</td>
      <td>68.797433</td>
      <td>67.958434</td>
      <td>...</td>
      <td>85.006394</td>
      <td>93.983182</td>
      <td>93.632825</td>
      <td>83.176269</td>
      <td>79.573398</td>
      <td>86.265091</td>
      <td>84.502190</td>
      <td>85.340671</td>
      <td>103.157207</td>
      <td>114.571623</td>
    </tr>
    <tr>
      <th>AL</th>
      <td>185.751234</td>
      <td>166.721273</td>
      <td>160.476962</td>
      <td>130.928309</td>
      <td>121.216643</td>
      <td>128.218390</td>
      <td>124.952114</td>
      <td>133.113160</td>
      <td>134.066770</td>
      <td>133.513797</td>
      <td>...</td>
      <td>159.858215</td>
      <td>157.575238</td>
      <td>133.051359</td>
      <td>101.642475</td>
      <td>102.129842</td>
      <td>104.202819</td>
      <td>96.090274</td>
      <td>97.020249</td>
      <td>95.016868</td>
      <td>96.354327</td>
    </tr>
    <tr>
      <th>AR</th>
      <td>125.684380</td>
      <td>114.103586</td>
      <td>111.533888</td>
      <td>96.217494</td>
      <td>79.329835</td>
      <td>74.848508</td>
      <td>80.936714</td>
      <td>93.264969</td>
      <td>81.641661</td>
      <td>86.254545</td>
      <td>...</td>
      <td>109.531653</td>
      <td>97.254666</td>
      <td>89.151915</td>
      <td>81.086040</td>
      <td>80.208754</td>
      <td>78.309651</td>
      <td>76.417019</td>
      <td>68.659026</td>
      <td>71.091488</td>
      <td>70.944580</td>
    </tr>
    <tr>
      <th>AZ</th>
      <td>173.755334</td>
      <td>167.773261</td>
      <td>165.686059</td>
      <td>165.238809</td>
      <td>152.521842</td>
      <td>146.258784</td>
      <td>167.101127</td>
      <td>147.028418</td>
      <td>136.560259</td>
      <td>134.515031</td>
      <td>...</td>
      <td>154.036558</td>
      <td>150.903513</td>
      <td>124.306791</td>
      <td>108.417725</td>
      <td>110.478614</td>
      <td>112.697788</td>
      <td>100.316549</td>
      <td>92.513015</td>
      <td>93.288439</td>
      <td>101.788021</td>
    </tr>
    <tr>
      <th>CA</th>
      <td>331.162747</td>
      <td>295.570613</td>
      <td>252.473038</td>
      <td>210.554994</td>
      <td>181.139782</td>
      <td>177.874428</td>
      <td>186.743166</td>
      <td>185.612325</td>
      <td>179.822682</td>
      <td>172.333950</td>
      <td>...</td>
      <td>193.433054</td>
      <td>188.776643</td>
      <td>173.403990</td>
      <td>155.647576</td>
      <td>144.069357</td>
      <td>148.739951</td>
      <td>139.573395</td>
      <td>125.491428</td>
      <td>135.564654</td>
      <td>139.589748</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python
crimes_rows = crimes_raw
```


```python
import seaborn as sns
fig = plt.figure(figsize=(12,12))
r = sns.heatmap(crime_matrix, cmap='BuPu')
r.set_title("Heatmap of crime rate from 1995 to 2016")
```




    Text(0.5,1,'Heatmap of crime rate from 1995 to 2016')




![png](output_5_1.png)



```python
plt.savefig("heatmap_year.png")
plt.show()
```


    <matplotlib.figure.Figure at 0x1a18051ac8>

