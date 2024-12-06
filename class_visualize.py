import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
class Visualize:
    """
    Visualization refers to the process of representing data or information in a visual format such as charts,
    graphs, maps, and other visual aids. The goal of visualization is to make complex data more accessible, 
    understandable, and usable, often revealing patterns, trends, and insights that might not be immediately 
    apparent from raw data alone. 

    Types of Visualization:

    Charts and Graphs: Bar charts, line graphs, pie charts, histograms, etc.
    Maps: Geographical data visualization, heat maps.
    Diagrams: Flowcharts, network diagrams, tree diagrams.
    Interactive Visuals: Dashboards, interactive maps, and graphs that allow user interaction.

    Tools and Libraries:

    Python Libraries: Matplotlib, Seaborn, Plotly, Bokeh, and more.
    Software and Platforms: Tableau, Power BI, Google Data Studio.
    
    Input: DataFrame Server Log from class Transform
    Output: Graphs about frequency; Heat maps about correlation about features
    """
    def read():
        df = pd.read_csv(r"D:\code-python\project_practice\server_log\dataframe\data_after_preprocessing.csv")
        return df
df = Main.read()
df = df.drop(['Unnamed: 0'], axis =1)
#Visualization
df_filter = df [(df['status']>=400) & (df['status']<500)]
plt.figure(figsize=(10,6))
sns.histplot(df_filter['status'], bins =30, kde = True, color = 'skyblue')
plt.xlabel('Status')
plt.ylabel('Frequency')
plt.title('Frequency of Status between 400 and 500')
plt.grid()
plt.show()

plt.figure(figsize=(10,6))
sns.histplot(df_filter['bytes_sent'], bins =30, kde = True, color = 'white')
for p in plot.patches: 
    plot.annotate(format(p.get_height(), '.0f'), 
    (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', 
    va = 'center', xytext = (0, 10), textcoords = 'offset points')
plt.xlabel('Bytes Sent')
plt.ylabel('Frequency')
plt.title('Frequency of Bytes Sent Among Status between 400 and 500')
plt.grid()
plt.show()

#Visualization
df_filter1 = df [df['status']>=500]
plt.figure(figsize=(10,6))
sns.histplot(df_filter1['status'], bins =30, kde = True, color = 'red')
plt.xlabel('Status')
plt.ylabel('Frequency')
plt.title('Frequency of Status equal and greater than 500')
plt.grid()
plt.show()

