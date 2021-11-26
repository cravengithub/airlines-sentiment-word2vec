import pandas as pd
from os import system
import numpy as np
from matplotlib import pyplot as plot

system('clear')
file = './dataset/Tweets.csv'
data_frame = pd.read_csv(file)
label = data_frame.get('airline_sentiment')
res = label.value_counts()
label = 'positive', 'negative', 'neutral'
sizes = res['positive'], res['negative'], res['neutral']
fig, ax = plot.subplots()
ax.pie(sizes, labels=label, shadow=True, startangle=90,
       autopct=lambda p: '{:.2f}%  ({:,.0f})'.format(p, p * sum(sizes)/100))
# ax.axis('equal')
plot.show()
