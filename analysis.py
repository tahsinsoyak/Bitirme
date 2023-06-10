# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 10:52:11 2023

@author: Tahsin
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Özellikler
features = ["acousticness", "danceability", "duration_ms", "energy", "instrumentalness", "speechiness", "valence", "danceability_energy_diff", "duration_energy_density"]

dataframe = pd.read_csv('Source/dataset.csv')
# Eksik veya hatalı verileri temizleme
dataframe.dropna(inplace=True)

import seaborn as sns
import matplotlib.pyplot as plt

sns.distplot(dataframe['popularity'])
plt.axvline(dataframe['popularity'].mean(), linewidth=2, color='r', label='Ortalama')
plt.axvline(dataframe['popularity'].median(), linewidth=2, color='k', label='Ortanca Değer')
plt.xticks(ticks=np.arange(0, 100, 5))
plt.legend()
plt.show()



plt.figure(figsize=(16, 8))
corr = dataframe.corr()
sns.heatmap(corr,annot=True)



color_list = ['#8ec6ff', 'red']
color_palette = sns.color_palette(color_list)
sns.set_palette(color_palette)
sns.set_style('darkgrid')


highly_popular_dance = dataframe[dataframe['popularity'] > 50]['danceability']
low_popular_dance = dataframe[dataframe['popularity'] <= 50]['danceability']



fig = plt.figure(figsize = (12, 8))
plt.title("Normalleştirilmiş Frekans vs Dans Edilebilirlik", fontsize=16)
#highly_popular_dance.hist(alpha = 1, bins = 15, label = 'High Popularity')
#low_popular_dance.hist(alpha = 0.5, bins = 15, label = 'Low Popularity')

sns.distplot(low_popular_dance, label = 'Düşük Populerlik', bins = 25)
sns.distplot(highly_popular_dance, label = 'Yüksek Populerlik', bins = 25)

plt.legend(loc = 'upper right', fontsize = 12)
plt.xlabel('Dansedilebilirlik', fontsize=18)
plt.ylabel('Normalleştirilmiş Frekans Yoğunluğu', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=12)

#Dansedilebilirlik değerinin 0.6 olması en iyi populerliğe işaret ediyor.

color_list = ['#8ec6ff', 'red']
sns.set_palette(color_list)
sns.set_style('darkgrid')
highly_popular_key = dataframe[dataframe['popularity'] > 50]['key']
low_popular_key = dataframe[dataframe['popularity'] <= 50]['key']

fig3 = plt.figure(figsize = (12, 8))
plt.title("Normalleştirilmiş Frekans vs Anahtar", fontsize = 16)


#low_popular_key.hist(alpha = 0.5, bins = 15, label = 'Low Popularity')
#highly_popular_key.hist(alpha = 0.5, bins = 15, label = 'High Popularity')

sns.distplot(low_popular_key, label = 'Düşük Populerlik', bins = 10)
sns.distplot(highly_popular_key, label = 'Yüksek Populerlik', bins = 10)

plt.legend(loc = 'upper right', fontsize = 12)
plt.xlabel('Anahtar', fontsize = 18)
plt.ylabel('Normalleştirilmiş Frekans', fontsize = 18)

#Uç noktalardaki 0 ​​ve 10 tuşlarının oldukça popüler değerler için nasıl daha iyi göründüğüne dikkat edin.


color_list = ['#8ec6ff', 'red']
color_palette = sns.color_palette(color_list)
sns.set_palette(color_palette)
sns.set_style('white')
fig4 = plt.figure(figsize = (12, 8))
plt.scatter(dataframe['speechiness'],dataframe['popularity'],s=1)
plt.title("Populerlik vs Konuşkanlık", fontsize = 16)
plt.xlabel('Konuşkanlık', fontsize = 16)
plt.ylabel('Popülerlik', fontsize = 16)

#Çok çok yüksek konuşkanlık hiçbir zaman çok popüler görünmüyor. Yaklaşık %10'luk konuşma oranı, popüler şarkılar için iyi bir miktar gibi görünüyor.

highly_popular_tempo = dataframe[dataframe['popularity'] > 50]['tempo']
low_popular_tempo = dataframe[dataframe['popularity'] <= 50]['tempo']
sns.set_style('darkgrid')
fig5 = plt.figure(figsize = (12, 8))
plt.title("Normalleştirilmiş Frekans vs Tempo", fontsize = 16)
sns.distplot(low_popular_tempo, label = 'Düşük Populerlik', bins = 15)
sns.distplot(highly_popular_tempo, label = 'Yüksek Populerlik', bins = 15)
plt.legend(loc = 'upper right', fontsize = 12)
plt.xlabel('Tempo', fontsize = 18)
plt.ylabel('Normalleştirilmiş Frekans', fontsize = 18)


#100-150 civarındaki bir tempo aralığının en popülerlik açısından iyi bir performans gösterdiği açıktır.

