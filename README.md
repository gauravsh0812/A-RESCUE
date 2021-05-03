# A-RESCUE(Adaptable Retention STT-RAM Caches for User Experience Optimization) 

A Machine Learning-based model that used a single _Decision Tree Classifier_ to predict the best retention time based on applications' characteristics for mobile architecture (ARM V7 Cortex) and models of user behavior and experience expectations.

The model is able to predict both _latency_ and _energy_ and reduced the model's memory overhead by 20x compared to prior work **_SCART_**: _Predicting STT-RAM Cache Retention Times Using Machine Learning_**. The model predicts the best retention time for both the _data_ and _instruction_ _cache_. The model was compared with data and instruction cache latency and energy of STT-RAM with the best homogeneous retention time(1ms for the data cache and 1s for instruction cache). 

The model reduced the average data cache access latency and energy by 16.09% and 18.20%, respectively while, instruction cache access latency and energy by 7.86% and 6.64%, respectively. The model achieved an accuracy of 82% and 64% for instruction and data cache respectively, with the deterioration of only 2.5% from the Exhaustive Search

# Prerequisite 

Python 3

GEM5 simulator

sklearn

pandas

NumPy

openpyxl (to get excel sheets containing final tables)
