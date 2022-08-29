#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 15:34:51 2022

@author: Anni Zheng
"""

#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LogisticRegression
from scipy.special import expit # this is the logistic sigmoid function
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import linear_model
import statsmodels.api as sm
from scipy.stats import f_oneway
from scipy.stats import bootstrap

#%%

data = np.genfromtxt('movieReplicationSet.csv', delimiter=',', skip_header=1)
print(data.shape)

#%%

data_pd = pd.read_csv('movieReplicationSet.csv')
titles = data_pd.columns

#%%

# 1. What is the relationship between sensation seeking and movie experience?

# row-wise reduction of nans

ss = data[:, 400:420] #1097, 20
me = data[:, 464:474] #1097, 10
ss_me = np.concatenate((ss, me), axis=1) #1097, 30
ss_me = ss_me[~np.isnan(ss_me).any(axis=1), :] #1029, 30

ss = ss_me[:, :20]
me = ss_me[:, 20:]

print(ss.shape)
print(me.shape)

#%%

# PCA dimension reduction of SS

ss_zscored = stats.zscore(ss)
ss_pca = PCA().fit(ss_zscored)
ss_eigenvals = ss_pca.explained_variance_
ss_loadings = ss_pca.components_
ss_rotated = ss_pca.fit_transform(ss_zscored)

x = np.linspace(1, 20, 20)
plt.title('Eigenvalues for Seeking Sensation')
plt.bar(x, ss_eigenvals)
plt.plot([0, 20], [1,1],color='orange')
plt.xlabel('Principal Components')
plt.ylabel('Eigenvalues')
plt.show()

ss_varExplained = ss_eigenvals/sum(ss_eigenvals)*100

#%%

# PCA dimension reduction of ME

me_zscored = stats.zscore(me)
me_pca = PCA().fit(me_zscored)
me_eigenvals = me_pca.explained_variance_
me_loadings = me_pca.components_
me_rotated = me_pca.fit_transform(me_zscored)

x = np.linspace(1, 10, 10)
plt.title('Eigenvalues Movie Experience')
plt.bar(x, me_eigenvals)
plt.plot([0, 10], [1,1],color='orange')
plt.xlabel('Principal Components')
plt.ylabel('Eigenvalues')
plt.show()

me_varExplained = me_eigenvals/sum(me_eigenvals)*100

#%%

# I would go with the Kaiser One

kaiserThreshold = 1

print('Number of factors selected by Kaiser criterion:', 
      np.count_nonzero(ss_eigenvals > kaiserThreshold))

print('Number of factors selected by Kaiser criterion:', 
      np.count_nonzero(me_eigenvals > kaiserThreshold))

# ss kaiser: 6
# me kaiser: 2

threshold = 90

ss_eigSum = np.cumsum(ss_varExplained)
print('Number of factors to account for at least 90% variance:', np.count_nonzero(ss_eigSum < threshold) + 1)

me_eigSum = np.cumsum(me_varExplained)
print('Number of factors to account for at least 90% variance:', np.count_nonzero(me_eigSum < threshold) + 1)

# ss kaiser: 16
# me kaiser: 8

#%%

# transform ss and me data into lower dimension data

ss_compressed = ss_rotated[:, :6]
me_compressed = ss_rotated[:, :2]

print(ss_compressed.shape) # 1029, 6
print(me_compressed.shape) # 1029, 2

#%%

# SS Loadings

# print(ss_loadings[whichPrincipalComponent,:] * -1)
temp_ss_loadings = []

for pc in range(6):
    temp_ss_loadings.append(ss_loadings[pc,:] * -1)

# use standard deviation to choose the best pc
for t in temp_ss_loadings:
    ## print(np.std(t))
    print(np.var(t))

print(np.sort(temp_ss_loadings[2])) # pc# = 3

# SS Loadings in Graph

whichPrincipalComponent = 2
x = np.linspace(1,20,20)
plt.bar(x,ss_loadings[whichPrincipalComponent,:] * -1) # note: eigVecs multiplied by -1 because the direction is arbitrary
plt.xlabel('Questions')
plt.ylabel('Loadings')
plt.title('Loadings of Seeking Sensation (PC = 3)')
plt.show()

# Q: 6, 10

#%%

whichPrincipalComponent = 0
x = np.linspace(1,20,20)
plt.bar(x,ss_loadings[whichPrincipalComponent,:] * -1) # note: eigVecs multiplied by -1 because the direction is arbitrary
plt.xlabel('Questions')
plt.ylabel('Loadings')
plt.title('Loadings of Seeking Sensation (PC = 1)')
plt.show()

#%%

# ME Loadings

whichPrincipalComponent = 0
x = np.linspace(1,10,10)
plt.bar(x,me_loadings[whichPrincipalComponent,:] * -1)
plt.xlabel('Questions')
plt.ylabel('Loadings')
plt.title('Loadings of Movie Experience (PC = 1)')
plt.show()

# One will Use the 2nd PC here

# Q: 2

#%%

# Multiple Regression (Multiple Linear Regression)

ss_space = np.stack([ss[:,5], ss[:,9], ss[:,13]], axis=1)
me_space = me[:,2]

print(ss_space.shape, me_space.shape)

regr = linear_model.LinearRegression()
regr.fit(ss_space, me_space)

x = sm.add_constant(ss_space) # adding a constant
model = sm.OLS(me_space, ss_space).fit()
predictions = model.predict(ss_space) 
print_model = model.summary()
print(print_model)

# R-squared = 0.733

print(np.sqrt(0.733)) # 0.856

#%%

# specific questions asked

print('sensation seeking')
print(titles[405]) 
print(titles[408])
print(titles[412])
print()
print('movie experience')
print(titles[466])

#%%

# 2. Is there evidence of personality types based on the data of these research participants? If so, characterize these types both quantitatively and narratively
# Result: 2 Clusters

# Personality type

p = data[:,420:464]
p = p[~np.isnan(p).any(axis=1), :]
print(p.shape)

p_zscored = stats.zscore(p)
p_pca = PCA().fit(p_zscored)
p_eigenvals = p_pca.explained_variance_
p_loadings = p_pca.components_
p_rotated = p_pca.fit_transform(p_zscored)

x = np.linspace(1, 44, 44)
plt.title('Eigenvalues for Personality')
plt.bar(x, p_eigenvals)
plt.plot([0, 44], [1,1],color='orange')
plt.xlabel('Principal Components')
plt.ylabel('Eigenvalues')
plt.show()

p_varExplained = p_eigenvals/sum(p_eigenvals)*100

kaiserThreshold = 1

print('Number of factors selected by Kaiser criterion:', 
      np.count_nonzero(p_eigenvals > kaiserThreshold)) # 8

temp_p_loadings = []
for pc in range(8):
    temp_p_loadings.append(p_loadings[pc,:] * -1)
# again, use standard deviation to choose the best pc

for t in temp_p_loadings:
    print(np.var(t))

# print(np.sort(temp_p_loadings[#])) # pc_num = 3

# the 8th pc
whichPrincipalComponent = 7
x = np.linspace(1,44,44)
plt.bar(x, p_loadings[whichPrincipalComponent,:] * -1)
plt.xlabel('Questions')
plt.ylabel('Loadings')
plt.title('Loadings of Personality (PC = 8)')
plt.show()

# the 1st pc
whichPrincipalComponent = 0
plt.bar(x, p_loadings[whichPrincipalComponent,:] * -1)
plt.title('Loadings of Personality (PC = 1)')
plt.show()

# compare between two graphs, decide to choose the first pc

x_temp=[]
for i in range(len(p_loadings[0])):
    x_temp.append(0)

fig, ax = plt.subplots()
ax.set_xticklabels(["Question"])
plt.scatter(x_temp, p_loadings[0] * -1)
plt.title('Loadings of Personality (PC = 1) in Scattered Graph')
plt.ylabel('Loadings')
plt.xlabel('Question')
plt.show()

#%%

# DBSCAN Cluster

p_loadings[0] = p_loadings[0] * -1
    
x_for_clustering = np.stack((p_loadings[0], np.array(x_temp)))
x_for_clustering = np.transpose(x_for_clustering)
print(x_for_clustering.shape)
dbscanModel = DBSCAN(eps=0.1, min_samples=3).fit(x_for_clustering)
labels = dbscanModel.labels_
print(labels)

a = []
b = []

for i in range(len(labels)):
    if (labels[i] == 1):
        a.append(p_loadings[0][i])
    elif(labels[i] == 0):
        b.append(p_loadings[0][i])

ax_temp = []
bx_temp = []

for i in range(len(labels)):
    if (labels[i] == 0):
        bx_temp.append(i)
    elif(labels[i] == 1):
        ax_temp.append(i)

plt.scatter(ax_temp, a, color='green')
plt.scatter(bx_temp, b, color='purple')
plt.title('Clusters of Personalities')
plt.ylabel('Loadings')
plt.xlabel('Question Number')
plt.show()

### Another Graph
        
ax_temp = []
bx_temp = []
for i in range(len(a)):
    ax_temp.append(0)
for i in range(len(b)):
    bx_temp.append(0)
    
fig, ax = plt.subplots()
ax.set_xticklabels(["Question"])
plt.scatter(ax_temp, a, color='green')
plt.scatter(bx_temp, b, color='purple')
plt.title('Clusters of Personalities')
plt.ylabel('Loadings')
plt.xlabel('Question')
plt.show()

# Characterize Clusters (Narratively)

#%%

# 3. Are movies that are more popular rated higher than movies that are less popular?
# Result: Positive Relationship

movies = data[:, 0:400]

# loop through 400 ratings to get a sense of popularity

first = movies[:, 0][~np.isnan(movies[:, 0])]

print(len(first))

pp = []

for i in range(400):
    temp = movies[:, i][~np.isnan(movies[:, i])] # element-wise reduction
    pp.append(len(temp))

pp_median = np.median(pp)
pp_mean = np.mean(pp)

print(pp_median) # popularity 197.5

x = np.array(range(400))
plt.scatter(x, pp)
plt.plot([0, 400], [pp_mean, pp_mean],color='purple') # mean
plt.plot([0, 400], [pp_median, pp_median],color='green') # median
plt.title("Popularities")
plt.xlabel("Movies")
plt.ylabel("Popularities")
plt.show()

# using median is more reasonable since there are more movies with low popularities than that of high
# popularities, so the mean is dragged down by those movies with low popularities

pp_scores = []

for i in range(400):
    temp = movies[:, i][~np.isnan(movies[:, i])] # element-wise reduction
    # however, mean is used to determine score
    # since mean accounts for the whole population regardless if a viewer likes to dislikes a movie
    # which might be a better reflection of a movie than median which is scored by only one viewer
    pp_scores.append(np.mean(temp))

print(np.median(pp_scores)) # 2.582

plt.scatter(pp, pp_scores)
plt.title("Relationship Between Popularities and Scores")
plt.xlabel("Popularities")
plt.ylabel("Scores")
plt.show()

pp = np.array(pp)
pp_scores = np.array(pp_scores)

pp_ppscores_cor = np.corrcoef(pp, pp_scores)
print(pp_ppscores_cor) # 0.699 - positive relationship

# Do Logistic Regression Here

# Populartiy > 197.5 would consider popular
# Popularity < 197.5 would consider not popular
# ignore those equal to median, since poopulation cannot have decimal

pp_logistic_highpp = []
pp_logistic_lowpp = []

for i in range(len(pp)):
    if (pp[i] > 197.5):
        pp_logistic_highpp.append(pp_scores[i])
    elif (pp[i] < 197.5):
        pp_logistic_lowpp.append(pp_scores[i])

y_highpp = []
y_lowpp = []

for i in range(len(pp_logistic_highpp)):
    y_highpp.append(1)
for i in range(len(pp_logistic_lowpp)):
    y_lowpp.append(0)

'''
plt.scatter(pp_logistic_highpp, y_highpp, color='green')
plt.scatter(pp_logistic_lowpp, y_lowpp, color='purple')
plt.title('Logistic Relationship between Popularities and Scores')
plt.xlabel('Scores')
plt.ylabel('Popularities')
plt.show()
'''

# Logistic Regression

import seaborn as sns

x = np.concatenate((pp_logistic_highpp, pp_logistic_lowpp))
y = np.concatenate((y_highpp, y_lowpp))
temp = np.transpose(np.stack((x, y)))
logic = sns.regplot(x=x, y=y, data=temp, logistic=True, ci=None, line_kws={'color': 'orange'}).set(title='Logistic Relationship between Popularities and Scores', xlabel = 'Scores', ylabel = 'Popularities')

#%%

# 4. Is enjoyment of ‘Shrek (2001)’ gendered, i.e. do male and female viewers rate it differently?
# Result: No Connection

index = 0

# find title
for i in range(len(titles)):
    if (titles[i] == 'Shrek (2001)'):
        index = i
        break

print(index) # 87

shrek = data[:, 87]
gender = data[:, 474]
gender = gender[~np.isnan(shrek)] # remove gender identity if score is nan
shrek = shrek[~np.isnan(shrek)] # remove nans

# separate bewteen three groups (female, male, self-described)

female = []
male = []
self = []

for i in range(len(gender)):
    if (gender[i] == 1):
        female.append(shrek[i])
    elif (gender[i] == 2):
        male.append(shrek[i])
    else:
        self.append(shrek[i])

# examine actual difference

# decide to use mean, because, again, mean represents a whole population

f_mean = np.mean(female)
m_mean = np.mean(male)
s_mean = np.mean(self)

print(f_mean, m_mean, s_mean) # 3.155 3.083 2.979

# the mean difference is not large, but the sample

print(len(female), len(male), len(self)) # 743 241 24
# size of self-described people are way too small

# so, one may use ANOVA and T-test as an extra confirmation of the existence of difference

# Null hypothesis: no connection

# ANOVA
res = f_oneway(female, male, self)
print(res) # p-value = 0.376

# T-test (for two groups' population)
print(stats.ttest_ind(female, male)) # p-value = 0.271
print(stats.ttest_ind(female, self)) # p-value = 0.349
print(stats.ttest_ind(male, self)) # p-value = 0.562

# all p-values are greater than 0.05
# the p-values are too large that one can't reject null hypothesis, 
# so, there is no connection based on the data

'''
# Since, again, the ANOVA and T-Test do not account for 
# the fact that the population between self-described people's gXroup are significantly
# different than that of the other groups
# One may use Bootstrapping several times (10000 times in this case) to model

female_seq = (female[:],)
male_seq = (male[:],)
self_seq = (self[:],)

f_bootstrapCI = bootstrap(female_seq, np.mean, n_resamples = 1e4, confidence_level=0.95) 
m_bootstrapCI = bootstrap(male_seq, np.mean, n_resamples = 1e4, confidence_level=0.95) 
s_bootstrapCI = bootstrap(self_seq, np.mean, n_resamples = 1e4, confidence_level=0.95) 

print(f_bootstrapCI) # High: 3.217; Low: 3.087; SEM: 0.033
print(m_bootstrapCI) # High: 3.180; Low: 2.973; SEM: 0.053
print(s_bootstrapCI) # High: 3.271; Low: 2.521; SEM: 0.187

print("High Difference: ", 3.271 - 3.180) # 0.091
print("Low Difference: ", 3.087 - 2.521) # 0.566

# Still, no difference
# But notice that self-described viewers' data is hacing a higher standard error
# this means that sample means are widely spread around the population mean
# which also means that the average scores estimated by the self-described people's data
# is not as representative as that of female viewers' or male viewers'.
'''

#%%

# 5. Do people who are only children enjoy ‘The Lion King (1994)’ more than people with siblings?
# Result: Yes

# : Column 476: Only child (1 = yes, 0 = no, -1 = no response)
only_child = data[:,475]

# find title
index = 0
for i in range(len(titles)):
    if (titles[i] == 'The Lion King (1994)'):
        index = i
        break
print(index) # 220

lion_king = data[:,220]

# remove nans
only_child = only_child[~np.isnan(lion_king)] # 937
lion_king = lion_king[~np.isnan(lion_king)] # 937

print(len(lion_king))

with_sibling = []
without_sibling = []

for i in range(len(only_child)):
    if (only_child[i] == 1): 
        without_sibling.append(lion_king[i])
    elif (only_child[i] == 0):
        with_sibling.append(lion_king[i])
    # note that one does not need to account for those who did not report

print("with sibling: ", len(with_sibling)) # 776
print("without sibling: ", len(without_sibling)) # 151

print ("gap: ", len(with_sibling) - len(without_sibling)) # BIG gap: 625
print("mean gap: ", np.mean(with_sibling) - np.mean(without_sibling)) # small gap: 0.134

# very similar mean --> do t test

# Null Hypothesis: No difference

# so, one may use t-test to examine if there is a difference between two groups (assume CLT)
print(stats.ttest_ind(with_sibling, without_sibling)) # p-value = 0.0403
# p-value < 0.05 there is a difference, though the means are very close

# Plot to Examine the Difference

fig, ax = plt.subplots()
ax.set_xticklabels(["Only Child", "Not Only Child"])
plt.boxplot([without_sibling, with_sibling])
plt.xlabel("Viewers")
plt.ylabel("Scores")
plt.title("Score Difference between Only Child and Not Only Child Viewers")
plt.show()

# Yes, those with siblings favor The Lion King (1994) more

# To put that into logistic regression model
# Let median of scores differentiate those who are liking the movie and those who aren't liking the movie.

#%%

# 6. Do people who like to watch movies socially enjoy ‘The Wolf of Wall Street (2013)’ 
#    more than those who prefer to watch them alone?
# Result: No

# : Column 477: Social viewing preference - alone? 1 = y, 0 = n, -1 = nr)
preference = data[:,476]

# find title
index = 0
for i in range(len(titles)):
    if (titles[i] == 'The Wolf of Wall Street (2013)'):
        index = i
        break
print(index) # 357

wolf = data[:, 357]

# remove nans

preference = preference[~np.isnan(wolf)]
wolf = wolf[~np.isnan(wolf)] # 667

# one may do the same step as those indicated in question 5
# first find if a connection exists

alone = []
not_alone = []

for i in range(len(preference)):
    if (preference[i] == 1):
        alone.append(wolf[i])
    elif (preference[i] == 0):
        not_alone.append(wolf[i])

print(len(alone)) # 393
print(len(not_alone)) # 270

# Null hypothesis: no connection

# T-test
print(stats.ttest_ind(alone, not_alone)) # p-value = 0.117

# can't reject null hypothesis
