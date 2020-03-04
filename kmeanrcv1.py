# -*- coding: utf-8 -*-
#  Implementation of sparse matrix on k-means  
#  for RCV1 Dataset
#  written by Happy Sisodia
#  kmeanmrcv1.py

from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import tensorflow.examples.tutorials.mnist.input_data as input_data
import time
import numpy as np
from sklearn.decomposition import PCA
from sklearn import metrics
from math import floor
import sklearn.preprocessing

# data path
fpath = "C:\\Users\\Happy\\Desktop\\Computer Science\\Machine learning\\Kmeansparsifcation\\code\\datasets\\RCV1\\"

# how many data are used to cluster
train_num = 15000
sample_size = 300
p = 0.05

# read data
RCV1 = input_data.read_data_sets(fpath)
data = RCV1.train.images[0 : train_num]              # X
labels = RCV1.train.labels[0 : train_num]            # Y

wcss = []

#pca = TruncatedSVD(n_components = 100)
#trmatrix = pca.fit_transform(data)

#print("Begin clustering on raw data...")
#print("Data shape = ", data.shape)
#start = time.time()
#for i in range(50,55):
#   kmeans = KMeans(n_clusters = i,init='k-means++',max_iter=300,n_init=10,random_state=0)
#   kmeans.fit(data)
#   wcss.append(kmeans.inertia_)
#end = time.time()
#print("Clustering on raw data, using time = ", end - start)
#plt.plot(range(50,55),wcss)
#plt.title('the elbow method')
#plt.xlabel('Number of clusters')
#plt.ylabel('WCSS')
#plt.show()

#can be seen from the elbow diagram that the K=53 is best
#performing k-means for 53 cluster center         

print(82 * '_') 

n_samples, n_features = data.shape

print("\t n_samples %d, \t n_features %d"
      % ( n_samples, n_features))

print(82 * '_')
print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')

#clustering on raw data
t0 = time.time()
kmeans = KMeans(init='k-means++', n_clusters=53, n_init=10)
kmeans.fit(data)
print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % ('k-means++', (time.time() - t0), kmeans.inertia_,
             metrics.homogeneity_score(labels, kmeans.labels_),
             metrics.completeness_score(labels, kmeans.labels_),
             metrics.v_measure_score(labels, kmeans.labels_),
             metrics.adjusted_rand_score(labels, kmeans.labels_),
             metrics.adjusted_mutual_info_score(labels,  kmeans.labels_,
                                                average_method='arithmetic'),
             metrics.silhouette_score(data, kmeans.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))

t0 = time.time()
kmeans = KMeans(init='random', n_clusters=53, n_init=10)
kmeans.fit(data)
print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % ('random', (time.time() - t0), kmeans.inertia_,
             metrics.homogeneity_score(labels, kmeans.labels_),
             metrics.completeness_score(labels, kmeans.labels_),
             metrics.v_measure_score(labels, kmeans.labels_),
             metrics.adjusted_rand_score(labels, kmeans.labels_),
             metrics.adjusted_mutual_info_score(labels,  kmeans.labels_,
                                                average_method='arithmetic'),
             metrics.silhouette_score(data, kmeans.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))   
             
def _get_mean(sums, step):
    return sums/step

#Scale dataset
minmax_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled = minmax_scaler.fit_transform(data)        

n = np.shape(data)[1]
m = np.shape(data)[0]
centroids = np.mat(np.zeros((10,n)))

# Sum all elements of each row, add as col to original dataset, sort
composite = np.mat(np.sum(data, axis=1))
ds = np.append(composite.T, data, axis=1)
ds.sort(axis=0)

# Step value for dataset sharding
step = floor(m/10)

# Vectorize mean ufunc for numpy array
vfunc = np.vectorize(_get_mean)

# Divide matrix rows equally by k-1 (so that there are k matrix shards)
# Sum columns of shards, get means; these columnar means are centroids
K= 10
for j in range(10):
    if j == K-1:
        centroids[j:] = vfunc(np.sum(ds[j*step:,1:], axis=0), step)
    else:
        centroids[j:] = vfunc(np.sum(ds[j*step:(j+1)*step,1:], axis=0), step) 
 

centroids_scaled = centroids     

t0 = time.time()
kmeans = KMeans(init=centroids_scaled, n_clusters=10, n_init =1)
kmeans.fit(data)
print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
      % ('Shards', (time.time() - t0), kmeans.inertia_,
         metrics.homogeneity_score(labels, kmeans.labels_),
         metrics.completeness_score(labels, kmeans.labels_),
         metrics.v_measure_score(labels, kmeans.labels_),
         metrics.adjusted_rand_score(labels, kmeans.labels_),
         metrics.adjusted_mutual_info_score(labels,  kmeans.labels_,
                                            average_method='arithmetic'),
         metrics.silhouette_score(data, kmeans.labels_,
                                  metric='euclidean',
                                  sample_size=sample_size)))        

print(82 * '_') 

# #############################################################################
# Visualize the results on PCA-reduced data
             
reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(init='k-means++', n_clusters=53, n_init=10)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()  

print(82 * '_')

# data dimension
raw_dim =  n_features                 # raw dimension change ?
low_dim = 100                      # random projection to low-dimension

### random_projection matrix
rj_matrix = 1.0 - 2.0 * (np.random.rand(raw_dim, low_dim) > 0.5)
rj_matrix = rj_matrix / np.sqrt(low_dim)
print(np.sum(rj_matrix), np.max(rj_matrix), np.min(rj_matrix))

print("Begin clustering on sparsed data...")
print("Data shape = ", data.shape)
    
print("First random projection...")
start = time.time()
rj_data = np.dot(data, rj_matrix)
end = time.time()
print("Random projection time = ", end - start)
    
print("Second random sparsification...")
start = time.time()
# construct random sparsification matrix
n = rj_data.shape[0]                      # the number of data points    
max_v = np.max(np.abs(rj_data))           # max value
tau = p * ((rj_data / max_v) ** 2)        # tau_ij
prob = np.zeros_like(tau)                 # sparsification probability
sqrt_tau = 64 * np.sqrt(tau / n) * np.log(n) * np.log(n)
prob[tau > sqrt_tau] = tau[tau > sqrt_tau]
prob[tau <= sqrt_tau] = sqrt_tau[tau <= sqrt_tau]

sparse_map = np.random.rand(rj_data.shape[0], rj_data.shape[1]) <= prob

 # sparsification
rs_data = rj_data.copy()
index = (prob != 0.0) & (sparse_map == 1.0)         
rs_data[index] = rs_data[index] / prob[index]         # data[i][j]/prob[i][j] 
rs_data[sparse_map == 0.0] = 0.0                      # data[i][j] = 0.0
    
end = time.time()
print("Random projection time = ", end - start)

print("Before sparsification, the number of zero-elements is:", np.sum(rj_data == 0.0)/(rj_data.shape[0] * rj_data.shape[1]))
print("After sparsification, the number of zero-elements is:", np.sum(rs_data == 0.0)/(rs_data.shape[0] * rs_data.shape[1]))

n_samples, n_features = rs_data.shape
print(82 * '_') 
print("\t n_samples %d, \t n_features %d"
      % ( n_samples, n_features))    
print(82 * '_')
print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')

#clustering on sparse data
t1 = time.time()
kmeans = KMeans(init='k-means++', n_clusters=53, n_init=10)
kmeans.fit(rs_data)
print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % ('k-means++', (time.time() - t1), kmeans.inertia_,
             metrics.homogeneity_score(labels, kmeans.labels_),
             metrics.completeness_score(labels, kmeans.labels_),
             metrics.v_measure_score(labels, kmeans.labels_),
             metrics.adjusted_rand_score(labels, kmeans.labels_),
             metrics.adjusted_mutual_info_score(labels,  kmeans.labels_,
                                                average_method='arithmetic'),
             metrics.silhouette_score(data, kmeans.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))


t1 = time.time()
kmeans = KMeans(init='random', n_clusters=53, n_init=10)
kmeans.fit(rs_data)
print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % ('random', (time.time() - t1), kmeans.inertia_,
             metrics.homogeneity_score(labels, kmeans.labels_),
             metrics.completeness_score(labels, kmeans.labels_),
             metrics.v_measure_score(labels, kmeans.labels_),
             metrics.adjusted_rand_score(labels, kmeans.labels_),
             metrics.adjusted_mutual_info_score(labels,  kmeans.labels_,
                                                average_method='arithmetic'),
             metrics.silhouette_score(data, kmeans.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))
             

#Scale dataset
minmax_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled = minmax_scaler.fit_transform(rs_data)        

n = np.shape(rs_data)[1]
m = np.shape(rs_data)[0]
centroids = np.mat(np.zeros((10,n)))

# Sum all elements of each row, add as col to original dataset, sort
composite = np.mat(np.sum(rs_data, axis=1))
ds = np.append(composite.T, rs_data, axis=1)
ds.sort(axis=0)

# Step value for dataset sharding
step = floor(m/10)

# Vectorize mean ufunc for numpy array
vfunc = np.vectorize(_get_mean)

# Divide matrix rows equally by k-1 (so that there are k matrix shards)
# Sum columns of shards, get means; these columnar means are centroids
K= 10
for j in range(10):
    if j == K-1:
        centroids[j:] = vfunc(np.sum(ds[j*step:,1:], axis=0), step)
    else:
        centroids[j:] = vfunc(np.sum(ds[j*step:(j+1)*step,1:], axis=0), step) 
 

centroids_scaled = centroids

t1 = time.time()
kmeans = KMeans(init=centroids_scaled, n_clusters=10, n_init =1)
kmeans.fit(rs_data)
print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
      % ('Shards', (time.time() - t1), kmeans.inertia_,
         metrics.homogeneity_score(labels, kmeans.labels_),
         metrics.completeness_score(labels, kmeans.labels_),
         metrics.v_measure_score(labels, kmeans.labels_),
         metrics.adjusted_rand_score(labels, kmeans.labels_),
         metrics.adjusted_mutual_info_score(labels,  kmeans.labels_,
                                            average_method='arithmetic'),
         metrics.silhouette_score(data, kmeans.labels_,
                                  metric='euclidean',
                                  sample_size=sample_size)))             

print(82 * '_') 
# #############################################################################
# Visualize the results on PCA-reduced data
             
reduced_data = PCA(n_components=2).fit_transform(rs_data)
kmeans = KMeans(init='k-means++', n_clusters=53, n_init=10)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()  

           