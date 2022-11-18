import seaborn
import numpy
import pandas
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.preprocessing import StandardScaler




class Recommender:
  def __init__(self) -> None:
    self.dataset = None
    self.X = None
    self.model = None
    self.k_means_cluster_centers = None
    self.k_means_labels = None
    self.wcss = None
  
  def get_best_k_elbow_method(self, n_clusters:int, **kwargs):
    kelbow_visualizer(KMeans(**kwargs), self.X, k=(1, n_clusters), timings=False)
  
  def get_best_k_mean_shift(self, n_clusters:int, **kwargs):
    # using MeanShift to get an estimate
    bandwidth = estimate_bandwidth(self.X, quantile=0.3, n_jobs=-1)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=False, n_jobs=-1, max_iter=500)
    ms.fit(self.X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_unique = numpy.unique(labels)
    n_clusters_ = len(labels_unique)
    print(f"Number of estimated clusters MS : {n_clusters_}")
    
  def get_best_cluster_number(self, n_clusters:int, method:str='both' ,**kwargs):
    if (method == 'both'):
      self.get_best_k_elbow_method(n_clusters=n_clusters, **kwargs)
      self.get_best_k_mean_shift(n_clusters=n_clusters, **kwargs)
    
    elif(method == 'elbow'):
      self.get_best_k_elbow_method(n_clusters=n_clusters, **kwargs)
    
    elif(method == 'ms'):
      self.get_best_k_mean_shift(n_clusters=n_clusters, **kwargs)
    
    else:
      raise Exception(f'method should be: both, elbow or ms, got:{method}')  
    
  def plot_results(self, figsize:list = [10,10], seaborn_grid_style:str='darkgrid', cluster_color:str='red'):
    fig = plt.figure(figsize=figsize)
    seaborn.set(style=seaborn_grid_style)
    ax = fig.add_subplot(111, projection = '3d')
    x = self.X[:,0]
    y = self.X[:,1]
    z = self.X[:,2]
    ax.set_xlabel('price_level')
    ax.set_ylabel('rating')
    ax.set_zlabel('user_ratings_total')
    ax.scatter(x, y, z)
    for cluster in self.k_means_cluster_centers:
      ax.scatter(cluster[0],cluster[1],cluster[2], color=cluster_color)
    plt.show()
  
  def plot_intercluster_distance(self):
    intercluster_distance(self.model, 
                          self.X, 
                          embedding='mds', 
                          random_state=12)

    
  def preprocess(self, dataframe:pandas.DataFrame, standarize:bool=False)->None:
    self.dataset = dataframe
    if(standarize==True):      
     self.X = StandardScaler().fit_transform(self.dataset.values)
    
    else:
      self.X = self.dataset.values

  def train(self, n_clusters:int=8,  **kwargs):    
      self.model = KMeans(n_clusters=n_clusters, **kwargs).fit(self.X)
      self.k_means_cluster_centers = self.model.cluster_centers_
      self.k_means_labels = self.model.labels_
  
  def predict_over_a_dataset(self, dataframe:pandas.DataFrame):
      return self.model.predict(StandardScaler().fit_transform(dataframe.values.reshape(1, -1)))
