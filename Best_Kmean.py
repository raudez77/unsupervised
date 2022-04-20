
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.cluster import Kmeans
from sklearn.metrics import silhouette_score,silhouette_samples
import matplotlib.cm as cm
import matplotlib_venn as vplt
import squarify  
sns.set_style("ticks")



def Best_Kmean(X,k,df, random_state=104):
    
    """ Return a dashboard showing n-k means approach 
    Arg: 
    x : array (depedent variables)
    k : int (2 - infinitive) # number of K means to consider
    df : DatFrame Pandas (This must be given because it takes columns name, and other information to create the dasboard
    """
    
    #taking Dataset copy to Find Clusters
    df_cluster = df.copy
    
    # Creting K
    silhouette_=[]
    Inertia = []
    range_n_clusters = list(range(2,k+1))
    time = 0
    
    
    for n_clusters in range_n_clusters:
        fig,(ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize =(25,5))
        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
        
        clusterer = KMeans(n_clusters=n_clusters, random_state=random_state,init = 'k-means++')
        cluster_labels = clusterer.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_.append(silhouette_avg)
        Inertia.append(clusterer.inertia_)
        unique, counts = np.unique(cluster_labels, return_counts=True)
        #print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
        
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i+1))
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
        #---------------------> Optimal K silhouette_
        k_optimal = np.argmax(silhouette_) + 2
        ax2.plot(range(2,n_clusters),silhouette_[:time],linestyle = '--',color = 'blue', linewidth =3, markersize= 8)
        ax2.scatter(n_clusters,silhouette_avg)
        time +=1
        
        #Label and More
        ax2.set_title('Silhouette Curve for optimal Number of clusters' , family = 'Arial', fontsize = 8)
        ax2.set_xlabel(f'K={n_clusters}', fontsize = 14, family = 'Arial') 
        ax2.set_ylabel('Silhouette_score', fontsize = 14, family = 'Arial')
        ax2.axvline(x=k_optimal,label='Optimal number of cluster ({optimal})'.format(optimal=k_optimal),linestyle = '--', color ='red')
        ax2.set_yticks(np.linspace(0.1,1,k))  
        ax2.set_xticks(np.arange(2,k+1))
        ax2.scatter(k_optimal ,silhouette_[k_optimal-2], color = 'red', s=100)
        ax2.legend(fontsize = 'medium')
        
        #---------------------> K - Elbow Method
        ax3.set_title('Elbow method for Optimal K' , family = 'Arial', fontsize = 10)
        ax3.plot(np.arange(2,n_clusters+1),Inertia[:time],linestyle = '--',color = 'blue', linewidth =3)
        ax3.scatter(np.arange(2,n_clusters+1),Inertia[:time],color = 'red') #dots
        ax3.scatter(n_clusters,clusterer.inertia_,color = 'red', s=100) #dots
        ax3.set_xlabel('Number of Clusters')
        ax3.set_ylabel('Inertia')
        
        #---------------------> Bar Plot
        ax4.set_title('Cluster Distribution',family = 'Arial', fontsize = 8)
        bar = sns.barplot(x=unique+1, y = counts ,palette='Paired')
        bar.set_xticklabels(bar.get_xticklabels(),  rotation=90, horizontalalignment='right');
        sns.despine(left=True, bottom=True ,ax=ax4)
        ax4.get_yaxis().set_visible(False)
        i=0


        for p in bar.patches:
            height = p.get_height()
            bar.text(p.get_x()+p.get_width()/2., height + 0.1, counts[i],ha="center")
            i += 1

    plt.show()
