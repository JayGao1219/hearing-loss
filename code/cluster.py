import numpy as np
from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import xlrd
import matplotlib.pyplot as plt
import pandas as pd

def get_cell(table):
    nrows=table.nrows
    ncols=table.ncols
    res=[]
    for i in range(1,nrows):
        cur=[]
        for j in range(1,ncols):
            if table.cell_value(i,j)!='':
                try:
                    cur.append(int(table.cell_value(i,j)))
                except:
                    break
        res.append(cur)
    return res

def get_data(filename,sheet_name):
    data=xlrd.open_workbook(filename)
    whole=[]
    for item in sheet_name:
        cur_table=data.sheet_by_name(item)
        whole=whole+get_cell(cur_table)
    return whole

def get_array_data(filename):
    # Read in the data
    df = pd.read_excel(filename, sheet_name='Sheet')
    return df.to_numpy()

def kmeans_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(data)
    return labels

def dbscan_clustering(data, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data)
    return labels

def agglomerative_clustering(data, n_clusters, linkage):
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = agglomerative.fit_predict(data)
    return labels

def save_image(x, y, xlabel, ylabel, title, filename):
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def run_cluster(cluster_method):
    print("开始聚类",cluster_method)
    data=get_array_data('../data/au.xlsx')
    data=data[:,1:] # 去掉第一列, 第一列是序号
    data = data[~np.isnan(data).any(axis=1)]
    print("过滤掉nan数据点")
    # 标准化数据
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    print("标准化数据")
    # 使用肘部法则选择合适的聚类数目
    max_clusters = 5
    sse = []
    silhouette_coefficients = []

    for k in range(2, max_clusters + 1):
        print("聚类数目为",k)
        if cluster_method=='kmeans':
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(data_scaled)
            sse.append(kmeans.inertia_)
            silhouette_coefficients.append(silhouette_score(data_scaled, kmeans.labels_))
        elif cluster_method=='dbscan':
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            dbscan.fit(data_scaled)
            silhouette_coefficients.append(silhouette_score(data_scaled, dbscan.labels_))
        elif cluster_method=='agglomerative':
            agglomerative = AgglomerativeClustering(n_clusters=k, linkage='ward')
            agglomerative.fit(data_scaled)
            # sse.append(agglomerative.inertia_)
            silhouette_coefficients.append(silhouette_score(data_scaled, agglomerative.labels_))

    if len(sse)>0:
        # 绘制肘部法则图
        save_image(range(2, max_clusters + 1), sse, 'Number of Clusters', 'SSE', 'Elbow Method', '../image/'+cluster_method+'_elbow.png')

    # 绘制轮廓系数图
    save_image(range(2, max_clusters + 1), silhouette_coefficients, 'Number of Clusters', 'Silhouette Coefficient', 'Silhouette Coefficient', '../image/'+cluster_method+'_silhouette.png')

if __name__ == '__main__':
    run_cluster('kmeans')
    # run_cluster('dbscan')
    # run_cluster('agglomerative')