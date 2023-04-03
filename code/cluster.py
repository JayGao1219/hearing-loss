import numpy as np
from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import xlrd


def get_cell(table):
    nrows=table.nrows
    ncols=table.ncols
    res=[]
    print(nrows,ncols)
    for i in range(1,nrows):
        cur=[]
        for j in range(1,ncols):
            cur.append(int(table.cell_value(i,j)))

        res.append(cur)
    return res

def get_data(filename,sheet_name):
    data=xlrd.open_workbook(filename)
    right_table=data.sheet_by_name(sheet_name[0])
    left_table=data.sheet_by_name(sheet_name[1])
    whole=get_cell(right_table)
    whole=whole+get_cell(left_table)
    return whole

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
