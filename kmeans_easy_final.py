import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("easy_queue_data.csv")

X = data[['wait_time', 'service_time']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=2, random_state=0)
data['cluster'] = kmeans.fit_predict(X_scaled)

print(data.head())
