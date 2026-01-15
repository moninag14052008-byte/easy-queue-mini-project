import pandas as pd
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv("easy_queue_data.csv")

X = data[['wait_time', 'service_time']]
y = data['label']

model = GaussianNB()
model.fit(X, y)

data['prob_problematic'] = model.predict_proba(X)[:, 1]

print(data.head())
