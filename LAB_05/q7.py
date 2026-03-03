import pandas as pd
import ast
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
data = pd.read_csv(r"D:\Sem4\ml\lab\assignments\done\LAB05\projectds.csv")

data["embedding"] = data["embedding"].apply(ast.literal_eval)
X = pd.DataFrame(data["embedding"].to_list())
distortions = []

for k in range(2, 20):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(X)
    distortions.append(kmeans.inertia_)
plt.plot(range(2, 20), distortions, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.show()
