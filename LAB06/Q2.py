import pandas as pd

def gini(data):
    total = len(data)
    freq = {}
    for v in data:
        freq[v] = freq.get(v, 0) + 1
    g = 1
    for c in freq.values():
        p = c / total
        g -= p * p
    return g

file = input("Enter CSV file path: ")
df = pd.read_csv(file)

target = input("Enter target column: ")
y = df[target].values

print("Gini Index:", gini(y))
