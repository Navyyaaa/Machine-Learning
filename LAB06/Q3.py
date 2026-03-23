import pandas as pd
import math

def entropy(data):
    total = len(data)
    freq = {}
    for v in data:
        freq[v] = freq.get(v, 0) + 1
    h = 0
    for c in freq.values():
        p = c / total
        h += -p * math.log2(p)
    return h

def info_gain(feature, target):
    total_entropy = entropy(target)
    values = set(feature)
    w_entropy = 0
    for v in values:
        subset = []
        for i in range(len(feature)):
            if feature[i] == v:
                subset.append(target[i])
        w_entropy += (len(subset)/len(target)) * entropy(subset)
    return total_entropy - w_entropy

file = input("Enter CSV file path: ")
df = pd.read_csv(file)

target = input("Enter target column: ")
y = df[target].values

cols = [c for c in df.columns if c != target]

gains = []
for c in cols:
    g = info_gain(df[c].values, y)
    gains.append(g)
    print(c, "Gain:", g)

print("Root Feature:", cols[gains.index(max(gains))])
