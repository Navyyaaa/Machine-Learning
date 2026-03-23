import pandas as pd
import math

def binning(data, bins=4):
    mn = min(data)
    mx = max(data)
    w = (mx - mn) / bins
    res = []
    for v in data:
        idx = int((v - mn) / w)
        if idx == bins:
            idx -= 1
        res.append(idx)
    return res

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

file = input("Enter CSV file path: ")
df = pd.read_csv(file)

target = input("Enter target column: ")
y = df[target].values

y = binning(y)

print("Entropy:", entropy(y))
