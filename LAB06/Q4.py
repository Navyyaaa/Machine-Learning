import pandas as pd

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

file = input("Enter CSV file path: ")
df = pd.read_csv(file)

col = input("Enter column to bin: ")
bins = int(input("Enter bins (0 for default): "))

data = df[col].values

if bins == 0:
    result = binning(data)
else:
    result = binning(data, bins)

print(result)
