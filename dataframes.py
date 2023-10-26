import pandas as pd

# Sample DataFrames
df1 = pd.DataFrame({"A": [1, 2, 5], "B": [3, 4, 9]})
df2 = pd.DataFrame({"A": [5, 6], "B": [7, 8]})

# Concatenate with ignore_index set to True
result = pd.concat([df1, df2])
r2 = result.drop_duplicates(subset="A")
