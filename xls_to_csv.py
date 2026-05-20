import pandas as pd
df = pd.read_excel("car_speeds.xlsx")
df.to_csv("car_speeds.csv", index=False)