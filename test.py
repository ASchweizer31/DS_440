from eodhd import APIClient
from urllib.request import urlopen
import json
import pandas as pd

ticker = "NVDA"
api_token = "6366c9de5a5ae4.53441424"

# api = APIClient(api_token)
# url = "https://eodhistoricaldata.com/api/fundamentals/{}.US?api_token={}&fmt=json".format(ticker, api_token)
# response = urlopen(url)
# data = json.loads(response.read())

# print(data["General"]["Name"])


df = pd.read_csv("one_day.csv", index_col=False)

print("------------")
print("1", df.head())
# print(df[0])
print("-------------")
print("2:-", df.iloc[1])
print("------------")
print("3:-", list(df.iloc[[0]]["Ticker"])[0])
