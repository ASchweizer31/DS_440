
import pandas as pd 
import numpy as np
import random

import config as fig 


## for eod api calls 
from eodhd import APIClient
from urllib.request import urlopen
import json


def get_list_of_all_tickers_to_use():
    tickers_to_use = ["AAPL","NVDA","MU", "DIS", "MSFT", "TSLA", "WMT", "AMZN", "PFE", "HD", "JNJ", "XOM","KO","BAC", "RTX", "NFLX", "NKE", "HON", "SBUX", "LMT", "BA", "CAT", "TMUS", "TGT"]
    return tickers_to_use

def example_data_creation():
    curr_path = "/home/openvessel/Documents/Data Science/Capstone_Stock_Ai/Front_End_and_Back_End (DS 440)"

    predictions_dict = dict()

    pred_columns = ["Ticker", "Price_Change"]    
    one_day_df = pd.DataFrame(columns=pred_columns)
    three_day_df = pd.DataFrame(columns=pred_columns)
    five_day_df = pd.DataFrame(columns=pred_columns)
    screener_columns = ["one_day", "three_day", "five_day"]
    screener_df = pd.DataFrame(columns=screener_columns)
    screener_df["predicted_price_change_rank"] = [i for i in range(1,11)]

    counter = 0 

    days_list = ["one_day", "three_day", "five_day"]

    for df in [one_day_df, three_day_df, five_day_df]:
        
        new_tickers_to_use = get_list_of_all_tickers_to_use()
        
        for i in range(0, 10):
            ticker = random.choice(new_tickers_to_use)
            
            rand_sign = random.randint(0,1)
            
            if rand_sign == 0:
                rand_sign = -1
            
            rand_fraction = round(random.random(), 2)

            rand_price_increase = random.randint(1,15)

            predicted_price_change = round(rand_sign * rand_price_increase * rand_fraction, 2)

            df.loc[len(df.index)] = [ticker, predicted_price_change]

            new_tickers_to_use.remove(ticker)
        
        sorted_index = df.Price_Change.abs().sort_values(ascending=False).index

        df = df.reindex(sorted_index)

        curr_day = days_list[counter]
        
        predictions_dict[curr_day] = df

        df.to_csv(curr_path+"/"+curr_day+".csv", index=False)

        # ticker (price_change %) 
        restructured_column = [list(df.iloc[[i]]["Ticker"])[0] + " ({}%)".format(list(df.iloc[[i]]["Price_Change"])[0]) for i in range(0,len(df.index))]

        screener_df[curr_day] = restructured_column

        counter+=1 
    
    predictions_dict["stock_screener"] = screener_df

    screener_df.to_csv(curr_path+"/stock_screener.csv", index=False)

    return predictions_dict



def get_stock_info_from_EOD(ticker):
    api = APIClient(fig.api_token)
    url = "https://eodhistoricaldata.com/api/fundamentals/{}.US?api_token={}&fmt=json".format(ticker, fig.api_token)
    response = urlopen(url)
    data = json.loads(response.read())

    # need: ticker, company_name, logo, sector, industry, description, ceo_name, exchange 
    record = dict()
    record["ticker"] = ticker
    record["company_name"] = data["General"]["Name"]
    record["logo_path"] = "https://eodhistoricaldata.com" + data["General"]["LogoURL"]
    record["sector"] = data["General"]["GicSector"]
    record["industry"] = data["General"]["GicIndustry"]
    record["description"] = data["General"]["Description"]
    record["ceo_name"] = data["General"]["Officers"]["0"]["Name"]
    record["exchange"] = data["General"]["Exchange"]
    
    return record



def combine_predictions_and_info_for_GCP_insertion(predictions_dict):
    
    days_list = ["one_day", "three_day", "five_day"]
    pred_columns= ["predicted_price_change_rank", "ticker", "company_name", "predicted_price_change","logo","sector", "industry", "description","ceo_name", "exchange"]
        
    gcp_ready_data_dict = dict()
    
    if "screener" in predictions_dict.keys:
        gcp_ready_data_dict["screener"] = predictions_dict["screener"]
    
    counter = 0 

    for df_key in predictions_dict:

        if df_key == "screener":
            continue
        
        curr_pred_df = predictions_dict[df_key]

        new_df = pd.DataFrame(columns=pred_columns)

        for row in range(0, len(curr_pred_df)):
            
            ticker = list(curr_pred_df.iloc[[row]]["Ticker"])[0]
            predicted_price = list(curr_pred_df.iloc[[row]]["Price_Change"])[0]
            info = get_stock_info_from_EOD(ticker)
            
            rank = row+1
            full_record_list = [rank, ticker, info["company_name"],predicted_price, info["logo_path"], info["sector"], info["industry"], info["description"], info["ceo_name"], info["exchange"]]
            
            new_df.loc[len(new_df.index)] = full_record_list
        
        gcp_ready_data_dict[df_key] = new_df

        counter+=1 

    return gcp_ready_data_dict