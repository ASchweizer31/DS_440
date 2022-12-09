

# "DELETE FROM model_results"
#wix_model_results_data_update_query = "INSERT INTO model_results (ticker, three_days, three_weeks, three_months) VALUES (%s, %s, %s, %s)"

stock_screener_query = "INSERT IGNORE INTO stock_screener(predicted_price_change_rank, one_day, three_day, five_day) VALUES (%s, %s, %s, %s)"

daily_update_queries = {}

daily_update_queries["stock_screener"] = stock_screener_query

predictions_time_list = ["one", "three", "five"]

for time in predictions_time_list:
    predictions_query = "INSERT IGNORE INTO {}_day_predictions(predicted_price_change_rank, ticker, company_name, predicted_price_change, logo, sector, industry, description, ceo_name, exchange) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)".format(time) 
    daily_update_queries[time+"_day"] = predictions_query


password = "oz6l+='4miV-v7/6"
host = "34.122.21.59" 

    # '34.122.21.59', (other IP address at school)
    # at home: 96.245.60.196

project = 'ds-440' 

instance = 'stock-screener-2' 

database = 'ai_predictions'  


## eod 
api_token = "6366c9de5a5ae4.53441424"

create_data = True