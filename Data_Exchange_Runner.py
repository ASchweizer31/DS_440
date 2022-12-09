from Database_Connections import *

from Data_Retrieval import * 

import config as fig

### Data Retrieval ###


# unshaped_X_train = np.loadtxt("Sample_Stock_Dataset_X_train.txt")


# my_X_train = unshaped_X_train.reshape(
#     unshaped_X_train.shape[0], unshaped_X_train.shape[1] // unshaped_X_train.shape[2], unshaped_X_train.shape[2])


# my_y_train = pd.read_csv("Sample_Stock_Dataset_y_train.csv")


# predicted_dataframe = pd.read_csv("/home/openvessel/Documents/Data Science/LSTM_Stock_Project/Sample_Stock_Training_Data/Sample_LSTM_Model_Results_4_Predictions_for_GCP_insertion.csv", index_col=False)

# dataframe = predicted_dataframe.drop(predicted_dataframe.columns[[0]], axis=1)




### Function to Run ###

def get_stock_pred_df_from_df(model, dataframe):

    pred_df = pd.DataFrame()

    for record in dataframe:

        prediction = model.predict(record)

        pred_df.iloc[pred_df.index] = [prediction]

    return pred_df

################
## Get Stock Predictions Data 


if fig.create_data == True:
    pred_dict = example_data_creation()
    
else:
    stock_input_data_dict = get_stock_input_data()
    pred_dict = {}
    for stock_data_df_key in stock_input_data_dict:
        pred_dict[stock_data_df_key] = get_stock_pred_df_from_df(stock_input_data_dict[stock_data_df_key])


############

gcp_ready_dict = combine_predictions_and_info_for_GCP_insertion(pred_dict)

def insert_predictions_and_info_into_GCP(gcp_ready_predictions_dict):

    for key in gcp_ready_predictions_dict:
        #gcp_ready_dict[key].to_csv(key+".csv")
        insert_data_into_GCP_table(gcp_ready_predictions_dict[key], fig.daily_update_queries[key], fig.password, fig.host, fig.database)


### Deleting everything in the tables 

for table_name in ["stock_screener", "one_day_predictions", "three_day_predictions", "five_day_predictions"]:
    delete_from_table(table_name, fig.password, fig.host, fig.database)

import os 

curr_path = "/home/openvessel/Documents/Data Science/Capstone_Stock_Ai/Front_End_and_Back_End (DS 440)"
folder_path = curr_path + "/Sample_Data_For_Wix"
if not os.path.exists(folder_path):
    os.mkdir(folder_path)

pred_dict = example_data_creation()

gcp_ready_dict = combine_predictions_and_info_for_GCP_insertion(pred_dict["one_day"],pred_dict["three_day"], pred_dict["five_day"])

for key in gcp_ready_dict:
    gcp_ready_dict[key].to_csv(folder_path+"/"+key+".csv", index=False)    

# update_daily_stock_info()

####

# clock(fig.daily_update_queries)

# print_GCP_DB_info(fig.project, fig.instance, fig.database)

# connect_to_service()

# connect_to_GCP(fig.password, fig.host, fig.database)

# insert_data_into_GCP_table(fig.dataframe, fig.wix_screener_data_update_query, fig.password, fig.host, fig.database)