import pandas as pd
import numpy as np
from LSTM_Model import cross_validate, plot_histories, create_LSTM

import tensorflow as tf 
from tensorflow import saved_model

############ Config ############ 
to_put_model_results_path = '/home/openvessel/Documents/Data Science/LSTM_Stock_Project/LSTM_Model_Results_4'

saved_model_results_path = "/home/openvessel/Documents/Data Science/LSTM_Stock_Project/LSTM_Model_Results_4"

predictions_csv_path = "/home/openvessel/Documents/Data Science/LSTM_Stock_Project/Sample_Stock_Training_Data/Sample_LSTM_Model_Results_4_Predictions_for_GCP_insertion.csv"

############### Data #############
# og_X_train_3d_shape = 28

# unshaped_X_train = np.loadtxt("/home/openvessel/Documents/Data Science/LSTM_Stock_Project/Sample_Stock_Training_Data/Sample_Stock_Dataset_X_train_Model_Ready.txt")

# my_X_train = np.array(unshaped_X_train.reshape(
#     unshaped_X_train.shape[0], unshaped_X_train.shape[1] // og_X_train_3d_shape, og_X_train_3d_shape))

# my_y_train = pd.read_csv("/home/openvessel/Documents/Data Science/LSTM_Stock_Project/Sample_Stock_Training_Data/Sample_Stock_Dataset_y_train_Model_Ready.csv")

# my_y_train = my_y_train.to_numpy()

og_X_train_3d_shape = 28

x_train_path = "/home/openvessel/Documents/Data Science/LSTM_Stock_Project/Sample_Stock_Training_Data/Sample_Stock_Dataset_X_train_Model_Ready.txt"
y_train_path = "/home/openvessel/Documents/Data Science/LSTM_Stock_Project/Sample_Stock_Training_Data/Sample_Stock_Dataset_y_train_Model_Ready.csv"
ticker_dataset_path = "/home/openvessel/Documents/Data Science/LSTM_Stock_Project/Sample_Stock_Training_Data/Sample_Stock_Data_Raw_No_IDs.csv"

x_train = np.loadtxt(x_train_path)

x_train = x_train.reshape(
    x_train.shape[0], x_train.shape[1] // og_X_train_3d_shape, og_X_train_3d_shape)


for row in [498, 1566, 37756, 45895]:
    print(x_train[None,row,:,:]) 


y_train = pd.read_csv(y_train_path)

y_train = y_train.to_numpy()


ticker_dataset = pd.read_csv(ticker_dataset_path)

############# Runner Functions ###########


def run_LSTM(my_X_train, my_y_train, to_put_model_results_path):

    scores, histories = cross_validate(my_X_train, my_y_train, to_put_model_results_path)
    plot_histories(histories)



def predict_from_LSTM(saved_model_results_path, x_train, y_train, ticker_dataset, predictions_csv_path):
    model = create_LSTM()
    loaded_weight_model = tf.keras.models.load_model(saved_model_results_path)
    dataframe = pd.DataFrame()
    for row in [498, 1566, 37756, 45895]:
        final_data_row = list()
        
        prediction = loaded_weight_model.predict(x_train[None,row,:,:])
        # de-standardize and compare to original price to calculate percentage gain 


        print("loaded_weight_model.predict(x_train[None,row,:,:]) = ", prediction)

        final_data_row.append(str(ticker_dataset.iloc[[row]]["symbol"].values[0]))
        
        print('str(ticker_dataset.iloc[[row]]["symbol"].values[0]) = ', str(ticker_dataset.iloc[[row]]["symbol"].values[0]))

        print("list(np.squeeze(predictions[0,0:3])= ", list(np.squeeze(prediction[0,0:3])))

        for pred in list(np.squeeze(prediction[0,0:3])):
            final_data_row.append(str(round(float(pred)*10, 2)))
        
        print("final_data_row= ", final_data_row)

        dataframe = dataframe.append(pd.Series(final_data_row), ignore_index=True)
    
    dataframe.to_csv(predictions_csv_path)

    print("Dataframe Saved!!! @ :", predictions_csv_path) 

######### Running Session ###########

# run_LSTM(x_train, y_train, to_put_model_results_path)

#predict_from_LSTM(saved_model_results_path, x_train, y_train, ticker_dataset, predictions_csv_path)