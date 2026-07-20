#This file is intended to enable some cross-file communication
from src.data_loading import Datasets

#This part lets the strategy file communicate with the global eval function
#To say whether to run evaluation after a server/client communication round
last_update_round: int = 0
total_model_updates: int = 0

#This part communicates the dataset from serverapp to model/data loading
dataset: Datasets