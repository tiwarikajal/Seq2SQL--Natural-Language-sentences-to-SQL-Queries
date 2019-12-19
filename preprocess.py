from util.DataConversionUtil import DataConversionUtil

# Create a tokenized dataset for the train, test and validation datasets
# Enables model to run faster during actual training since all data is already 
# preprocessed
data_converter = DataConversionUtil()
data_converter.build_tokenized_dataset("train")
data_converter.build_tokenized_dataset("dev")
data_converter.build_tokenized_dataset("test")