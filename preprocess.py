from __future__ import unicode_literals, print_function, division
from util.DataConversionUtil import DataConversionUtil

data_converter = DataConversionUtil()

data_converter.build_tokenized_dataset("train")
data_converter.build_tokenized_dataset("dev")
data_converter.build_tokenized_dataset("test")