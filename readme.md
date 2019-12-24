
# Natural Language to SQL


This project is an implementation of the Seq2SQL model described in [https://arxiv.org/pdf/1709.00103.pdf](https://arxiv.org/pdf/1709.00103.pdf)

Here we have also implemented the baseline sequence to sequence model


### Setup Instructions
- The dataset must be downloaded from [https://github.com/salesforce/WikiSQL](https://github.com/salesforce/WikiSQL) and then unzipped and placed in the data directory
- Install sqlite using the links here [https://www.sqlite.org/download.html](https://www.sqlite.org/download.html)
- Next, install the project requirements using `pip install -r requirements.txt`
- Download the glove embeddings from http://nlp.stanford.edu/data/glove.6B.zip
- Extract the archive into the glove folder
- Run the pre-processing script `python preprocess.py` . This will create the tokenized versions of the dataset
- Run `python main.py` . This will run the baseline model followed by the target model.
- Running `main.py` will take approximately 10 hours. Please make sure to use a system with a good GPU.
- It is highly recommended that this project is run in an anaconda environment. This will give the interpreter access to common libraries that may have been missed in requirements.txt


### Folder Structure
- The `data` and `glove` directory are for the dataset and embeddings
- The `library` folder contains code provided by WikiSQL to perform basic data conversions and query running
- The `util` directory contains files related to common functionality such as plotting graphs, loading datasets, preparing parallel datasets in-memory for fast access, creating batch sequences for models, and checking model accuracy.
- The `baseline` directory contains all code necessary for the baseline to run
- The `seq2sql` directory contains all code pertaining to the target model 
- The `saved_model` directory is where the target model will save the best model after training


### Important Files
The entry point to the project is the `main.py`  file. From here it is possible to control which model(s) we want to run. The `preprocess.py` is another essential file as it results in the generation of the tokenized dataset. Altering the tokenizing logic could significantly impact the results. `constants.py` contains multiple parameters used by the target model like batch size, learning rate, number of epochs, etc.

Upon completion of the run, the code will generate loss graphs and store the results of the target model into a text file in the root directory of the project