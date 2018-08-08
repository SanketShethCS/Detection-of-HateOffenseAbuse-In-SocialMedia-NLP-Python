# Detection-of-HateOffenseAbuse-In-SocialMedia-NLP-Python

The analysis and model is explained in the detection-hate-offensive.pdf file which also serves as the technical paper for this research.

For code:
This is a readme for the main classifier code named NLP_project.py

This file is written in python and requires the data file named labeled_data.csv
to run along with many packages like pandas,nltk,scikit learn readil available
in anaconda and spyder. 

The file contains two modules primarily one is for training the model on the given data
which has to be in a specific format as stated in the data readme and then tests unseen data 
in same format to analyze the accuracy and results acheived on the model.
The training module is commented initially as there is an already tarined module
pickled and ready to be tested on provided in the folder named Project_Classifier_model.sav

The code is commented and each parts are explained.

For Data:
The data is provided as a csv file named lavbeled_data.csv along with the
other files required for submission, the data is split into 7 columns with each 
column representing something different.
The explaination of each column is stated below-

 Column 1: Tweet ID
 Column 2: Total number of human annotators 
 Column 3: Total number of annotations for hate speech
 Column 4: Total number of annotations for offensive speech
 Column 5: Total number of annotations for neither
 Column 6: Final voted label attached to the tweet
 Column 7: Tweet text

The data is fed in the model by using pandas in python.

For Auxiliary Files:
This is a Readme for the auxiliary files listed below-
-Experiment.py
-seperate_0_1.py
-seperate_0_2.py
-seperate_1_2.py

For the first one that is experiment.py, this file is purely for
experimentation and only requires the labeled_data.csv along with 
other python packages readily available with anaconda and spyder to
run. This file has a feature,classifier and model bank that are commented
at the end of the code that can be used to replace modules in the present code 
also provided in the same experiment.py file to run various combinations
of features and models to check out results and perform analysis.


For the rest of the files, these files are purely for testing the effects
of skewing and normalizing the data and only requires the labeled_data.csv 
along with  other python packages readily available with anaconda and spyder to
run. This files are named in such a way that the two number seperated by an 
underscore represent the two labels compared in a binary classification problem.
Also there is code commented for nomrmalizing the data in order to use that 
comment the indicated code and uncomment the normalizing code to run and analyze
the results achieved through that process.

Note- These 4 files has uncommented code as they are beyond the project and 
have just been provided as a source of justification for few of the results
stated in the report.