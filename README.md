# Subjectivity-Classifier-
Attempting to identify text that is subjective


- This is my first NLP project - A sentence-level subjectivity classification task

TO RUN:  

- In main.py an args script is given, hopefully the "help" parameter straightforward. Through this script you can run different experiments
on the different models, with different training and test data (development data is configured automatically). You can also load the best perfroming
SVM model (check paper). Finally, you can select to input a sentence interactively which will then be classified by the selected model. To do this, set the --user_input argument to True

- cd into src and type python3 main.py

- If you want to run the pretrained model please move the SVM10000POS.joblib into the src folder that was uploaded on the student portal. Also, set the respective argument in the parser script (i.e. --load_model) to True.


Let me know if you require any further information or if you want to expand on the project. 

I hope you find it interesting!
