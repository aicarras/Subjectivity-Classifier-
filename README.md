# Subjectivity-Classifier-

## The project: 
Neutral writing and text is expected in many sectors related with the medium of language. For example, scientists, and journalists are all expected to use less opinionated and more factual language. However, the ambiguity of language makes the task of spotting bias or subjectivities in text very hard. Moreover, the terms objective and subjective are themselves ambiguous. Consequently, developing a model to classify text as being neutral and subjective is a challenging yet interesting task. There are also significant implications for Natural Language Processing (NLP) since subjectivity can be helpful for information extraction (Riloff, Wiebe, and Philips, 2005). Subjectivity analysis can also be useful in industry, in the form of an application, in order to assess the objectivity of text, for example in a given Wikipedia entry, a newspaper article, or an academic publication. 
	Identifying biased and neutral text is a notoriously difficult task (Wiebe, Wilson, Bruce, Bell, and Martin, 2004). There are different types of bias such as societal bias, prejudice, opinions/private states and more. This paper is concerned with identifying text and textual elements that reveal the latter, that is private states, opinions, speculations, and subjective attitudes (Wiebe, 1994). Moreover, subjectivity may differ in terms of its source, that is who’s private state is being revealed (Recasens, Danescu-Mizil, and Jurafsky 2013). For example, a journalist may simply report the emotions of someone else, revealing a private state, albeit not the writer’s. That said, there are linguistic cues being addressed in the literature that are often found in subjective sentences. For example, factive verbs, personal and possessive pronouns, and some adjectives are often found in biased text (Recanses et al., 2013). 
Given recent work on subjectivity analysis (discussed in the next section) and the difficulty but also usefulness of creating a model that can identify biased text, this paper answers the research question: What is the best performing model, its parameter configuration, and given the importance of language cues, what is the influence of pre-processing in a binary bias classification task? 


Attempting to identify text that is subjective


- A sentence-level subjectivity classification task

### TO RUN:  

- In main.py an args script is given, hopefully the "help" parameter straightforward. Through this script you can run different experiments
on the different models, with different training and test data (development data is configured automatically). You can also load the best perfroming
SVM model (check paper). Finally, you can select to input a sentence interactively which will then be classified by the selected model. To do this, set the --user_input argument to True

- cd into src and type python3 main.py

- If you want to run the pretrained SVM model please use the SVM10000POS.joblib. To do so set the respective argument in the parser script (i.e. --load_model) to True.


Let me know if you require any further information or if you want to expand on the project. 

I hope you find it interesting!
