# Subjectivity-Classifier-

### Table of Contents:

[The Project](#The-Project)<br>
[Related Work](#Related-work)<br>
[The Model and the Task](#The-Model-and-the-Task)<br>
[The Dataset](#The-Dataset)<br>
[Conculsion](#Conclusion)<br>
[Limitations](#Limitations)<br>
[Future Work](#Future-Work)<br>
**[TO RUN](#TO-RUN)**<br>
[References/Sources](#References/Sources)<br>

## The Project: 
Neutral writing and text are the norm in many sectors related to the medium of language. For example, scientists and journalists are both expected to use less opinionated and more factual language. However, the ambiguity of language makes the task of spotting bias or subjectivities in text very hard. Moreover, the terms objective and subjective are themselves ambiguous. Consequently, developing a model to classify text as being neutral and subjective is a challenging yet interesting task. There are also significant implications for Natural Language Processing (NLP) since subjectivity can be helpful for information extraction (Riloff, Wiebe, and Philips, 2005). Subjectivity analysis can also be useful in industry, in the form of an application, in order to assess the objectivity of text, for example in a given Wikipedia entry, a newspaper article, or an academic publication. 
	Identifying biased and neutral text is a notoriously difficult task (Wiebe, Wilson, Bruce, Bell, and Martin, 2004). There are different types of bias such as societal bias, prejudice, opinions/private states and more. This paper is concerned with identifying text and textual elements that reveal the latter, that is private states, opinions, speculations, and subjective attitudes (Wiebe, 1994). Moreover, subjectivity may differ in terms of its source, that is who’s private state is being revealed (Recasens, Danescu-Mizil, and Jurafsky 2013). For example, a journalist may simply report the emotions of someone else, revealing a private state, albeit not the writer’s. That said, there are linguistic cues being addressed in the literature that are often found in subjective sentences. For example, factive verbs, personal and possessive pronouns, and some adjectives are often found in biased text (Recanses et al., 2013). 
Given recent work on subjectivity analysis (discussed in the next section) and the difficulty but also usefulness of creating a model that can identify biased text, this paper answers the research question: What is the best performing model, its parameter configuration, and given the importance of language cues, what is the influence of pre-processing in a binary bias classification task? 

This project was largerly insipired by the paper titled: ["Automatically Neutralizing Subjective Bias in Text"](https://arxiv.org/abs/1911.09709)


## Related Work

Past research in subjectivity analysis has employed rule-based, lexicon-based (i.e. containing cues often found in subjective text), unsupervised, and supervised approaches (Wiebe et al., 2004). More recent approaches by Pryzant, Martinez, Das, Kurohashi, Jurafsky, and Yang (2019) have used a BERT model that finds specific words in a text that reveal a private state and replaces them with neutral words. A lot of research has been made on the identification of cues that make a sentence subjective. For example, Recasens et al., (2013, p. 1650) developed a set of linguistic features “including factive verbs, implicative, hedges, and subjective intensifiers” which are often found in subjective text. In general, past work in the field has focused on document-level bias, or word-level bias. That is, classifying a whole text as biased and identifying the bias inducing word in a sentence, that is the word revealing some private, subjective view. Other notable work by Pang and Lee (2004) uses graph theory and Naïve Bayes classifier to identify bias in a series of related sentences.


## The Model and the Task

This paper differs from past research in that the subjectivity classification is done on a sentence-level. This approach was chosen for the following reasons: First, bias is ubiquitous and very subtle, therefore there is a very small chance that a whole document is classified as neutral (Wiebe et al., 2004). Second, word-level subjectivity analysis is very difficult due to the subtleness of the subjective cues and may classify a sentence as biased whenever any private state is present disregarding whether a text is an objective report of a subjective fact (Pang and Lee, 2004). A sentence-level subjectivity analysis however looks for bias that has as a source the writer of the text. Examples in appendix A may further illustrate this point. Moreover, sentence-level classification is more versatile allowing for larger training data sizes and make the models created in this project extendable as they have been adapted to handle user – input sentences for classification on whether their sentence is biased or not. 
	The models are supplied a sentence-label pair during training. Given the isolated-sentence-level classification task classifiers that can receive individual feature vectors as input and that can label each test item in isolation were chosen. A baseline Naïve Bayes classifier was implemented and also more advanced SVM classifier. Both were implemented using sklearn packages. Finally, a deep learning model with different hidden layers, varying configurations, and a softmax activation was implemented to compare with the simpler binary classifiers. Additionally, drawing from the literature the importance of textual specificities, experiments were conducted using various pre-processing techniques including POS tagging (implemented through nltk).


## The Dataset

The data used is that from the experiments of Pryzant et al. (2019) and contains the Wikipedia Neutral Point of View corpus. It comprises of text submitted to Wikipedia prior to revision and adaptation for conformity to the NPOV principles  and the revised text. The former makes up the biased instances while the latter, the neutral instances. The data set contains sentences and meta - data and was already tokenised and separated in two distinct documents containing biased and neutral sentence instances. Thus, a script was written to remove the meta – data and label all neutral and biased sentences respectively (i.e. neutral = 0; biased = 1). Given the text-level task one instance is one sentence

[More info on NPOV principles](https://en.wikipedia.org/wiki/Wikipedia:Neutral_point_of_view)

## Conclusion

To summarise some results:
-	The best performing model was a SVM trained on 10000 training instances. This is probably due to the global optimisation capabilities of the model.
-	Text pre-processing, in particular POS tagging substantially increased accuracies throughout the experiments
-	Sentence level classification yielded better results than past document-level performance, and its versatility may be used to extend the model and create an application while also provided insight for the operation of the system in the wild.


## Limitations
This project is also plagued by numerous limitations. The minimal time and scope of this project posed a big restriction on the number of different implementations and experiments that could be made and discussed. Limited knowledge in deep learning proved to be a contributing factor to the poor performance of the neural network models. The task itself is in a way a limitation. As illustrated throughout the paper defining subjectivity is very hard even for humans and often depends on very subtle cues. This may have been amplified by the use of a single corpus, namely the Wikipedia NPOV revisions which on the one hand rarely contain very subjective sentences yet on the other the non-peer reviewed character provides no guarantee on the neutral text fed to the models.


## Future Work
In the future, a BERT model can be implemented to test against the current state of the art in subjectivity classification but on a sentence-level. Furthermore, the models used in the project may be expanded to handle multinomial classification and identify different types of biases identified by Recasens et al. (2013). Future work may focus on improving the data set by perhaps merging it with more obviously biased text such as for example, movie reviews. Finally, the field of subjectivity analysis can benefit from further discussion and linguistic on what subjectivity and bias and how to identify it.


## TO RUN:  

Attempting to identify text that is subjective; A sentence-level subjectivity classification task


- In `main.py` an args script is given, hopefully the "help" parameter straightforward. Through this script you can run experiments using different models, with different training and test data sizes (development data is configured automatically). You can also load the best perfroming SVM model (check paper). Finally, you can select to interactively input a sentence which will then be classified by the selected model. To do this, set the `--user_input` argument to `True`

- `cd` into `src` and type `python3 main.py`

- If you want to run the pretrained SVM model please use the `SVM10000POS.joblib`. To do so set the respective argument in the parser script:  `--load_model` to `True`.```

I encourage you to play around with the argument parser to try out different configurations.
Let me know if you require any further information or if you want to expand on the project. 

I hope you find it interesting!


## References/Sources


Pang, B., & Lee, L. (2004, July). A sentimental education: Sentiment analysis using subjectivity summarization based on minimum cuts. In Proceedings of the 42nd annual meeting on Association for Computational Linguistics (p. 271). Association for Computational Linguistics.

Pryzant, R., Martinez, R. D., Dass, N., Kurohashi, S., Jurafsky, D., & Yang, D. (2019). Automatically Neutralizing Subjective Bias in Text. arXiv preprint arXiv:1911.09709.

Recasens, M., Danescu-Niculescu-Mizil, C., & Jurafsky, D. (2013, August). Linguistic models for analyzing and detecting biased language. In Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) (pp. 1650-1659).

Riloff, E., Wiebe, J., & Phillips, W. (2005, July). Exploiting subjectivity classification to improve information extraction. In AAAI (pp. 1106-1111).

Wiebe, J., Wilson, T., Bruce, R., Bell, M., & Martin, M. (2004). Learning subjective language. Computational linguistics, 30(3), 277-308.

Wilson, T. A. (2008). Fine-grained subjectivity and sentiment analysis: recognizing the intensity, polarity, and attitudes of private states (Doctoral dissertation, University of Pittsburgh).

Wilson, T., Hoffmann, P., Somasundaran, S., Kessler, J., Wiebe, J., Choi, Y., ... & Patwardhan, S. (2005, October). OpinionFinder: A system for subjectivity analysis. In Proceedings of HLT/EMNLP 2005 Interactive Demonstrations (pp. 34-35).

