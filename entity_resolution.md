## Homework 8 - Entity Resolution

What is entity resolution?
* Entity resolution is relating and linking data between one or more datasets using machine learning. It is particularly useful when combining two or more separate datasets.

Relate every single week's work up to now to entity resolution
* Week 1: This week we discussed GitHub and its importance in the programming sphere. Having an example of entity resolution to showcase on your GitHub is a good way to stand out to employers. Also, we first discussed features and labels this week which are crucial in creating classifiers for entity resolution models.

* Week 2: Week 2 was the first week we worked with the articles and grants datasets. Pulling out the individual 'grantees' from the PI_NAMEs column was crucial in our entity resolution problem because it allowed us to match article authors to grantees.

* Week 3: The human activity regogntion data was introduced this week. We were tasked with building a classifier to predict sleep. The machine learning concept of building classifiers is crucial to the entity resolution problem. In the case of the phonebook problem, I would need to build a classifier to predict if people from two different phonebooks are a match or not. Features and lables would be designed based on the existing data of forename, surname, phone number, and address. Model choice is another aspect to this. Should I choose a more complex black-box model? Or a less complex open-box model? 

* Week 4: Natural language processing techniques like fasttext are extremely helpful in entity resolution. Word2Vec and fasttext take words and vectorize them. Vectors with smaller distances between them represent more similar words. Entity resolution with text relies on turning words to vectors to quantify some form of similarity. Because not all potential matches will be 100% matches, finding the similarity between words allows us to still find these matches that aren't exact.

* Weeks 5/6: These weeks are when we actually began to discuss entity resolution directly. Creating and simulating training data is crucial to developing an effective model (obviously classifiers need training data). With large datasets, however, you cannot simply compare every entry in one dataset to every entry in the other and see if they match. That is inefficient and will take forever. Limiting matching by doing it only on a subset of the data solves this problem.

* Week 7/8: Because entity resolution is often done between (very) large datasets, databases are necessary to store and organize the data. The actual entity resolution classification should be done in pandas, but the data can be pass from the database into pandas, and then after, the new matched data can b passed back into SQL.

You should be able to relate the human activity recognition dataset to entity resolution. Not directly, but rather you should understand that we were learning a core conept about machine learning. What was it?
* The core machine learning concept learned here was classification, which is crucial to machine learning and entity resolution specifically.

