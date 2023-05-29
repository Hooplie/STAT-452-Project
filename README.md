# STAT 452 Project - Statistical Learning and Prediction

The goal of this project is to obtain the best prediction possible using techniques and models learned in the course. A csv file given by the professor containing data of p explanatory variables and one response variable is used to train the model. After the model is selected a different csv file containing explanatory variables without the response variable is used for the final prediction. The project is a competition and my prediction score ranked top 5% in the class.

I use a 10-fold CV to evaluate and compare my models. I fit each model on all explanatory variables and mainly keep their default parameters except for NNET which I tune a little within the 10-fold CV. After fitting each model, I calculate their MSPE’s and store them in a table containing the MSPE’s for each of the 10 folds along with the last column being on the full data. I look for the model which had the smallest MSPE on the full data and look at the MSPE and RMSPE boxplots before selecting the best model to use. The result of the initial 10-fold CV is that an untuned Boosting model performed the best compared to the other models and is the model I select for tuning. Below is my MSPE table, MSPE and RMSPE boxplots supporting my findings.

![image](https://github.com/Hooplie/STAT-452-Project/assets/78288771/daf74b4b-985d-458e-9947-1c7cbcabc927)
