# CART
Project: Predict whether customer will take personal loan or not, based on various demographics and behavioural variables.


Key Findings: By targeting top four decile, having the maximum KS Statistics of 0.62, I can target these customers with a probability of 0.28 for cross selling Personal loan to them. On initial fitting of model accuracy on training set was 89% and test set was 79%. From these details we can observe that the model is overfitting. Therefore, to improve the results, model was subjected to cross validation. However, there was not a sufficient improvement in results, the model was still overfitting. Therefore, to improve the results, grid search cross validation was applied on the model. Post grid search, the model accuracy on training set was at ~82% and that on test set was ~75%, and also the overfitting problem was reduced significantly.
