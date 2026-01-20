# Logistic-Regression-Interview
Logistic regression is a statistical model in which the target variable/dependent variable takes a discrete value and independent variables can be either to be continuious or discrete. It is a supervised learning algorithm. It can solve classification problems.
In logistic regression, we are essentially trying to find the weights that will convert our input data(independent variables) into predictions. The logistic function or sigmoid function is used to ensure these predictions are in the range of 0 to 1. The sigmoid function maps any real-valued number into another value between 0 and 1. In the case of logistic regression, it converts the linear regression output to a probability.

* Logistic Regression is a classification algorithm used when the target variable is binary, such as 0 or 1 and Yes or No.
* Logistic Regression starts exactly like linear regression. We compute linear combination of inputs(input variables)
        z = m1X1 + m2X2 + .....mnXn + C
  The problem is
        * This z value can range from -Infinity to +Infinity
        * But for classification, we need a probability between 0 and 1.
* To convert the linear output (z) into probability, we apply the sigmoid function.
        Sigmoid(z) = 1/1+e-z
  Sigmoid function converts the linear output into a probability value based 0 and 1.
* After sigmoid function we get probability, but probability is bounded, and small change in probability near 0 and 1 do not reglect changes in risk symmetrically.
* To overcome this, we convert probability into odds, which represent how likely an event is to occur compared to it not occuring.
* Odds are defined as the ratio of the probability of success to the probability of failure.
* But odds are range from 0 to infinity.
* To make suitable for linear modelling, we take log-odds or the logit function. This log-odds can take any real value from negative to positive infinity.
* Logistic regression assumes that this log-odds value is a linear combination of input features.
* so far we have the model structure, but the coefficients are still unknown.
* We convert log-odds back into a probability using the sigmoid function, This gives the model's predicted probability for each data point.
* The model checks how well these probabilities match the actualclass labels in the training data. If the actual label is 1, the model should give a high probability; if the label is 0, the model should give a low probability.
* Logistic regression adjusts its coefficients so that correct predictions get higher probability and incorrect predictions get lower probability.
* This adjustment is done using Maximum Likelihood Estimation, which means choosing coefficient values that make the observed labels most likely.
* The optimization is performed using gradient descent.
* Once training is complete, the model outputs probabilities, which are converted into class labels using a threshold such as 0.5.

# Logistic Regression Interview Questions

# 1. What is log loss in logistic regression?
   
   Log loss, also called cross-entroy loss, is the cost function used in logistic regression to measure how well the predicted probabilites match    the actual class labels.
   It penalizes wrong predictions, especially when the model is very confident but incorrect. Lower log loss means better model performance.

# 2. How does logistic regression differ from linear regression?
   
   There are several differences between the logistic regression and linear regression models. One major difference is that logistic regression      is useful for solving classification problems, while linear regression is helpful for resolving regression problems.
   You can predict the value of categorical variables in logistic regression, while you can predict the value of continuous variables in linear      regression. You can classify samples by finding the S-shaped sigmoid curve in logistic regression, while you can predict the continuous value     output by finding the best-fitted line in linear regression.
   In logistic regression we use maximum likelihood estimation to calculate the loss function.
   In linear regression we use mean squared error for calulating the loss function.

# 3. How does the softmax and sigmoid functions differ from each other?
   
   The softmax function, also known as the normalised exponential function or softargmax, it is useful in the multinomial logistic regression        model to classify the multiple classes.
   The sigmoid function is a mathematical function is a mathematicl funciton that forms an S-shaped curve, known as sigmoid curve, on a graph. It    applies to the binary logistic regression mdoel in multi-label classification.

# 4. What is regularisation and why is it important?
   
   Regularisation is a technique that is used to avoid the problem of overfitting in logistic regression, it is important because it modifies        learning algorithms to enable them to generalise better on usneen data and reduce their generalisation errors.

# 5. How does logistic regression perfrorm feature selection?
   
   Logistic regression naturally implements feature selection through L1 regularization(LASSO) which introduces sparsity.
   L1 regularisation adds a penatly term to the loss function that is the absolute value of the coefficents. This tends to shrink coefficients to    zero, effectively removing the associated fetures. Regularization strength is typically controled by the parameter Lambda.

# 6. Why is it called regression if it is a classification algorithm?
    
   Because it models the log-odds(logit) as a linear combination of input features, similar to linear regression.

# 7. What is the decision boundary in logistic regression?
    
   It is the threshold(usually 0.5) above which we predict class 1 and below which class 0.

# 8. What is log-odds(logit)?
   Log-odds is the logarithm of the odds ratio
   i.e log(p/1-p)
   it converts probabilities into a linear scale.

# 9. What are the assumptions of Logistic regression?
    
    - Binary dependent variable.
    - Indpendence of observations
    - No multicollinearity.
    - Linear relationship between predictors and log-odds.

# 10. How do you handle multicollinearity?
    
    - Remove correlated features
        When two or more indenpendent variables are highly correlated, logistic regression faces
        - Unstable coefficients
        - Hard to interpret restuls.
        If two features give almost the same information, keep only one.
        Used correlation matrix by visualising in heatmap
    - Use VIF
        VIF measures how much a feature is explained by other features.
        VIF quantify multicollinearity and iteratively remove features with high VIF.
                  VIF = 1/1-R^2
       from statsmodels.stats.outliers_influence import variance_inflation_factor      
    - Apply L1 or L2 regularization.
        L1 adds an absolute penalty to coefficients
            Loss + lambda(summation(|w1 + w1 + .. wn|)
        - shrinks coefficients
        - can make coefficients zero
        - peforms feature selection.
        When to use L1 Regularisation
        - High-dimensional data
        - Feature selection is needed.
        - Interpretability is important
       L2 adds a penatly to squared coefficients
            Loss + lambda(summation(w1^2 + w2^2 +.... wn^2))
        - shrinks coefficients
        - dont make them zero
       When to use L2 Regularisation
        - When all features are important
          
   
