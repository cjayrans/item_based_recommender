# Item-based Recommendations for Amazon Products
This exercise shows how to create and test variations of item-based recommendation systems. We apply recommenderlab's suite of functions to create and test the accuracy of suggested products based on the ratings of those products based on a scale of 1 to 5. The [Amazon Product Ratings](https://www.kaggle.com/skillsmuggler/amazon-ratings) data set made available on Kaggle, which contains product ratings ranging from May 1996 - July 2014.

Within this exercise we test out three different forms of normalization, as well as three different measures of similarity when performing our k-Nearest Neighbor calculations. We than compare the accuracy of our predictions on our test set using Root Mean Squared Error (RMSE), Mean Squared Error (MSE), and Mean Absolute Error (MAE).
### Normalization Techniques
- None
- Center *(xi - sample mean)*
- Z-score *(xi - sample mean)/standard deviation*

### Measures of Similarity
- Euclidean Distance
- Cosine Similarity
- Pearson Correlation
