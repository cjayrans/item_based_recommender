library(data.table); library(recommenderlab); library(DataExplorer); library(arules); library(knitr)

# read in amazon user ratings data
amazon_raw <- read.csv("~/amazon_related_products.csv", stringsAsFactors = FALSE)

# Determine the top 20 ocurring products to accommodate limited file size as this exercise is meant to be an example
top_20 <- setDT(amazon_raw)[,
                            list(
                              num_users = length(Rating > 0)
                            ),
                            by=ProductId]


top_20 <- top_20[order(top_20$num_users, decreasing=TRUE),]
top_20 <- top_20[1:20,]

# Inner join top_20 products with original amazon_raw df
productKeyCols <- c("ProductId")
setkeyv(amazon_raw, productKeyCols)
setkeyv(top_20, productKeyCols)
amazon_sub <- merge(amazon_raw, top_20, by='ProductId')

# Use only ProductId, UserId, and Rating columns
amazon_sub <- amazon_sub[,c('ProductId', 'UserId', 'Rating')]

# Convert the data frame from long to wide format
d1 <- as.data.frame(dcast(as.data.table(amazon_sub), UserId~ProductId, value.var = 'Rating', fun.aggregate = mean))

# Save buyer ids
buyers <- d1$UserId

# Remove UserId column so that we only have ratings
d2 <- d1[,-1]


# Convert data frame to a matrix, then into a recommenderlab::realRatingMatrix
amazon_matrix <- as(as.matrix(d2), 'realRatingMatrix')


# CReate training and testing data sets
rating_matrix <- evaluationScheme(amazon_matrix, method="split", train=0.8, given=-1, goodRating=4)

# Maximize available memory
rm(amazon_raw, amazon_sub, d1, inverseMatrix, tempDf, userProductMatrix)


#### Apply Item-based Collaborative Filtering Cosine measure of similarity
# No normalization 
ibcf_n_c <- Recommender(getData(rating_matrix, "train"), "IBCF",
                        param=list(normalize=NULL, method="Cosine"))

# Center normalization (xi - sample mean) 
ibcf_c_c <- Recommender(getData(rating_matrix, "train"), "IBCF",
                        param=list(normalize="center", method="Cosine"))

# Z-score normalized (xi - sample mean)/standard deviation
ibcf_z_c <- Recommender(getData(rating_matrix, "train"), "IBCF",
                        param=list(normalize="Z-score", method="Cosine"))

# Predict ratings and measure accuracy on our test set
p1 <- predict(ibcf_n_c, getData(rating_matrix, "known"), type="ratings")

p2 <- predict(ibcf_c_c, getData(rating_matrix, "known"), type="ratings")

p3 <- predict(ibcf_z_c, getData(rating_matrix, "known"), type="ratings")


# Since we are not able to restrict the predictions to a certain boundary, we must set predictions that fall outside of our boundary (1,5) to
# their min/max boundary values
p1@data@x[p1@data@x[] < 1] <- 1
p1@data@x[p1@data@x[] > 5] <- 5

p2@data@x[p2@data@x[] < 1] <- 1
p2@data@x[p2@data@x[] > 5] <- 5

p3@data@x[p3@data@x[] < 1] <- 1
p3@data@x[p3@data@x[] > 5] <- 5

# Aggregate performance statistics
error_ICOS <- rbind(
  ibcf_n_c = calcPredictionAccuracy(p1, getData(rating_matrix, "unknown")),
  ibcf_c_c = calcPredictionAccuracy(p2, getData(rating_matrix, "unknown")),
  ibcf_z_c = calcPredictionAccuracy(p3, getData(rating_matrix, "unknown"))
)
kable(error_ICOS)

# Visualize distribution of predicted values for our item-based collaborative filtering using cosine similarity Z-score normalization
hist(as.vector(as(p3, "matrix")), main = "Distrib. of Predicted Values for IBCF Z-Score/Cosine Model", col = "blue", xlab = "Predicted Ratings")

# Remove recommender systems to maximize available memory
rm(ibcf_n_c, ibcf_c_c, ibcf_z_c)




#### Apply Item-based Collaborative Filtering with Euclidean Distance measure of similarity
# No normalization 
ibcf_n_e <- Recommender(getData(rating_matrix, "train"), "IBCF",
                        param=list(normalize=NULL, method="Euclidean"))

# Center normalized 
ibcf_c_e <- Recommender(getData(rating_matrix, "train"), "IBCF",
                        param=list(normalize="center", method="Euclidean"))

# Z-score normalized 
ibcf_z_e <- Recommender(getData(rating_matrix, "train"), "IBCF",
                        param=list(normalize="Z-score", method="Euclidean"))

# Predict ratings and measure accuracy on our test set
p1 <- predict(ibcf_n_e, getData(rating_matrix, "known"), type="ratings")

p2 <- predict(ibcf_c_e, getData(rating_matrix, "known"), type="ratings")

p3 <- predict(ibcf_z_e, getData(rating_matrix, "known"), type="ratings")

# Since we are not able to restrict the predictions to a certain boundary, we must set predictions that fall outside of our boundary (1,5) to
# their min/max boundary values
p1@data@x[p1@data@x[] < 1] <- 1
p1@data@x[p1@data@x[] > 5] <- 5

p2@data@x[p2@data@x[] < 1] <- 1
p2@data@x[p2@data@x[] > 5] <- 5

p3@data@x[p3@data@x[] < 1] <- 1
p3@data@x[p3@data@x[] > 5] <- 5

# Aggregate performance statistics
error_IEUC <- rbind(
  ibcf_n_e = calcPredictionAccuracy(p1, getData(rating_matrix, "unknown")),
  ibcf_c_e = calcPredictionAccuracy(p2, getData(rating_matrix, "unknown")),
  ibcf_z_e = calcPredictionAccuracy(p3, getData(rating_matrix, "unknown"))
)
kable(error_IEUC)

rm(ibcf_n_e, ibcf_c_e, ibcf_z_e)





#### Apply Item-based Collaborative Filtering w/Pearson correlation measure of similarity
# No normalization 
ibcf_n_p <- Recommender(getData(rating_matrix, "train"), "IBCF",
                        param=list(normalize=NULL, method="pearson"))

# Center normalized 
ibcf_c_p <- Recommender(getData(rating_matrix, "train"), "IBCF",
                        param=list(normalize="center", method="pearson"))

# Z-score normalized 
ibcf_z_p <- Recommender(getData(rating_matrix, "train"), "IBCF",
                        param=list(normalize="Z-score", method="pearson"))

# compute predicted ratings
p1 <- predict(ibcf_n_p, getData(rating_matrix, "known"), type="ratings")

p2 <- predict(ibcf_c_p, getData(rating_matrix, "known"), type="ratings")

p3 <- predict(ibcf_z_p, getData(rating_matrix, "known"), type="ratings")

# Since we are not able to restrict the predictions to a certain boundary, we must set predictions that fall outside of our boundary (1,5) to
# their min/max boundary values
p1@data@x[p1@data@x[] < 1] <- 1
p1@data@x[p1@data@x[] > 5] <- 5

p2@data@x[p2@data@x[] < 1] <- 1
p2@data@x[p2@data@x[] > 5] <- 5

p3@data@x[p3@data@x[] < 1] <- 1
p3@data@x[p3@data@x[] > 5] <- 5

# Aggregate performance statistics
error_IPC <- rbind(
  ibcf_n_p = calcPredictionAccuracy(p1, getData(rating_matrix, "unknown")),
  ibcf_c_p = calcPredictionAccuracy(p2, getData(rating_matrix, "unknown")),
  ibcf_z_p = calcPredictionAccuracy(p3, getData(rating_matrix, "unknown"))
)
kable(error_IPC)

# When comparing all nine Recommender models against one another we find that of the Item-based Recommender systems, our model that measures similarity
# using Euclidean distance with Z-score normalization outperforms our other 8 models
kable(error_ICOS) # Cosine similarity 
kable(error_IEUC) # Euclidean Distance 
kable(error_IPC) # Pearson Correlation 