### Introduction
The motivation of this project is to build the recommenders system using Apache Spark about user's book preference. Recommender systems uses data models in order to predict certain behaviour. The trained model will compute recommendations for a given user as well as compute similar or related items for a given item.

### Dataset
The dataset used for this project is the [goodbooks-10k](https://www.kaggle.com/zygmunt/goodbooks-10k) and was obtained on Kaggle.
This dataset contains six million ratings for ten thousand most popular books. 
The dataset contains several csv files. The file book.csv contains ten thousand book entries with a number of labels: various IDs, ISBN, author(s), original title, original publication year, language, books count, various ratings etc. In addition to the books.csv, there are four more csv files (book_tags, ratings, tags, to_read) , representing additional information, that are bounded by one of a primary key (book_id, user_id, tag_id).

### Recommendation
The following three approaches were applied in order to do the recommendations:
* ALS (Alternating Least Squares): Its an approach based on collaborative filtering which is a technique based on mapping a user item association using a user-item matrix. ALS tries to find latent factors for items and users and those factors are used to predict the rating of a given item by a given user. So the books which are predicted to be of higher rating for a given user, can be recommended to that person.
* KMeans : K-means clustering is a type of unsupervised learning and its goal is to find groups in the data, with the number of groups represented by the variable K. The algorithm works iteratively to assign each data point to one of K groups based on the features that are provided. So in this case, it means grouping books into different clusters based on some feature.
* Deep Learning model : Deep learning model was used as one of the alternative models to predict the rating by giving user-id and book-id as features.


