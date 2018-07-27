import pandas as pd 
import numpy as np
from scipy.sparse.linalg import svds

pd.set_option('display.max_columns', 7)  

print "Reading Files"

ratings_df = pd.read_csv('ratings.csv')
movies_df = pd.read_csv('movies.csv')

print "Calculating SVD"

R_df = ratings_df.pivot(index = 'userId', columns = 'movieId', values = 'rating').fillna(0)
R = R_df.values
user_ratings_mean = np.mean(R, axis = 1)
R = R - user_ratings_mean.reshape(-1,1)
U, sigma, Vt = svds(R, k = 50)
sigma = np.diag(sigma)

R_pred = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1,1)
preds_df = pd.DataFrame(R_pred, columns = R_df.columns)

print "Done calculating"

def recommend_movies(predictions_df, userID, movies_df, original_ratings_df, num_recs = 5):
	userRow = userID - 1
	sorted_user_preds = predictions_df.iloc[userRow].sort_values(ascending = False)
	user_data = original_ratings_df[original_ratings_df.userId == userID]
	user_full = (user_data.merge(movies_df, how = 'left', left_on = 'movieId', right_on = 'movieId').
					sort_values(['rating'], ascending = False)
				)
	print 'User {0} has already rated {1} movies.'.format(userID, user_full.shape[0])
	print 'Recommending the highest {0} predicted ratings movies not already rated'.format(num_recs)

	recommendations = (movies_df[~movies_df['movieId'].isin(user_full['movieId'])].
						merge(pd.DataFrame(sorted_user_preds).reset_index(), how = 'left',
							left_on = 'movieId', right_on = 'movieId').
						sort_values(userRow, ascending = False).
						iloc[:num_recs, :-1]
						)
	return user_full, recommendations

def recommend_users(predictions_df, movieID, movies_df, original_df, num_recs = 10):
	sorted_movie_preds = predictions_df.loc[: ,movieID].sort_values(ascending = False)
	sorted_movie_preds = pd.DataFrame(sorted_movie_preds).reset_index()
	sorted_movie_preds.columns = ['userId','rating']
	movie_data = original_df[original_df.movieId == movieID]
	# print movie_data.head()
	movie_full = (movie_data.merge(movies_df, how = 'left', left_on = 'movieId', right_on = 'movieId'))
	# print movie_full.head()
	print 'Movie {0} has been seen by {1} users.'.format(movieID, movie_full.shape[0])
	print 'Finding the {0} most likely users to see this movie'.format(num_recs)
	unseenUsers = pd.DataFrame(original_df[~original_df.userId.isin(movie_data['userId'])].
					userId.unique(), columns = ['userId'])
	recommendations = (unseenUsers.merge(sorted_movie_preds, how = 'left', 
						left_on = 'userId', right_on = 'userId').
					sort_values(by = ['rating'], ascending = False).
					iloc[:num_recs, :1]
					)
	return pd.DataFrame(movie_data.userId, columns = ['userId']), recommendations


seen, recommends = recommend_movies(preds_df, 437, movies_df, ratings_df, 10)
print "This user has seen the following movies:"
print seen.head(10)
print "He would be recommended the following:"
print recommends.head(10)
print ""
print "Now recommending Movies: \n"
users, recommends = recommend_users(preds_df, 318, movies_df, ratings_df, 10)
print "The following users have seen this movie:"
print users.head(10)
print "The most likely users are:"
print recommends.head(10)