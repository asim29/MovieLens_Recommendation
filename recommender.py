import pandas as pd 
import numpy as np
from scipy.sparse.linalg import svds

pd.set_option('display.max_columns', 7)  

ratings_df = pd.read_csv('ratings.csv')
movies_df = pd.read_csv('movies.csv')

R_df = ratings_df.pivot(index = 'userId', columns = 'movieId', values = 'rating').fillna(0)
R = R_df.values
user_ratings_mean = np.mean(R, axis = 1)
R = R - user_ratings_mean.reshape(-1,1)
U, sigma, Vt = svds(R, k = 50)
sigma = np.diag(sigma)

R_pred = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1,1)
preds_df = pd.DataFrame(R_pred, columns = R_df.columns)

def recommend_movies(predictions_df, userID, movies_df, original_ratings_df, num_recs = 5):
	userRow = userID - 1
	sorted_user_preds = predictions_df.iloc[userRow].sort_values(ascending = False)
	print type(sorted_user_preds)
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


seen, recommends = recommend_movies(preds_df, 437, movies_df, ratings_df, 10)
