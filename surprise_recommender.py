import pandas as pd 
import numpy as np
from scipy.sparse.linalg import svds
from time import time
from collections import defaultdict
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import GridSearchCV

pd.set_option('display.max_columns', 7)  

print "Reading Files"

ratings_df = pd.read_csv('ratings.csv')
movies_df = pd.read_csv('movies.csv')

ratings_df = ratings_df.drop(columns = ['timestamp'])
reader = Reader(rating_scale = (0.5,5))
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)

# param_grid = {'n_epochs':[5, 10], 'lr_all':[0.002, 0.005], 'reg_all':[0.4, 0.6]}
# gs = GridSearchCV(SVD, param_grid, measures = ['rmse', 'mae'], cv = 3)

# gs.fit(data)
# print gs.best_score['rmse']
# print gs.best_params['rmse']

def recommend_movies(predictions, n = 10):
	top_n = defaultdict(list)
	#map predictions to each user
	for uid, iid, true_r, est, _ in predictions:
		top_n[uid].append((iid, est))

	#sort predictions to each user
	for uid, user_ratings in top_n.items():
		user_ratings.sort(key = lambda x: x[1], reverse = True)
		top_n[uid] = user_ratings[:n]

	return top_n

def recommend_users(predictions, n = 10):
	top_n = defaultdict(list)

	for uid, iid, true_r, est, _ in predictions:
		top_n[iid].append((uid, est))

	for iid, user_ratings in top_n.items():
		user_ratings.sort(key = lambda x: x[1], reverse = True)
		top_n[iid] = user_ratings[:n]

	return top_n

trainset = data.build_full_trainset()
algo = SVD()
t0 = time()
print "Fitting"
algo.fit(trainset)
testset = trainset.build_anti_testset()
print "done in %0.3fs" % (time() - t0)

print "Predicting"
t0 = time()
predictions = algo.test(testset)
print "done in %0.3fs" % (time() - t0)

# print predictions[0][0]
t0 = time()
top_n_movies = recommend_movies(predictions, n = 10)
print "done in %0.3fs" % (time() - t0)

usermovies = {}
for uid, user_ratings in top_n_movies.items():
	usermovies[uid] = ([iid for (iid,_) in user_ratings])

usermovies_df = pd.DataFrame(usermovies)	
print usermovies_df
print ""

t0 = time()
top_n_users = recommend_users(predictions, n = 10)
print "done in %0.3fs" % (time() - t0)

movieusers = {}
for iid, user_ratings in top_n_users.items():
	movieusers[iid] = ([uid for (uid,_) in user_ratings])


movieusers_df = pd.DataFrame(movieusers)	
print movieusers_df
print ""
