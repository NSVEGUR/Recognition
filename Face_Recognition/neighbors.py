import numpy as np

class KNeighborsClassifier():
		def __init__(self, n_neighbors):
				self.n_neighbors = n_neighbors
				
		def fit(self, X, y):
				self.X_train = X
				self.y_train = y
				
		def distance(self, X1, X2):
				distance = np.linalg.norm(X1 - X2)
				return distance
		
		def predict(self, X_test):
				final_output = []
				for i in range(len(X_test)):
						d = []
						outputs = []
						for j in range(len(self.X_train)):
								dist = self.distance(self.X_train.iloc[j] , X_test.iloc[i])
								d.append([dist, j])
						d.sort()
						d = d[0:self.n_neighbors]
						for d, j in d:
								outputs.append(self.y_train.iloc[j])
						results = []
						ans = max(set(outputs), key = outputs.count)
						final_output.append(ans)
						
				return final_output
		
		def score(self, X_test, y_test):
				predictions = self.predict(X_test)
				return (predictions == y_test).sum() / len(y_test)

