import pandas as pd
import numpy as np
from util import image
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

class Data():

	def __init__(self):
		pass

	def import_data(self,name):
		path=("dataset/"+name+".csv")
		if not os.path.isfile(path):
			print("Archivo no encontrado")
			return
		else:
			self.data=pd.read_csv(path)
			print(self.data.head(5))
			return

	def correlation(self,target="result"):
		corr_matrix = self.data.corr()
		print("Correlacion con respecto a "+target)
		print(corr_matrix[target].abs().sort_values(ascending=False))


	def corr_matrix(self,labels="All",save=False):
		if labels=="All":
			corre=self.data.corr()
			labels=corre["result"].abs().sort_values(ascending=False)
		pd.plotting.scatter_matrix(self.data[labels.index[:]])
		if save==True:
			image.save("Corr_Matrix")
			print("Imagen guarda con exito")
		plt.show(block=False)

	def encoder(self,target):
		for tag in target:
			cat_encoder = OneHotEncoder(sparse=False)
			encoded=cat_encoder.fit_transform(self.data[[tag]])
			categories=[]
			for cat in cat_encoder.categories_[0]:
				categories.append(tag+"="+str(cat));
			encoded=pd.DataFrame(encoded,columns=categories,dtype=np.int8)
			self.data=self.data.drop([tag],axis=1)
			self.data=pd.concat([self.data,encoded],sort=True,axis=1)
		print(self.data)

	def prepare(self,imputer="median"):
		num_pipeline = Pipeline([
	        ('imputer', SimpleImputer(strategy=imputer)),
	        ('std_scaler', StandardScaler()),
		])
		train=self.train_set.drop("result",axis=1)
		test=self.test_set.drop("result",axis=1)
		train_prep=num_pipeline.fit_transform(train)
		test_prep=num_pipeline.fit_transform(test)
		return train_prep, test_prep

	def split(self,porc=0.2,random=42):
		self.train_set, self.test_set=train_test_split(self.data, test_size=porc, random_state=random)
		return self.train_set, self.test_set
