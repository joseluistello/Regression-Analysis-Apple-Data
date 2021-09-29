appledf.corr()


x=appledf.drop(['revenue'], axis=1).values
y=appledf['revenue'].values


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.3, random_state=0)

from sklearn.linear_model import LinearRegression
ml=LinearRegression()
ml.fit(x_train,y_train)