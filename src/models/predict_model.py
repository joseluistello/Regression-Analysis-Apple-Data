y_pred=ml.predict(x_test)
print(y_pred)

from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


pred_y_df=pd.DataFrame({'Actual Value':y_test,'Predicted Value':y_pred, 'Difference': y_test-y_pred})
pred_y_df[0:20]