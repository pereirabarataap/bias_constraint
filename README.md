# bias_constraint

https://pereirabarataap.github.io/bias_constraint/index


#### (hopefully, at some point, this will stop driving me insane)


<code>
  from fgboost import FGBClassifier as FGBC
  clf = FGBC()
  clf.fit(X_train, y_train, s_train)
  y_prob = clf.predict_proba(X_test)[:,1]
</code>
