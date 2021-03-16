# bias_constraint

https://pereirabarataap.github.io/bias_constraint/index


#### (hopefully, at some point, this will stop driving me insane)


<code>
  from fgboost import FGBClassifier as FGBC
 </code>
 <code>
  clf = FGBC()</br>
 </code>
 <code>
  clf.fit(X_train, y_train, s_train)</br>
</code>
<code>
  y_prob = clf.predict_proba(X_test)[:,1]
</code>
