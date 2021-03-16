# bias_constraint

https://pereirabarataap.github.io/bias_constraint/index


#### (hopefully, at some point, this will stop driving me insane)

<pre><code>
from fgboost import FGBClassifier as FGBC
clf = FGBC()</br>
clf.fit(X_train, y_train, s_train)</br>
y_prob = clf.predict_proba(X_test)[:,1]
</code></pre>
<code>
  from fgboost import FGBClassifier as FGBC
 </code>
 </br>
 <code>
  clf = FGBC()</br>
 </code>
 </br>
 <code>
  clf.fit(X_train, y_train, s_train)</br>
</code>
</br>
<code>
  y_prob = clf.predict_proba(X_test)[:,1]
</code>
</br>
