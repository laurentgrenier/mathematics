import numpy as np


print("\nvalues")
X = np.array([1,2,3])
Y = np.array([4,5,9])
print("\tX=", X)
print("\tY=", Y)

print("\nmean")
print("\tmean X=", np.mean(X))
print("\tmean Y=", np.mean(Y))

print("\nmedian")
print("\tmedian X=", np.median(X))
print("\tmedian Y=", np.median(Y))

print("\nvariance")
print("\tVx=", X.var())
print("\tVy=", Y.var())

print("\nstandard deviation")
print("\tX.std=", X.std())
print("\tY.std=", Y.std())

print("\nstandard deviation from variance")
print("\tX.std=", np.sqrt(X.var()))
print("\tY.std=", np.sqrt(Y.var()))

print("\ncovariance")
print("\tCov(X,Y)=", np.cov(X,Y, bias=True))
print("\t# keep only the cov(X,Y) value")
print("\tCov(X,Y)=", np.cov(X,Y, bias=True)[0][1])
print("\t# cov(X,X) = X.var()")
print("\tCov(X,X)=", np.cov(X,X, bias=True)[0][1])

print("\npearson correlation coefficient")
print("\tcorrcoef(X,Y)=", np.corrcoef(X, Y))
print("\t# calculation from covariance and standard deviation")
print("\tcorrcoef(X,Y)=", (np.cov(Y,X, bias=True)[0][1] / (X.std() * Y.std())))


