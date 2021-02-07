import numpy as np


Q = np.eye( 2,2 )
print(Q)


Qs = [Q, Q]

Q = np.eye( 2,3 )
Qs.append(Q)
print( type(Qs) )
print(Qs)
print( len(Qs) )

print( Qs[-1] )