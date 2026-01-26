If you want diagonal element from matrix $A x B$, you can use 2 method (or above) to select them by:

- `np.diag( A @ B )` (High time consume, but easy reading) 
- `np.sum( A * B.T, axis = 1)` (Low time consume, but hard reading)

Where $A$ is $m$ x $n$ matrix, $B$ is $n$ x $m$ matrix, and $C$ is $m$ x $1$ matrix (column vector).

How??? You can proof it!
