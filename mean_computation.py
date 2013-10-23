import numpy as np
from numpy.linalg import solve
issues_in_public_fold_test_data = 0.3 * 149576 # Approximate
n = 3 * issues_in_public_fold_test_data

# Data from spreadsheet
B = np.array([[0.405465108108164,   3.71357206670431,    1.38629436111989],
              [0.095310179804325,   2.28238238567653,    0.587786664902119],
              [0,                   1.02961941718116,    0.832909122935104]])
# MS = RMS^2
A = 0.5218328644
MS = np.array([4.03809025,
      1.4300572225,
      0.3742625329])

z = np.array([np.mean(r[0:3]**2) for r in B])
y = 3*A/2 - 3*MS/2 + z/2

x = solve(B, y)
