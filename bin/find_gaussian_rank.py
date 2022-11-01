import sys
import os
print(os.getcwd())
sys.path.extend([os.getcwd()])
print(sys.path)

import gaussianstates.gaussianrank.gaussian_rank_magic as g_rank

# ~~~~~~~~~~~~ Parameters controlling script ~~~~~~~~~~~~~~
n = 1
should_overwrite = False
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

g_rank.main()
