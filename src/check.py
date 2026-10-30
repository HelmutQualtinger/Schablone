import pickle, sys, numpy as np
sys.path.insert(0, 'src')
from stencil_geom import build, report
res = build(); report(res)
pickle.dump(res, open(sys.argv[1], 'wb'))
