from VideoClassification.utils.Logger import Logger

import math

log = Logger("/tmp/runs/r3")

for step in range(10):
    v1 = math.sin(step)
    v2 = math.cos(step)
    v3 = 2000*step
    log.scalar_summary('v1',v1,step)
    log.scalar_summary('v2',v2,step)
    log.scalar_summary('v3',v3,step)

