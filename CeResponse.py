# Compare upstream and downstream resolution at tracker mid for e- reflections
#
import awkward as ak
import behaviors
from matplotlib import pyplot as plt
import uproot
import numpy as np
from scipy.optimize import curve_fit
import math
from scipy import special

def fxn_expGauss(x, amp, mu, sigma, lamb):
    z = (mu + lamb*(sigma**2) + x)/(np.sqrt(2)*sigma)
    comp_err_func = special.erfc(z)
    val = amp*(lamb/2)*((math.e)**((lamb/2)*(2*mu+lamb*(sigma**2)+2*x)))*comp_err_func
    return val

#file = [ "/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/00/50/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000776.root" ]
file = [ "/Users/brownd/data/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000776.root" ]
f03files = [
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/00/50/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000776.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/00/5b/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000238.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/00/99/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000849.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/00/eb/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000721.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/01/0a/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000715.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/01/46/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000007.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/01/68/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000510.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/01/cd/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000289.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/01/dd/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000669.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/02/03/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000333.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/02/16/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000348.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/02/a2/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000416.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/02/e4/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000756.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/03/36/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000892.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/03/60/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000273.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/03/db/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000863.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/04/52/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000000.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/04/54/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000854.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/04/b9/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000140.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/04/cb/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000848.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/05/3f/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000411.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/05/46/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000114.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/05/5a/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000571.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/05/83/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000451.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/06/0a/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000432.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/06/c8/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000169.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/08/09/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000352.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/08/6b/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000852.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/08/ba/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000546.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/08/bc/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000376.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/08/ca/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000010.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/08/dc/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000796.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/08/e7/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000588.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/09/89/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000361.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/09/a7/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000180.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/0a/90/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000541.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/0a/ae/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000590.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/0a/b6/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000598.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/0a/d4/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000827.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/0b/16/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000132.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/0b/4c/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000072.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/0b/a3/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000789.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/0b/a9/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000716.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/0b/e4/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000613.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/0c/06/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000757.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/0c/0d/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000497.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/0c/4b/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000504.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/0c/bd/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000303.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/0d/43/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000718.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/0d/53/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000556.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/0e/9d/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000595.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/0e/b8/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000749.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/0f/38/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000201.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/10/5c/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000788.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/10/74/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000662.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/10/fa/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000022.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/11/25/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000895.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/11/39/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000652.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/11/55/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000375.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/11/7c/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000477.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/12/79/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000643.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/12/a6/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000059.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/12/b7/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000780.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/12/f0/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000254.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/13/57/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000173.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/13/5a/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000864.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/13/86/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000237.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/13/9c/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000148.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/13/c5/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000682.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/13/de/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000326.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/14/14/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000360.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/14/60/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000347.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/14/c4/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000543.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/15/5e/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000366.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/15/6e/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000203.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/17/51/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000610.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/17/b3/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000244.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/18/0f/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000359.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/18/c8/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000311.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/18/db/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000831.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/19/11/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000592.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/19/2c/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000653.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/19/2f/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000811.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/19/7f/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000531.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/19/a8/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000575.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/19/ee/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000043.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/1a/35/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000298.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/1a/8d/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000528.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/1a/9b/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000551.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/1a/c6/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000464.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/1b/ae/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000344.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/1c/14/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000607.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/1c/c3/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000663.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/1c/c6/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000395.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/1d/19/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000384.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/1d/46/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000482.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/1e/08/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000862.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/1e/16/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000174.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/1e/bb/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000840.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/1e/cf/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000103.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/1f/37/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000107.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/00/6b/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000006.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/00/74/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000504.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/00/b5/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000179.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/00/d0/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000386.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/01/1f/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000635.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/01/77/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000801.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/01/7b/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000116.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/01/ce/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000397.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/01/d3/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000622.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/01/eb/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000289.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/01/f4/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000641.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/02/4c/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000768.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/02/62/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000189.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/02/c3/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000273.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/03/1e/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000299.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/03/26/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000200.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/03/2a/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000221.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/03/a1/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000835.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/03/a5/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000222.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/05/08/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000543.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/05/4f/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000240.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/05/67/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000594.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/05/a2/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000542.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/05/eb/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000787.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/06/7f/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000360.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/06/ab/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000708.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/07/35/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000693.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/07/71/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000924.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/07/c4/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000126.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/08/f9/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000018.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/09/18/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000603.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/09/30/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000661.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/09/68/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000415.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/09/c1/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000052.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/0a/88/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000494.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/0a/ac/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000713.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/0a/d4/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000254.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/0b/24/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000894.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/0b/2e/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000076.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/0b/7d/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000546.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/0c/38/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000536.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/0c/48/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000694.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/0c/7a/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000669.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/0c/ae/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000979.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/0c/fa/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000934.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/0e/56/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000612.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/0e/8d/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000383.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/0e/a2/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000255.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/0e/a4/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000763.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/0e/d3/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000711.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/0f/76/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000811.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/0f/9f/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000483.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/10/18/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000183.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/10/25/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000127.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/10/6a/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000568.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/10/b2/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000827.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/11/20/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000357.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/11/a4/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000608.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/11/ac/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000918.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/11/dc/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000754.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/11/e6/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000500.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/12/39/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000819.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/12/3a/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000382.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/13/29/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000985.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/13/ad/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000804.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/13/ae/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000794.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/13/ca/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000338.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/14/1d/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000155.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/14/87/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000108.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/14/bc/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000284.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/15/c9/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000738.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/15/d5/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000323.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/16/29/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000621.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/16/39/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000327.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/16/72/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000490.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/16/b9/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000402.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/17/82/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000781.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/17/b8/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000660.root",
#"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/17/d2/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000675.root",
"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/17/e1/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000657.root",
"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/17/fa/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000412.root",
"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/17/fe/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000836.root",
"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/18/89/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000503.root",
"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/18/9d/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000566.root",
"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/18/c8/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000991.root",
"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/18/e9/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000376.root",
"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/19/64/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000774.root",
"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/19/79/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000132.root",
"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/19/be/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000737.root",
"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/19/e9/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000980.root",
"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/19/ee/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000620.root",
"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/1a/06/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000489.root",
"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/1a/15/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000050.root",
"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/1a/6e/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000040.root",
"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/1a/7e/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000312.root",
"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/1a/a3/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000049.root",
"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/1b/1a/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000720.root",
"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/1b/24/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000119.root",
"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/1b/42/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000112.root",
"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/1b/f8/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000390.root",
"/data/HD5/mu2e/MDC2020/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/1c/3e/nts.mu2e.CeEndpointOnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000316.root"
]
mbfiles = [
"/Users/brownd/data/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000000.root",
"/Users/brownd/data/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000001.root",
"/Users/brownd/data/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000002.root",
"/Users/brownd/data/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000003.root",
"/Users/brownd/data/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000004.root",
"/Users/brownd/data/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000005.root",
"/Users/brownd/data/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000006.root",
"/Users/brownd/data/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000007.root",
"/Users/brownd/data/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000009.root",
"/Users/brownd/data/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000776.root"]
# setup general constants
minNHits = 20
minFitCon = 1.0e-5
ibatch=0
elPDG = 11
# momentum range around a conversion electron
cemom = 104
dmom = 20
minMom = cemom - dmom
maxMom = cemom + dmom
# Surface Ids
TrkSID = [None]*3
TrkSID[0] = 0 # entrance
TrkSID[1] = 1 # mid
TrkSID[2] = 2 # exit
TrkLoc = [None]*3
TrkLoc[0] = "Tracker Entrance"
TrkLoc[1] = "Tracker Middle"
TrkLoc[2] = "Tracker Exit"
FigX =[None]*3
FigX[0] = 0.05
FigX[1] = FigX[0] + 1.0/3.0
FigX[2] = FigX[1] + 1.0/3.0
# Momentum and response
Mom = []
MCMom = []
MomReso = []
MomResp = []
originMomMC = []
elPDG = 11
TSDASID = 96 # Exit of DS

for isid in range(len(TrkSID)) :
    Mom.append([])
    MomReso.append([])
    MomResp.append([])
    originMomMC.append([])

for batch,rep in uproot.iterate(f03files,filter_name="/trk|trksegs|trkmcsim|trksegsmc/i",report=True):
    print("Processing batch ",ibatch)
    ibatch = ibatch+1
    segs = batch['trksegs'] # track fit samples
    nhits = batch['trk.nactive']  # track N hits
    fitcon = batch['trk.fitcon']  # track fit consistency
    trkMC = batch['trkmcsim']  # MC genealogy of particles
    segsMC = batch['trksegsmc'] # SurfaceStep infor for true primary particle
    Segs = segs[:,0] # assume the 1st track is the Ce
    SegsMC = segsMC[:,0] # upstream track MC truth
    TrkMC = trkMC[:,0] # upstream fit associated true particles
    FitCon = fitcon[:,0]
    Nhits = nhits[:,0]
    goodFit = (Nhits >= minNHits) & (FitCon > minFitCon)
    goodTrkMC = (TrkMC.pdg == elPDG) & (TrkMC.trkrel._rel == 0)
    TSDASeg = Segs[Segs.sid == TSDASID]
    noTSDA = ak.num(TSDASeg)==0
#    print(noTSDA)
    originMomMC = TrkMC[goodTrkMC].mom.magnitude()
    # basic consistency test
    assert((len(Segs) == len(SegsMC)) & (len(Segs) == len(TrkMC)) & (len(Nhits) == len(Segs)) & (len(originMomMC) == len(Segs)))
    originMomMC = originMomMC[(originMomMC>minMom) & (originMomMC < maxMom)]
    omomMC = ak.flatten(originMomMC,axis=1)
    MCMom.extend(omomMC)
    hasOriginMomMC = ak.count_nonzero(originMomMC,axis=1,keepdims=True)==1
    hasOriginMomMC = ak.flatten(hasOriginMomMC,axis=1)
#    print(hasOriginMomMC[0:10])
#    print(originMomMC[0:10])
    # sample the fits at 3 tracker locations
    for isid in range(len(TrkSID)) :
        sid = TrkSID[isid]
        segs = Segs[Segs.sid == sid]
        mom = segs.mom.magnitude()
        mom = mom[(mom > minMom) & (mom < maxMom)]
        hasmom= ak.count_nonzero(mom,axis=1,keepdims=True)==1
        hasmom =ak.flatten(hasmom,axis=1)
        segsMC = SegsMC[(SegsMC.sid == sid) & (SegsMC.mom.z() > 0.0) ]
        momMC = segsMC.mom.magnitude()
        hasMC = ak.count_nonzero(momMC,axis=1,keepdims=True)==1
        hasMC = ak.flatten(hasMC,axis=1)
        select = goodFit & hasmom & hasMC & hasOriginMomMC & noTSDA
        mom = mom[select]
        momMC = momMC[select]
        mom = ak.flatten(mom,axis=1)
        momMC = ak.flatten(momMC,axis=1)
        assert(len(mom) == len(momMC) )
        Mom[isid].extend(mom)
        momreso = mom - momMC
        MomReso[isid].extend(momreso)
        omomMC = originMomMC[select]
        omomMC = ak.flatten(omomMC,axis=1)
        momresp = mom - omomMC
        MomResp[isid].extend(momresp)
    # plot Momentum
nDeltaMomBins = 200
nMomBins = 100
momrange=(minMom,107)
momresorange=(-2.5,2.5)
momresprange=(-10,5)
#print(len(Mom[0]),Mom[0:10][0])
#print(len(MomReso[0]),MomReso[0:10][0])
momVal = [None]*3
momReso = [None]*3
momResp = [None]*3
momRespFit = [None]*3
MomRespHist = [None]*3

fig, (momVal[0], momVal[1], momVal[2], mcMom) = plt.subplots(1,4,layout='constrained', figsize=(10,5))
mcMom.hist(MCMom,label="MC Origin", bins=nMomBins, range=momrange, histtype='step')
mcMom.set_xlabel("MC Momentum (MeV)")
mcMom.set_title("MC Origin Momentum")
for isid in range(len(TrkSID)) :
    momVal[isid].hist(Mom[isid],label=TrkLoc[isid], bins=nMomBins, range=momrange, histtype='step')
    momVal[isid].set_xlabel("Fit Momentum (MeV)")
    momVal[isid].set_title(TrkLoc[isid])

fig, (momResp[0], momResp[1], momResp[2]) = plt.subplots(1,3,layout='constrained', figsize=(10,5))
for isid in range(len(TrkSID)) :
    MomRespHist[isid] =  momResp[isid].hist(MomResp[isid],label=TrkLoc[isid], bins=nMomBins, range=momresprange, histtype='step')
    momResp[isid].set_xlabel("Fit - MC Origin Momentum (MeV)")
    momResp[isid].set_title(TrkLoc[isid])

fig, (momReso[0], momReso[1], momReso[2]) = plt.subplots(1,3,layout='constrained', figsize=(10,5))
for isid in range(len(TrkSID)) :
    momReso[isid].hist(MomReso[isid],label=TrkLoc[isid], bins=nMomBins, range=momresorange, histtype='step')
    momReso[isid].set_xlabel("Fit - MC Momentum (MeV)")
    momReso[isid].set_title(TrkLoc[isid])
#
# response function fit
#
fig, (momRespFit[0],momRespFit[1],momRespFit[2]) = plt.subplots(1,3,layout='constrained', figsize=(10,5))
for isid in range(len(TrkSID)) :
    momRespHist = MomRespHist[isid]
    momRespErrors = np.zeros(len(momRespHist[1])-1)
    momRespBinMid = np.zeros(len(momRespHist[1])-1)
    for ibin in range(len(momRespErrors)):
        momRespBinMid[ibin] = 0.5*(momRespHist[1][ibin] + momRespHist[1][ibin+1])
        momRespErrors[ibin] = max(1.0,math.sqrt(momRespHist[0][ibin]))
    momRespIntegral = np.sum(momRespHist[0])
# initialize the fit parameters
    mu_0 = np.mean(momRespBinMid*momRespHist[0]/momRespIntegral) # initial mean
    var = np.sum(((momRespBinMid**2)*momRespHist[0])/momRespIntegral) - mu_0**2
    sigma_0 = np.sqrt(var) # initial sigma
    lamb_0 = sigma_0 # initial exponential (guess)
    binsize = momRespHist[1][1]-momRespHist[1][0]
    amp_0 = momRespIntegral*binsize # initial amplitude
    p0 = np.array([amp_0, mu_0, sigma_0, lamb_0]) # initial parameters
# fit, returing optimum parameters and covariance
    popt, pcov = curve_fit(fxn_expGauss, momRespBinMid, momRespHist[0], p0, sigma=momRespErrors)
    print("For SID=",TrkSID[isid],"Trk fit parameters",popt)
    print("Trk fit covariance",pcov)
    momRespFit[isid].stairs(edges=momRespHist[1],values=momRespHist[0],label="$\\Delta$ P")
    momRespFit[isid].plot(momRespBinMid, fxn_expGauss(momRespBinMid, *popt), 'r-',label="EMG Fit")
    momRespFit[isid].legend()
    momRespFit[isid].set_title("EMG fit at "+TrkLoc[isid])
    momRespFit[isid].set_xlabel("Fit-MC Origin Momentum (MeV)")
    fig.text(FigX[isid], 0.5, f"$\\mu$ = {popt[1]:.3f}")
    fig.text(FigX[isid], 0.4, f"$\\sigma$ = {popt[2]:.3f}")
    fig.text(FigX[isid], 0.3, f"$\\lambda$ = {popt[3]:.3f}")
plt.show()
