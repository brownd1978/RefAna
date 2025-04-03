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
files = [
"/data/HD5/users/brownd/61803286/nts.brownd.TAReflect.CRYAllOffSpill.001202_00000010.root:TARe/ntuple"
]
UpMomRes = []
DnMomRes = []
for batch,rep in uproot.iterate(offfiles,filter_name="/trk|trksegs|trkmcsim|gtrksegsmc/i",report=True):
    print("Processing batch ",ibatch)
    ibatch = ibatch+1
    segs = batch['trksegs'] # track fit samples
    nhits = batch['trk.nactive']  # track N hits
    fitcon = batch['trk.fitcon']  # track fit consistency
    fitpdg = batch['trk.pdg']  # track fit consistency
    trkMC = batch['trkmcsim']  # MC genealogy of particles
    segsMC = batch['trksegsmc'] # SurfaceStep infor for true primary particle
#    ak.type(segs).show()
#    print("segs axis 0: ",ak.num(segs,axis=0))
#    print("segs axis 1: ",ak.num(segs,axis=1))
#    print("segs axis 2: ",ak.num(segs,axis=2))
    upSegs = segs[:,0] # upstream track fits
    dnSegs = segs[:,1] # downstream track fits
    upSegsMC = segsMC[:,0] # upstream track MC truth
    dnSegsMC = segsMC[:,1] # downstream track MC truth
    upTrkMC = trkMC[:,0] # upstream fit associated true particles
    dnTrkMC = trkMC[:,1] # downstream fit associated true particles
    # basic consistency test
    assert((len(upSegs) == len(dnSegs)) & (len(upSegsMC) == len(dnSegsMC)) & (len(upSegs) == len(upSegsMC))& (len(upTrkMC) == len(dnTrkMC)) & (len(upSegs) == len(upTrkMC)))
    upMomRes = upResMom-upResMomMC
    UpMomRes.extend(upMomRes)
    dnMomRes = dnResMom-dnResMomMC
    DnMomRes.extend(dnMomRes)
# plot momentum resolution
nMomResBins = 200
fig, (upMomRes, dnMomRes)= plt.subplots(1,2,layout='constrained', figsize=(10,5))
upMomRes.hist(UpMomRes,label="Upstream",bins=nMomResBins, range=momresorange, histtype='bar')
upMomRes.set_title("Upstream Momentum Resolution at Tracker Mid")
upMomRes.set_xlabel("Reco - True Momentum (MeV)")
dnMomRes.hist(DnMomRes,label="Downstream",bins=nMomResBins, range=momresorange, histtype='bar')
dnMomRes.set_title("Downstream Momentum Resolution at Tracker Mid")
dnMomRes.set_xlabel("Reco - True Momentum (MeV)")
plt.show()

# plot dnstream and downstream momentum  Resolution
