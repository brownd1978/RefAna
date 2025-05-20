#
# class to analyze high-energy electrons
#
import uproot
import awkward as ak
import behaviors
from matplotlib import pyplot as plt
import uproot
import numpy as np
from scipy.optimize import curve_fit
import math
from scipy import special
import SurfaceIds as SID

def fxn_expGauss(x, amp, mu, sigma, lamb):
    z = (mu + lamb*(sigma**2) + x)/(np.sqrt(2)*sigma)
    comp_err_func = special.erfc(z)
    val = amp*(lamb/2)*((math.e)**((lamb/2)*(2*mu+lamb*(sigma**2)+2*x)))*comp_err_func


class CeAna(object):
    def __init__(self,momrange,minNHits,minFitCon,sids):
        self.MomRange = momrange
        self.minNHits = minNHits
        self.minFitCon = minFitCon
        self.SIDs = sids
        self.TrkLoc = [None]*3
        self.TrkLoc[0] = "Tracker Entrance"
        self.TrkLoc[1] = "Tracker Middle"
        self.TrkLoc[2] = "Tracker Exit"
        self.FigX =[None]*3
        self.FigX[0] = 0.05
        self.FigX[1] = self.FigX[0] + 1.0/3.0
        self.FigX[2] = self.FigX[1] + 1.0/3.0
    def Loop(self,files):
       # Momentum and response
        Mom = []
        MCMom = []
        MomReso = []
        MomResp = []
        originMomMC = []
        elPDG = 11
        for isid in range(len(self.SIDs)+1) : # add 1 for target
            Mom.append([])
            MomReso.append([])
            MomResp.append([])
            originMomMC.append([])
        ibatch=0
        itarget = len(Mom)-1;
#        Mom[itarget] = np.array()
        Mom[itarget] = []
        MomResp[itarget] = []

        rfile = uproot.open(files[0]+":EventNtuple")
        print(rfile.keys())

        for batch,rep in uproot.iterate(files,filter_name="/evtinfo|trk|trksegs|trkmcsim|trksegsmc/i",report=True):
            print(batch,type(batch))
            print("Processing batch ",ibatch)
            ibatch = ibatch+1
            runnum = batch['run']
            subrun = batch['subrun']
            event = batch['event']
            segs = batch['trksegs'] # track fit samples
            nhits = batch['trk.nactive']  # track N hits
            fitcon = batch['trk.fitcon']  # track fit consistency
            trkMC = batch['trkmcsim']  # MC genealogy of particles
            segsMC = batch['trksegsmc'] # SurfaceStep infor for true primary particle
            Segs = segs[:,0] # assume the 1st track is the Ce
            SegsMC = segsMC[:,0]
            TrkMC = trkMC[:,0]
            FitCon = fitcon[:,0]
            Nhits = nhits[:,0]
            goodFit = (Nhits >= self.minNHits) & (FitCon > self.minFitCon)
            goodTrkMC = (TrkMC.pdg == elPDG) & (TrkMC.trkrel._rel == 0)
            TSDASeg = Segs[Segs.sid == 96 ] #TSDA()] FIXME
            noTSDA = ak.num(TSDASeg)==0
        #    print(noTSDA)
            originMomMC = TrkMC[goodTrkMC].mom.magnitude()
            # basic consistency test
            assert((len(runnum) == len( Segs)) & (len(Segs) == len(SegsMC)) & (len(Segs) == len(TrkMC)) & (len(Nhits) == len(Segs)) & (len(originMomMC) == len(Segs)))
            originMomMC = originMomMC[(originMomMC>self.MomRange[0]) & (originMomMC < self.MomRange[1])]
            omomMC = ak.flatten(originMomMC,axis=1)
            MCMom.extend(omomMC)
            hasOriginMomMC = ak.count_nonzero(originMomMC,axis=1,keepdims=True)==1
            hasOriginMomMC = ak.flatten(hasOriginMomMC,axis=1)
        #    print(hasOriginMomMC[0:10])
        #    print(originMomMC[0:10])
            # sample the fits at the specified
            for isid in range(len(self.SIDs)) :
                sid = self.SIDs[isid]
                segs = Segs[(Segs.sid == sid) & (Segs.mom.z() > 0.0) ]
                mom = segs.mom.magnitude()
                mom = mom[(mom > self.MomRange[0]) & (mom < self.MomRange[1])]
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
            # look for the furthest upstream target intersection
            for outer in Segs:
#                print("New track")
                minfoil = 1000
                # find furthest upstream foil
                for inner in outer:
#                    print("SID",inner.sid)
                    if(inner.sid == SID.ST_Foils()):
#                        print("Foil intersection, index",inner.sindex)
                        if(inner.sindex < minfoil):
                            minfoil = inner.sindex
#                print("Setting minfoil",minfoil)
                for inner in outer:
#                    print("SID",inner.sid)
                    if((inner.sid == SID.ST_Foils()) & (inner.sindex == minfoil)):
                        mom = inner.mom.magnitude()
#                        print("Filling target mom",mom,"foil",minfoil)
#                        Mom[itarget].extend(mom)
                        Mom[itarget].append(mom)
                        MomResp[itarget].append(mom-105) # hack FIXME

            # count missing intersections
            hasent = (Segs.sid == 0) & (Segs.mom.z() > 0.0)
            hasmid = (Segs.sid == 1) & (Segs.mom.z() > 0.0)
            hasxit = (Segs.sid == 2) & (Segs.mom.z() > 0.0)
            hasent = ak.any(hasent,axis=1)
            hasmid = ak.any(hasmid,axis=1)
            hasxit = ak.any(hasxit,axis=1)
            hasall = hasent & hasmid & hasxit
            print("Found",ak.count(hasall,0) - ak.count_nonzero(hasall),"Instances of missing intersections in",ak.count(hasall,0),"tracks")
            for itrk in range(len(hasall)):
                if (not hasall[itrk]):
                    print("Missing intersection: ",hasent[itrk],hasmid[itrk],hasxit[itrk]," eid ",runnum[itrk],":",subrun[itrk],":",event[itrk],sep="")
            # plot Momentum
        nDeltaMomBins = 200
        nMomBins = 100
        momrange=(self.MomRange[0],107)
        momresorange=(-2.5,2.5)
        momresprange=(-10,5)
        #print(len(Mom[0]),Mom[0:10][0])
        #print(len(MomReso[0]),MomReso[0:10][0])
        momVal = [None]*3
        momReso = [None]*3
        momResp = [None]*3
        momRespFit = [None]*3
        MomRespHist = [None]*3
 #       print("Mom[itarget]",Mom[itarget])

        fig, (tgtMom, momVal[0], momVal[1], momVal[2]) = plt.subplots(1,4,layout='constrained', figsize=(10,5))
        tgtMom.hist(Mom[itarget],label="ST", bins=nMomBins, range=momrange, histtype='step')
        tgtMom.set_xlabel("Extrapolated Momentum (MeV)")
        tgtMom.set_title("Upstream ST Foil")
        for isid in range(len(self.SIDs)) :
            momVal[isid].hist(Mom[isid],label=self.TrkLoc[isid], bins=nMomBins, range=momrange, histtype='step')
            momVal[isid].set_xlabel("Fit Momentum (MeV)")
            momVal[isid].set_title(self.TrkLoc[isid])

        fig, (tgtMomResp,momResp[0], momResp[1], momResp[2]) = plt.subplots(1,4,layout='constrained', figsize=(10,5))
        tgtMomResp.hist(MomResp[itarget],label="ST", bins=nMomBins, range=momresprange, histtype='step')
        tgtMomResp.set_xlabel("Extrapolated - MC Origin Momentum (MeV)")
        tgtMomResp.set_title("Upstream ST Foil")
        for isid in range(len(self.SIDs)) :
            MomRespHist[isid] =  momResp[isid].hist(MomResp[isid],label=self.TrkLoc[isid], bins=nMomBins, range=momresprange, histtype='step')
            momResp[isid].set_xlabel("Fit - MC Origin Momentum (MeV)")
            momResp[isid].set_title(self.TrkLoc[isid])

        fig, (momReso[0], momReso[1], momReso[2]) = plt.subplots(1,3,layout='constrained', figsize=(10,5))
        for isid in range(len(self.SIDs)) :
            momReso[isid].hist(MomReso[isid],label=self.TrkLoc[isid], bins=nMomBins, range=momresorange, histtype='step')
            momReso[isid].set_xlabel("Fit - MC Momentum (MeV)")
            momReso[isid].set_title(self.TrkLoc[isid])
        #
        # response function fit
        #
        fig, (momRespFit[0],momRespFit[1],momRespFit[2]) = plt.subplots(1,3,layout='constrained', figsize=(10,5))
        for isid in range(len(self.SIDs)) :
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
            print("For SID=",self.SIDs[isid],"Trk fit parameters",popt)
            print("Trk fit covariance",pcov)
            momRespFit[isid].stairs(edges=momRespHist[1],values=momRespHist[0],label="$\\Delta$ P")
            momRespFit[isid].plot(momRespBinMid, fxn_expGauss(momRespBinMid, *popt), 'r-',label="EMG Fit")
            momRespFit[isid].legend()
            momRespFit[isid].set_title("EMG fit at "+self.TrkLoc[isid])
            momRespFit[isid].set_xlabel("Fit-MC Origin Momentum (MeV)")
            fig.text(self.FigX[isid], 0.5, f"$\\mu$ = {popt[1]:.3f}")
            fig.text(self.FigX[isid], 0.4, f"$\\sigma$ = {popt[2]:.3f}")
            fig.text(self.FigX[isid], 0.3, f"$\\lambda$ = {popt[3]:.3f}")
        plt.show()
