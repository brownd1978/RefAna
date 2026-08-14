#
# plot histograms of reflecting particles
#
from matplotlib import pyplot as plt
import MyHist
import h5py
class PlotReflections(object):
    def __init__(self,savefile):
        self.HUpTQ = MyHist.MyHist(name="HUpTQ",label="Up TrkQual",file=savefile)
        self.HDnTQ = MyHist.MyHist(name="HDnTQ",label="Down TrkQual",file=savefile)
        self.HUpFitCon = MyHist.MyHist(name="HUpFitCon",label="Up FitCon",file=savefile)
        self.HDnFitCon = MyHist.MyHist(name="HDnFitCon",label="Down FitCon",file=savefile)
        self.HUpNHits = MyHist.MyHist(name="HUpNHits",label="Up NActive",file=savefile)
        self.HDnNHits = MyHist.MyHist(name="HDnNHits",label="Down NActive",file=savefile)

        self.HNST = MyHist.MyHist(name="NInter",label="All ST",file=savefile)
        self.HNIPA = MyHist.MyHist(name="NInter",label="All IPA",file=savefile)
        self.HNSTTgt = MyHist.MyHist(name="NInter",label="Target ST",file=savefile)
        self.HNIPATgt = MyHist.MyHist(name="NInter",label="Target IPA",file=savefile)
        self.HDnMom = MyHist.MyHist(name="DnMom",label="All",file=savefile)
        self.HDnTgtMom = MyHist.MyHist(name="DnMom",label="$N_{ST}$>0",file=savefile)
        self.HDnNoMatMom = MyHist.MyHist(name="DnMom",label="No Material",file=savefile)
        self.HUpMom = MyHist.MyHist(name="UpMom",label="All",file=savefile)
        self.HUpTgtMom = MyHist.MyHist(name="UpMom",label="$N_{ST}$>0",file=savefile)
        self.HUpNoMatMom = MyHist.MyHist(name="UpMom",label="No Material",file=savefile)
        self.HDnDeltaMom = MyHist.MyHist(name="DnDeltaMom",label="All",file=savefile)
        self.HDnDeltaTgtMom = MyHist.MyHist(name="DnDeltaMom",label="$N_{ST}$>0",file=savefile)
        self.HDnDeltaNoMatMom = MyHist.MyHist(name="DnDeltaMom",label="No Material",file=savefile)
        self.HUpDeltaMom = MyHist.MyHist(name="UpDeltaMom",label="All",file=savefile)
        self.HUpDeltaTgtMom = MyHist.MyHist(name="UpDeltaMom",label="$N_{ST}$>0",file=savefile)
        self.HUpDeltaNoMatMom = MyHist.MyHist(name="UpDeltaMom",label="No Material",file=savefile)
        #
        self.HUpMatUpExtrap = MyHist.MyHist(name="UpMatUpExtrap",label="UpExtrap",file=savefile)
        self.HUpMatDnExtrap = MyHist.MyHist(name="UpMatDnExtrap",label="DnExtrap",file=savefile)
        self.HUpMatFit = MyHist.MyHist(name="UpMatFit",label="Fit",file=savefile)
        self.HUpMatTgtUpExtrap = MyHist.MyHist(name="UpMatTgtUpExtrap",label="UpExtrap",file=savefile)
        self.HUpMatTgtDnExtrap = MyHist.MyHist(name="UpMatTgtDnExtrap",label="DnExtrap",file=savefile)
        self.HUpMatTgtFit = MyHist.MyHist(name="UpMatTgtFit",label="Fit",file=savefile)
        self.HUpMatNoMatUpExtrap = MyHist.MyHist(name="UpMatNoMatUpExtrap",label="UpExtrap",file=savefile)
        self.HUpMatNoMatDnExtrap = MyHist.MyHist(name="UpMatNoMatDnExtrap",label="DnExtrap",file=savefile)
        self.HUpMatNoMatFit = MyHist.MyHist(name="UpMatNoMatFit",label="Fit",file=savefile)
        #
        self.HUpMatMC = MyHist.MyHist(name="UpMatMC",label="MC",file=savefile)
        self.HUpMatTgtMC = MyHist.MyHist(name="UpMatTgtMC",label="MC",file=savefile)
        self.HUpMatNoMatMC = MyHist.MyHist(name="UpMatNoMatMC",label="MC",file=savefile)
        #
        self.HupDeltaTime = MyHist.MyHist(name="DeltaTime",label="Upstream",file=savefile)
        self.HdnDeltaTime = MyHist.MyHist(name="DeltaTime",label="Downstream",file=savefile)


    def PlotQuality(self):
        fig, (anhit,afc,atq) = plt.subplots(1,3,layout='constrained', figsize=(15,5))
        upnhit = self.HUpNHits.plot(anhit)
        dnnhit = self.HDnNHits.plot(anhit)
        anhit.legend(loc="upper right")
        upfc = self.HUpFitCon.plot(afc)
        dnfc = self.HDnFitCon.plot(afc)
        afc.legend(loc="upper right")
        uptq = self.HUpTQ.plot(atq)
        dntq = self.HDnTQ.plot(atq)
        atq.legend(loc="upper right")

    def PlotIntersections(self):
        fig, (cmat,cselmat) = plt.subplots(1,2,layout='constrained', figsize=(10,5))
        nipa = self.HNIPA.plot(cmat)
        nst = self.HNST.plot(cmat)
        cmat.legend(loc="upper right")
        nipasel = self.HNIPATgt.plot(cselmat)
        nstsel = self.HNSTTgt.plot(cselmat)
        cselmat.legend(loc="upper right")

    def PlotMomentum(self):
        fig, (upMom, dnMom) = plt.subplots(1,2,layout='constrained', figsize=(10,5))
        upmom = self.HUpMom.plot(upMom)
        uptgtmom = self.HUpTgtMom.plot(upMom)
        upnomatmom = self.HUpNoMatMom.plot(upMom)
        upMom.legend(loc="upper right")
        dnmom = self.HDnMom.plot(dnMom)
        dntgtmom = self.HDnTgtMom.plot(dnMom)
        dnnomatmom = self.HDnNoMatMom.plot(dnMom)
        dnMom.legend(loc="upper right")

    def PlotDeltaMomentum(self):
        fig, (upDMom, dnDMom) = plt.subplots(1,2,layout='constrained', figsize=(10,5))
        dndelmom = self.HDnDeltaMom.plot(dnDMom)
        dndeltgtmom = self.HDnDeltaTgtMom.plot(dnDMom)
        dndelnomatmom = self.HDnDeltaNoMatMom.plot(dnDMom)
        dnDMom.legend(loc="upper right")
        updelmom = self.HUpDeltaMom.plot(upDMom)
        updeltgtmom = self.HUpDeltaTgtMom.plot(upDMom)
        updelnomatmom = self.HUpDeltaNoMatMom.plot(upDMom)
        upDMom.legend(loc="upper right")

    def PlotUpstreamMat(self):
        fig, (all, tgt, nomat) = plt.subplots(1,3,layout='constrained', figsize=(15,5))
        upxtrapdmom = self.HUpMatUpExtrap.plot(all)
        dnxtrapdmom = self.HUpMatDnExtrap.plot(all)
        fitdmom = self.HUpMatFit.plot(all)
        mcdmom = self.HUpMatMC.plot(all)
        all.legend(loc="upper left")
        upxtrapdmom = self.HUpMatTgtUpExtrap.plot(tgt)
        dnxtrapdmom = self.HUpMatTgtDnExtrap.plot(tgt)
        fitdmom = self.HUpMatTgtFit.plot(tgt)
        mcdmom = self.HUpMatTgtMC.plot(tgt)
        tgt.legend(loc="upper left")
        upxtrapdmom = self.HUpMatNoMatUpExtrap.plot(nomat)
        dnxtrapdmom = self.HUpMatNoMatDnExtrap.plot(nomat)
        fitdmom = self.HUpMatNoMatFit.plot(nomat)
        mcdmom = self.HUpMatNoMatMC.plot(nomat)
        nomat.legend(loc="upper left")

    def PlotTime(self):
        fig, (deltaTime) = plt.subplots(1,1,layout='constrained', figsize=(5,5))
        updtime = self.HupDeltaTime.plot(deltaTime)
        dndtime = self.HdnDeltaTime.plot(deltaTime)
        deltaTime.legend(loc="upper right")
