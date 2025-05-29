import numpy as np
import h5py
class MyHist(object):
    def __init__(self,label,bins=100,range=[],title="",xlabel="",file=""):
        self.label = label
        if(file != ""):
            #locate this histogram as a group with the given label
            with h5py.File(file, 'r') as hdf5file: # closes on exit
                self.data = hdf5file["/"+self.label+"/data"][:]
                self.edges = hdf5file["/"+self.label+"/edges"][:]
                self.title = hdf5file.get("/"+self.label+"/title").asstr()[0]
                self.xlabel = hdf5file.get("/"+self.label+"/xlabel").asstr()[0]
                self.print()
        else:
            data = []
            self.data, self.edges = np.histogram(data, bins=bins, range=range)
            self.title = title
            self.xlabel = xlabel

    def print(self):
        print("MyHist",self.label,self.title,"with",len(self.data),"bins and",self.integral(),"entries")

    def fill(self,data):
        newdata , newedges = np.histogram(data, bins=self.edges)
        self.data += newdata

    def plot(self,axis):
        plt = axis.stairs(self.data,self.edges,label=self.label)
        if(self.title != ""):
            axis.set_title(self.title)
        if(self.xlabel != ""):
            axis.set_xlabel(self.xlabel)
        return plt

    def plotErrors(self,axis):
        plt =self.plot(axis)
        errors = self.binErrors()
        centers = self.binCenters()
        axis.errorbar(x=centers,y=self.data,yerr=errors)
        return plt

    def integral(self):
        return np.sum(self.data)

    def binCenters(self):
        midbin = np.zeros(len(self.data))
        for ibin in range(len(self.data)):
            midbin[ibin] = 0.5*(self.edges[ibin] + self.edges[ibin+1]) # edges have 1 more entry than data
        return midbin

    def binErrors(self):
        # assume unweighted bins
        errors = np.sqrt(self.data)
        ones = np.ones(len(self.data))
        errors = np.maximum(ones,errors)
        return errors

    def save(self,hdf5file):
        grp = hdf5file.create_group(self.label)
        grp.create_dataset("data",data=self.data)
        grp.create_dataset("edges",data=self.edges)
        dst = grp.create_dataset("title", shape=1, dtype=h5py.string_dtype())
        dst[:] = self.title
        dsx = grp.create_dataset("xlabel", shape=1, dtype=h5py.string_dtype())
        dsx[:] = self.xlabel
