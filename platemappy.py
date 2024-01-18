""" makes 96-well plates for experiments """

__version__ = "0.1"

import pandas as pd
import random
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib.lines as mlines


class Metadata:
    """
    This is a metadata class that generates plate maps.

    Parameters:
        X: pd.DataFrame
            metadata obj, index=samples
        disease_map: str
            column name of x that is binary map for disease vs healthy: {1:disease, 0:healthy}
        n_ag: int
            number AG, per plate 
        n_canary: int
            number Canary, per plate 
        n_mAb: int
            number mAb, per plate 
        block_: int, default None
            size of continguious block to be added to plates
        duplicate_: int, default None 
            number of duplicates of each sample, total

    Attributes:
        pmaps: list of (8,12) dataframes
            96 well plate maps, with sample ids or controls as values
    """

    def __init__(self, 
                 X: pd.DataFrame, 
                 disease_map: str,
                 n_ag: int, 
                 n_canary: int, 
                 n_mAb: int, 
                 block_: int, 
                 duplicate_: int) -> None:
        self._X = X
        self._disease_map = disease_map
        self._ag = n_ag
        self._can = n_canary
        self._mab = n_mAb
        self._block=block_
        self._dup=duplicate_

        self.k = self._init_params() 
    
    def _init_params(self):
        """ 
        Calculates number of plates. 

        Returns: 
            n: int
                number of healthies
            m: int
                number of patients
            k: 
                number of plates
        """
        x = self._ag + self._can + self._mab # total controls per plate

        #plates is the total number of samples that fit into 96 well plate (minus controls), with round up
        if self._block and not self._dup: 
            k = math.ceil((self._X.shape[0])/(96-x-self._block))
        elif self._block and self._dup: 
            k = math.ceil((self._X.shape[0]*self._dup)/(96-x-self._block))
        elif self._dup and not self._block: 
            k = math.ceil((self._X.shape[0]*self._dup)/(96-x))
        else: 
            k = math.ceil((self._X.shape[0])/(96-x)) 

        return k
    
    def allocate_samples_to_plate(self)-> dict:
        """ 
        divides samples to different plates. the ratio of healthy to disease will be maintained in 
        this division (m/n = (m/k)/(n/k)), so don't need to worry about that. 

        Returns: 
            samples_to_plates: dict
                dictionary where keys are plate numbers 1..k and values are lists of healthies/disease. 

        """
        samples_to_plates={(i+1):list() for i in range(self.k)}
        
        for c in self._X[self._disease_map].unique():
            samples_c=self._X.index[self._X[self._disease_map]==c].tolist()

            # duplicate samples if needed
            if self._dup:
                samples_c=np.repeat(samples_c, self._dup) # this duplicates every value in the list

            #shuffle
            random.shuffle(samples_c)

            #find split to plates
            d=math.ceil(len(samples_c)/self.k)

            #allocate shuffled samples to plates
            for i in range(self.k-1):
                samples_to_plates[i+1].extend(samples_c[i*d:(i+1)*d])
            i=self.k-2
            samples_to_plates[self.k].extend(samples_c[(i+1)*d:])
        
        return samples_to_plates
        
    
    def create_plates(self, samples_to_plates) -> dict:
         """ 
        creates plates with the randomly allocated samples and controls.


        Returns: 
            plates: dict
                dictionary where keys are plate numbers 1..k and values are arrays. 

        """
         plates={}

         #generate number of controls per plate
         AG=['AG']*self._ag
         Canary=['Canary']*self._can
         mAb=['mAb']*self._mab
         
         #randomly shuffle samples and controls
         for i in range(self.k):
             samples=samples_to_plates[i+1]
             print(len(samples))
             samples.extend(AG)
             samples.extend(Canary)
             samples.extend(mAb)
            # all_vals=[samples_to_plates[i+1].extend(l) for l in [AG, Canary, mAb]] # concatenate all values
             random.shuffle(samples) # shuffle

             if self._block: 
                insert_loc=random.choice(range(96-self._block))
                for z in range(self._block):
                    samples.insert(insert_loc,'HC')
            
             #if not full 96 well plate, pad with zeros
             if len(samples)<96: 
                samples.extend([0]*(96-len(samples)))
             
             plates[i+1]=np.reshape(samples, (8,12))

         return plates
    
    def save_plates(plate_dict, base_path)->None: 
        """
        saves your plates into csv format 

        Parameters:
            plate_dict: dict
                dictionary of k 96-well plates
            base_path: str
                path to save the csv files
        """
        for p in plate_dict.keys(): 
            X=plate_dict[p]
            pd.DataFrame(data=X, columns=['1','2','3','4','5','6','7','8','9','10','11','12'], 
            index=['A','B','C','D','E','F','G','H']).to_csv(base_path+'plate_{}.csv'.format(p))
        return None

    def view_plate(self, plate, view_bin=False, key=False, save=False, save_loc=None) -> None: 
        """
        plotting function to visualize a plate layout 

        Parameters: 
            plate: np.array 
                8x12 array with ID
            view_bin: bool, default=False
                option to view the binary version of the criteria split instead of words
            key: bool, default=False
                option to make a legend
            save: bool, default=False
                option to save the image
            save_loc: string, default=None
                save path and name, must specify if save==True
        """
        fig, ax = plt.subplots(figsize=(15,10))

        #number of categories
        cats=self._X[self._disease_map].unique()
        n_cats=len(cats)

        # make the colormap for disease categories
        default_colors=['lightgreen','cornflowerblue','turquoise','deepskyblue','dodgerblue','seafoamgreen','lightgreen'] #colors for sample categories   
        cmap_=ListedColormap(default_colors[:n_cats]+['tomato','gold','darkorange','lemonchiffon','lightgrey']) # add in colors for control catgeories
        norm_=BoundaryNorm([0.5+x for x in range(-1,n_cats+4,1)]+[1000.5], cmap_.N) #define boundaries for cmap

        #binarize all labels 
        bin_ctrls={'AG':n_cats, 'Canary':n_cats+1, 'mAb':n_cats+2, 'HC':n_cats+3, '0':1000}
        bin_ctrls.update(self._X[self._disease_map].to_dict())
        bin_plate=np.vectorize(bin_ctrls.get)(plate)    

        for i in range(8):
            for j in range(12):
                if view_bin:
                    c=bin_plate[i,j]
                else: 
                    c=plate[i,j]
                ax.text(j, i, c, va='center', ha='center')
                ax.vlines(j+0.5, ymin=-.5,ymax=8.5, color='black')
                ax.hlines(i+0.5, xmin=-.5, xmax=12.5, color='black')

        if key: 
            handles_=[]
            for k in range(n_cats): 
                handles_.append(mlines.Line2D([],[], color=default_colors[k], marker='s', linestyle='None',
                          markersize=10, label=cats[k]))
            handles_.append(mlines.Line2D([],[], color='lemonchiffon', marker='s', linestyle='None',
                          markersize=10, label='Healthy Control'))
            plt.legend(handles=handles_, bbox_to_anchor=(1.05, 1),
                         loc='upper left', borderaxespad=0.)

        ax.matshow(bin_plate, cmap=cmap_, norm=norm_)

        ax.set_xticks(range(12))
        ax.xaxis.tick_top()
        ax.set_xticklabels(['1','2','3','4','5','6','7','8','9','10','11','12'], fontsize='x-large')

        ax.set_yticks(range(8))
        ax.set_yticklabels(['A','B','C','D','E','F','G','H'], fontsize='x-large')
        
        plt.show()
