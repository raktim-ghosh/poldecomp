# poldecomp

poldecomp is a perpetually incomplete, general purpose library for common post-processing task involving scattering power decompostion modelling in the domain polarimetric SAR (PolSAR) remote sensing. Currently, 12 different methods of PolSAR decomposition have been implemeted including some of the most recent and advanced model-based decompositions (6-Component Scattering Power Decompositions, 7-Component Scattering Power Decompositions).

"*** Programmer - Raktim Ghosh (MSc, University of Twente)  Date Written - May, 2020 ***"


# List of scattering power decomposition models

"""========================================================================"""

Touzi Decomposition                                                      Class Name (ModTouzi)          

Touzi Decomposition                                                      CLass Name (Touzi)             

H/A/Alpha Decompositions (Cloude-Pottier) -                              Class Name (HAAlpha)

Sinclair Decomposition                                                   Class Name (Sinclair)

Cloude Decomposition                                                     CLass Name (Cloude)

Pauli Decomposition                                                      Class Name (Pauli)

Van Zyl Decomposition                                                    Class Name (Vanzyl)

FreeMan-Durden Decomposition                                             Class Name (FreeMan)

Yamaguchi 4-Component Decomposition                                      Class Name (Yamaguchi2005)   

Yamaguchi 4-Component Decomposition                                      Class Name (Yamaguchi2011)

General 4-Component Decomposition (Singh)                                Class Name (General4SD)

Model-based 6-Component Decomposition (Singh)                            Class Name (General6SD)     

"Seven Component Decomposition (Singh)                                   Class Name (General7SD)

"""========================================================================"""
# Required packages 

python 3.6 or greater

numpy 1.16 or greater

scipy 

opencv

gdal

# How To Use

step1: create an object by importing gdal for your tiff file - 

example:
band = gdal.Open("Path/example.tif")

step2: create an object by specifying the class name from the above list for the decomposition, and inputting

the object defined in step1 (in this case, it is band), and window size 

for the ensemble averaging of the covariance or coherency matrix

example:
if you want to perform General4SD with window size 5 * 5, 

then,

decomposition = General4SD(band, 5)

step3: Finally, for getting the decomposition result, 

example:
decomposition.get_result()

And it will perform decomposition :)
