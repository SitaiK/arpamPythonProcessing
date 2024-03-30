# arpamPythonProcessing
This is for ARPAM python processing only, data format is USPA data, 8192*1000*n U16 data

## input
this code take input *only* from the ARPAM new sysmem labview code, from April 2023 and running, data format is *8192*1000*frameNum* U16 data in little-endian, please change accordingly if you're using an different version of code, please do not open the files for writing!

## processing
this code take care of the signal apodization, fitlering, compression, image conversion, and plotting, 
no normalization is present inside the code and please use if in your own discresion
suggested dynamic range for the signals are 40dB for PA and 50dB for US

## code components
- load data
- apod and filter data
- getting envelop and compression
- scan conversion
- image coregistration
- image combining

## control
through controls, you can limite the number of procesing steps to take for the image processing, and parallelerize the running of the code


Have fun
Sitai Kou
