# %% 
# defining functions

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import map_coordinates
from PIL import Image
import cv2 
from pathlib import Path

# read the RF data
def load_file_USPA(filePath,fileName,frameNum):
    """Open the bin file in binary mode and seek to the desired position"""
    # file_size = fileName.stat().st_size

    with open(fileName, 'rb') as file:
        # Number of bits to skip at the beginning of the bin file
        file.seek((8192*1000*2*frameNum)+1)  # Divide by 8 to convert bits to bytes
        
        # Read the  contents of the bin file
        data = file.read(8192*1000*2)
        values = np.frombuffer(data, dtype=np.uint16)
        
        #values = np.array(struct.unpack(f'{len(Data)//2}H', remainingData))
        values_volt = (values / (2**15 ) - 1) * 1
        file.close()
    
    RFdata = values_volt.reshape(1000, 8192).T
    if frameNum % 2 == 0: 
        RFdata = np.fliplr(RFdata)
        RFdata = RFdata [:,range(0,990)]
    else: 
        RFdata = RFdata [:,range(10,1000)]
    
    # inspection plot
    # plt.imshow(signal.decimate(signal.decimate(RFdata, 10, axis=1), 2, axis=0)*255, cmap='gray') 

    return RFdata

# create a filter
def bandpass_firwin(ntaps, lowcut, highcut, fs, window='blackmanharris'):
    taps = signal.firwin(ntaps, [lowcut, highcut], fs=fs, pass_zero=False,
                  window=window, scale=False)
    # visualize filter
    # w, h = signal.freqz(filter_coeffs, worN=8000)
    # plt.figure()
    # plt.plot(w, np.abs(h))
    # plt.title("Frequency Response")
    # plt.xlabel("Normalized Frequency (π radians/sample)")
    # plt.ylabel("Magnitude")
    # plt.grid(True)
    # plt.show()
    return taps

# filter B scan Signal
def bandpass_Bscan(RFdata,type_signal='US',cutoff_low=9,cutoff_high=20):
    # Bandpass procesing the b scan
    if type_signal == 'US':
        cutoff_low = 9
        cutoff_high = 22
    elif type_signal == 'PA':
        cutoffLow = 3
        cutoff_high = 18
    
    # perform filtering
    RFdataFilter = np.empty(RFdata.shape)
    filter_coeffs = bandpass_firwin(95, cutoff_low, cutoff_high, 180) # tap, low, high, freq
    for row in range(RFdata.shape[1]):
        RFdataFilter[:,row] = signal.lfilter(filter_coeffs, 1, RFdata[:,row])
    return RFdataFilter

# get signal conditioning and Process B scan data
def Process_Bscan_Data(RFdata,dB_limit=40,median_filter_size=0):
    if median_filter_size !=0: 
        RFdataProcess = signal.medfilt(RFdata, kernel_size=median_filter_size) 
    else: 
        RFdataProcess = RFdata   
    RFdataProcess = np.abs(signal.hilbert(RFdataProcess,axis=0))
    RFdataProcess = np.where(RFdataProcess == 0, 1E-6, RFdataProcess)

    RFdataProcessdB = np.empty(RFdataProcess.shape)
    RFdataProcessdB = (20*np.log10(RFdataProcess) + dB_limit)/dB_limit
    RFdataProcessdB = np.where(RFdataProcessdB < 0, 0, RFdataProcessdB)
    # data is resized to 0-1
    return RFdataProcessdB

def image_conversion(img_data,Resize_1=10,Resize_2=2, type_signal = 'US'):
    if type_signal == 'US':
        US_img_data_resize = signal.decimate(signal.decimate(img_data, Resize_1, axis=0), Resize_2, axis=1)*255.0
        US_img = np.empty([US_img_data_resize.shape[0],US_img_data_resize.shape[1],3])
        for i in range(3):
            US_img [:,:,i] = US_img_data_resize
        US_img = np.where(US_img < 0, 0, US_img)
        img_Out = US_img.astype(np.uint8)
        # plt.imshow(img_Out/255.0, extent=[0, 1, 0, 1])

    elif type_signal == 'PA':
        PA_img_data_resize = signal.decimate(signal.decimate(img_data, Resize_1//2, axis=0), Resize_2, axis=1)*255
        PA_img = np.empty([PA_img_data_resize.shape[0],PA_img_data_resize.shape[1],3])
        PA_img [:,:,0] = PA_img_data_resize * 2.5
        PA_img [:,:,1] = (PA_img_data_resize - 100.0) * 2.5
        PA_img [:,:,2] = (PA_img_data_resize - 200.0) * 2.5
        PA_img = np.where(PA_img < 0, 0, PA_img)
        img_Out = PA_img.astype(np.uint8)

    return img_Out

def coregistration(US_img, PA_img, threshold=15):

    PA_img[:,:,0] = np.where(PA_img[:,:,0] < threshold, 0, PA_img[:,:,0])  
    PA_img[:,:,1] = np.where(PA_img[:,:,0] < threshold, 0, PA_img[:,:,1])  
    PA_img[:,:,2] = np.where(PA_img[:,:,0] < threshold, 0, PA_img[:,:,2]) 

    US_img[:,:,0] = np.where(PA_img[:,:,0] > 0, 0,US_img[:,:,0])  
    US_img[:,:,1] = np.where(PA_img[:,:,0] > 0, 0,US_img[:,:,1])  
    US_img[:,:,2] = np.where(PA_img[:,:,0] > 0, 0,US_img[:,:,2])  

    Sum_img_data = US_img + PA_img

    return Sum_img_data

# convert to filer
def convert_To_Polar(imgData):
    
    polar_image = np.empty([imgData.shape[1]*2,imgData.shape[1]*2,imgData.shape[2]])

    image = imgData.astype(np.float32)
    
    polar_image[:,:,0] = cv2.warpPolar(image[:,:,0].T, (image.shape[1]*2,image.shape[1]*2),
                                 [image.shape[1],image.shape[1]], image.shape[1], cv2.WARP_INVERSE_MAP)
    polar_image[:,:,1] = cv2.warpPolar(image[:,:,1].T, (image.shape[1]*2,image.shape[1]*2),
                                 [image.shape[1],image.shape[1]], image.shape[1], cv2.WARP_INVERSE_MAP)
    polar_image[:,:,2] = cv2.warpPolar(image[:,:,2].T, (image.shape[1]*2,image.shape[1]*2),
                                 [image.shape[1],image.shape[1]], image.shape[1], cv2.WARP_INVERSE_MAP)
    
    polar_image = cv2.rotate(polar_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return polar_image.astype(np.uint8)

def save_image(img_data,filepath,num,typeData='Stack'):
    # Create an image from the uint8 data
    imageUS = Image.fromarray(img_data.astype(np.uint8))

    # Save the image
    name = typeData + '_' + num + '.tiff'
    name = filepath / name
    imageUS.save(name)
    

def plot_raw_data(RFdata, loc_plot=500,fig_title='RF_image.jpg'):

    fig, axs = plt.subplots(1,2,figsize=(12,9), gridspec_kw={'width_ratios': [2.5,1]})
    axs[0].imshow(RFdata, extent=[0, 1, 0, 1])
    axs[0].axis("off")
    axs[1].plot(RFdata[:,loc_plot],np.flip(range(0, RFdata.shape[0])))
    axs[1].axis("off")
    plt.savefig(fig_title)
# %% 
# reading and reformatting data    
root = Path('D:\Data')
file_num = 1
frame_num = 96
all_files = list(root.glob("*.bin"))


for file in all_files:
     print(file) # inspection
     new_folder = file.parent / (file.name[0:6]+'demo')
     new_folder.mkdir(parents = True, exist_ok = True)
     # print('total number of frame is '+ str(all_files[0].stat().st_size//1000//8192//2)) # get file size

# load data in the format of _8291*1000_
RFdata = load_file_USPA(root,all_files[file_num],frame_num)
plot_raw_data(RFdata,fig_title='RF_image.jpg')

# %% signal apodization
USRFdata = RFdata[range(2731, 8191),:]
USRFdata [range(0,800)] = 0



PARFdata = RFdata[range(0,2730),:]
PARFdata [range(2600,PARFdata.shape[0])] = 0

append_factor = 190
PARFdata = np.concatenate((np.zeros((append_factor, PARFdata.shape[1])),PARFdata), axis = 0)
USRFdata = np.concatenate((USRFdata,np.zeros((append_factor*2, USRFdata.shape[1]))), axis = 0)

plot_raw_data(USRFdata,fig_title='USRF_image.jpg')
plot_raw_data(PARFdata,fig_title='PARF_image.jpg')

# %%processing data

#Process US & PA data
USimgData = Process_Bscan_Data(bandpass_Bscan(USRFdata,'US'),dB_limit=50,median_filter_size=0)
PAimgData = Process_Bscan_Data(bandpass_Bscan(PARFdata,'PA'),dB_limit=45,median_filter_size=3)
# plt.imshow(USimgData, extent=[0, 1, 0, 1])

plot_raw_data(USimgData,fig_title='USdata_image.jpg')
plot_raw_data(PAimgData,fig_title='PAdata_image.jpg')

# %% 
US_img = image_conversion(USimgData,Resize_1=2,Resize_2=1, type_signal = 'US')
PA_img = image_conversion(PAimgData,Resize_1=2,Resize_2=1, type_signal = 'PA')

fig, axs = plt.subplots(1,2, figsize=(12,6), gridspec_kw={'width_ratios': [1,1]})
plt.subplot(1,2,1)
plt.imshow(PA_img, extent=[0, 1, 0, 1])
axs[0].axis("off")
plt.subplot(1,2,2)
plt.imshow(US_img, extent=[0, 1, 0, 1])
axs[1].axis("off")

plt.savefig('US & PA image.jpg')

US_img_polar = convert_To_Polar(US_img)
# %% superposition and save data
SUM_img = coregistration(US_img,PA_img,threshold=10)

plt.imshow(SUM_img, extent=[0, 1, 0, 1])
plt.savefig('US & PA image coregistered.jpg')
plt.axis("off")
# %% 

PA_img_polar = convert_To_Polar(PA_img)
SUM_img_polar = convert_To_Polar(SUM_img)

fig, axs = plt.subplots(1,2, figsize=(12,6), gridspec_kw={'width_ratios': [1,1]})
plt.subplot(1,2,1)
plt.imshow(US_img_polar, extent=[0, 1, 0, 1])
axs[0].axis("off")
plt.subplot(1,2,2)
plt.imshow(SUM_img_polar, extent=[0, 1, 0, 1])
axs[1].axis("off")

plt.savefig('STACKED image.jpg')

Stacked_image = combined_array = np.concatenate((US_img_polar, SUM_img_polar), axis=1)

save_image(Stacked_image,new_folder,str(frame_num),typeData='Stack')
print('processing of case '+ file.name[0:6]+ ' frame ' + str(frame_num) + ' is finisehd')




