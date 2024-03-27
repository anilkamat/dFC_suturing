rm(list=ls()) 
a = mni_to_region_name(x = 0, y = 0, z = 0) 
b =a
library("readxl")
library(label4MRI)

m = read_excel("C:\\Users\\_Kamat_\\Desktop\\RPI\\ResearchWork\\Papers_\\Effective_Connectivity\\EEG_fNIRS_paper_Brain_informatics\\channelEEG_codes_results\\Codes\\full_channel_locs_info.xlsx",sheet ="MNI_coordinates") #
Result <- t(mapply(FUN = mni_to_region_name, x = m$x, y = m$y, z = m$z))
print(Result[ , c(1, 2)])

# m = read_excel("C:\\Users\\_Kamat_\\Desktop\\RPI\\ResearchWork\\Papers_\\Effective_Connectivity\\EEG\\Data_EEG_fNIRS_Suturing\\EEG_preprocessed_sept27\\MNI_Coordinates\\IC_MNI_Coordinates.xlsx",sheet = "E09") #
# Result <- t(mapply(FUN = mni_to_region_name, x = m$x, y = m$y, z = m$z))
# print(Result[ , c(1, 2)])
# m = read.delim('C:\\Users\\_Kamat_\\Desktop\\RPI\\ResearchWork\\Papers_\\Effective_Connectivity\\EEG_fNIRS_paper_Brain_informatics\\channelEEG_codes_results\\Codes\\MNI_coordinates_of_EEG_motage.txt')
# Result <- t(mapply(FUN = mni_to_region_name, x = m$x, y = m$y, z = m$z))
# print(Result[ , c(1, 2)])

# m <- NULL 
# m = read_excel("C:\\Users\\_Kamat_\\Desktop\\RPI\\ResearchWork\\Papers_\\Effective_Connectivity\\EEG\\Data_EEG_fNIRS_Suturing\\EEG_preprocessed_sept27\\MNI_Coordinates\\IC_MNI_Coordinates.xlsx",sheet = "E011") #
# Result <- t(mapply(FUN = mni_to_region_name, x = m$x, y = m$y, z = m$z))
# print(Result[ , c(1, 2)])
# 
# m <- NULL 
# m = read_excel("C:\\Users\\_Kamat_\\Desktop\\RPI\\ResearchWork\\Papers_\\Effective_Connectivity\\EEG\\Data_EEG_fNIRS_Suturing\\EEG_preprocessed_sept27\\MNI_Coordinates\\IC_MNI_Coordinates.xlsx",sheet = "E012") #
# Result <- t(mapply(FUN = mni_to_region_name, x = m$x, y = m$y, z = m$z))
# print(Result[ , c(1, 2)])
# 
# m <- NULL 
# m = read_excel("C:\\Users\\_Kamat_\\Desktop\\RPI\\ResearchWork\\Papers_\\Effective_Connectivity\\EEG\\Data_EEG_fNIRS_Suturing\\EEG_preprocessed_sept27\\MNI_Coordinates\\IC_MNI_Coordinates.xlsx",sheet = "E013") #
# Result <- t(mapply(FUN = mni_to_region_name, x = m$x, y = m$y, z = m$z))
# 
# m <- NULL 
# m = read_excel("C:\\Users\\_Kamat_\\Desktop\\RPI\\ResearchWork\\Papers_\\Effective_Connectivity\\EEG\\Data_EEG_fNIRS_Suturing\\EEG_preprocessed_sept27\\MNI_Coordinates\\IC_MNI_Coordinates.xlsx",sheet = "E014") #
# Result <- t(mapply(FUN = mni_to_region_name, x = m$x, y = m$y, z = m$z))
# print(Result[ , c(1, 2)])
# 
# m <- NULL 
# m = read_excel("C:\\Users\\_Kamat_\\Desktop\\RPI\\ResearchWork\\Papers_\\Effective_Connectivity\\EEG\\Data_EEG_fNIRS_Suturing\\EEG_preprocessed_sept27\\MNI_Coordinates\\IC_MNI_Coordinates.xlsx",sheet = "E015") #
# Result <- t(mapply(FUN = mni_to_region_name, x = m$x, y = m$y, z = m$z))
# print(Result[ , c(1, 2)])
# 
# 
# print(Result[ , c(1, 2)])
# 
# 
# m <- NULL 
# m = read_excel("C:\\Users\\_Kamat_\\Desktop\\RPI\\ResearchWork\\Papers_\\Effective_Connectivity\\EEG\\Data_EEG_fNIRS_Suturing\\EEG_preprocessed_sept27\\MNI_Coordinates\\IC_MNI_Coordinates.xlsx",sheet = "N010") #
# Result <- t(mapply(FUN = mni_to_region_name, x = m$x, y = m$y, z = m$z))
# print(Result[ , c(1, 2)])
# 
# m <- NULL 
# m = read_excel("C:\\Users\\_Kamat_\\Desktop\\RPI\\ResearchWork\\Papers_\\Effective_Connectivity\\EEG\\Data_EEG_fNIRS_Suturing\\EEG_preprocessed_sept27\\MNI_Coordinates\\IC_MNI_Coordinates.xlsx",sheet = "N011") #
# Result <- t(mapply(FUN = mni_to_region_name, x = m$x, y = m$y, z = m$z))
# print(Result[ , c(1, 2)])
# 
# m <- NULL 
# m = read_excel("C:\\Users\\_Kamat_\\Desktop\\RPI\\ResearchWork\\Papers_\\Effective_Connectivity\\EEG\\Data_EEG_fNIRS_Suturing\\EEG_preprocessed_sept27\\MNI_Coordinates\\IC_MNI_Coordinates.xlsx",sheet = "N012") #
# Result <- t(mapply(FUN = mni_to_region_name, x = m$x, y = m$y, z = m$z))
# print(Result[ , c(1, 2)])
# 
# m <- NULL 
# m = read_excel("C:\\Users\\_Kamat_\\Desktop\\RPI\\ResearchWork\\Papers_\\Effective_Connectivity\\EEG\\Data_EEG_fNIRS_Suturing\\EEG_preprocessed_sept27\\MNI_Coordinates\\IC_MNI_Coordinates.xlsx",sheet = "N013") #
# Result <- t(mapply(FUN = mni_to_region_name, x = m$x, y = m$y, z = m$z))
# print(Result[ , c(1, 2)])
# 
# m <- NULL 
# m = read_excel("C:\\Users\\_Kamat_\\Desktop\\RPI\\ResearchWork\\Papers_\\Effective_Connectivity\\EEG\\Data_EEG_fNIRS_Suturing\\EEG_preprocessed_sept27\\MNI_Coordinates\\IC_MNI_Coordinates.xlsx",sheet = "N014") #
# Result <- t(mapply(FUN = mni_to_region_name, x = m$x, y = m$y, z = m$z))
# print(Result[ , c(1, 2)])
# 
# m <- NULL 
# m = read_excel("C:\\Users\\_Kamat_\\Desktop\\RPI\\ResearchWork\\Papers_\\Effective_Connectivity\\EEG\\Data_EEG_fNIRS_Suturing\\EEG_preprocessed_sept27\\MNI_Coordinates\\IC_MNI_Coordinates.xlsx",sheet = "N015") #
# Result <- t(mapply(FUN = mni_to_region_name, x = m$x, y = m$y, z = m$z))
# print(Result[ , c(1, 2)])