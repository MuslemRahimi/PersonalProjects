import pydicom as dicom
import matplotlib.pylab as plt

# specify your image path
image_path = 'inbreast/ALL-IMGS/20586934_6c613a14b80a8591_MG_L_CC_ANON.dcm'
ds = dicom.dcmread(image_path)

plt.imshow(ds.pixel_array)
plt.show()