import cv2
import numpy as np
from os import listdir#os module class used for fetching data from directory
from os.path import isfile, join

data_path = 'E:/faces/'
onlyfiles=[f for f in listdir(data_path) if isfile(join(data_path,f))]

Training_Data, Labels = [], []

for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

Labels = np.asarray(Labels, dtype=np.int32)

#model building
model = cv2.face.LBPHFaceRecognizer_create()#linear binary phase histogram face recorganizer

model.train(np.asarray(Training_Data),np.asarray(Labels))

print("Training Model is Completed!!!!!")
model.write('E:/trained.yml')
