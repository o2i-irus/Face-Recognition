import cv2
import os
cam = cv2.VideoCapture(0)

ID=str(input('enter person name'))
num = int(input("No. of images you want to collect"))
trainPath = os.path.join("train",ID)
testPath = os.path.join("test")
sampleNum=0
makeDirectoryCommand = "mkdir -p \"{}\"".format(trainPath)
os.system(makeDirectoryCommand)
makeDirectoryCommand = "mkdir -p \"{}\"".format(testPath)
os.system(makeDirectoryCommand)
while(True):
    ret, img = cam.read()
    #incrementing sample number 
    sampleNum=sampleNum+1
    if sampleNum<=(num/5)*4:
        #saving the captured face in the dataset folder
        cv2.imwrite(trainPath + "/" + str(sampleNum) + ".jpg", img)
        print(trainPath + "/" + str(sampleNum) + ".jpg")

    elif sampleNum>=(num/5)*4 and sampleNum<=num:
        #saving the captured face in the dataset folder
        cv2.imwrite(testPath + "/" + str(sampleNum) + ".jpg", img)
        print(testPath + "/" + str(sampleNum) + ".jpg")

    cv2.imshow('frame',img)
    #wait for 100 miliseconds 
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    # break if the sample number is morethan 20
    elif sampleNum>num:
        break
cam.release()
cv2.destroyAllWindows()
