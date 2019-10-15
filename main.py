import cv2
import os

dir = os.getcwd()
dirIn = dir + "/input"
dirOut = dir + "/output"
dirSrc = dir + "/src"
inputFile = ""

try:
    os.mkdir("input")
    os.mkdir("output")
    print("Directories Located Created")
except:
    print("Directories Located")

for r,d,f in os.walk("./input"):
    inputFile = f[0]
    break

print(inputFile)

#Places the faces

def placeFace():
    faceArea = inputFace[y:y+h,x:x+w]
    sizedCage = cv2.resize(cageFace,(faceArea.shape[0],faceArea.shape[1]))
    sizedCageA = cv2.resize(alpha,(faceArea.shape[0],faceArea.shape[1]))

    sizedCage = cv2.multiply(sizedCageA,sizedCage)
    faceArea = cv2.multiply(1-sizedCageA,faceArea)
    faceArea = cv2.add(faceArea,sizedCage)
    inputFace[y:y+faceArea.shape[0],x:x+faceArea.shape[1]] = faceArea

#Opens the presample of faces provided by openCV
haarCascade = cv2.CascadeClassifier("src/haarcascade_frontalface_default.xml")

cageFace = cv2.imread("src/cageface.png")
inputFace = cv2.imread("input/"+inputFile)
alpha = cv2.imread("src/cagefaceAlpha.png")


#In order to make edge dection comparisons image must be set to greyscale
greyInput = cv2.cvtColor(inputFace, cv2.COLOR_BGR2GRAY)


inputFace = inputFace.astype(float)
cageFace = cageFace.astype(float)

alpha = alpha.astype(float)/255

face = haarCascade.detectMultiScale(greyInput,scaleFactor=1.2,minNeighbors=5)

c = 0

for (x,y,w,h) in face:
    c=c+1

    placeFace()

   # print("Face ",c)

cv2.imshow('input',inputFace/255)
cv2.imwrite("output/output.jpg",inputFace)
cv2.waitKey(0)
cv2.destroyAllWindows()