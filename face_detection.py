from mtcnn import MTCNN
import cv2

 


#for image
img = cv2.imread("/Users/noushinahmadvand/Documents/smile detection/654d56ac6e_109485_maryam-mirzakhani-stanford-university-02.jpg")
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

detector = MTCNN()

out = detector.detect_faces(rgb_img)[0] 

print(out)
print(out["box"])

# in mtcnn box is : x,y,w,h

x, y, w, h = out["box"]
kp = out["keypoints"]
confidence = out["confidence"]
print(kp)


cv2.rectangle(img, (x,y), (x+w, y+h) , (0,255,0), 2)

for key, value in kp.items(): # for seeing and circling keypoints
    cv2.circle(img ,value , 3 , (0,0,255), -1)
cv2.putText(img, f"cf: {confidence: 2f}", (x, y-10),  cv2.FONT_HERSHEY_SIMPLEX, 0.9 , (0,255,0))    

cv2.imshow("face", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
