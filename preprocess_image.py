import cv2 
import numpy as np

class Preprocessor():
    def __init__(self):
        self.hsv_vars = [
            0, 51, 123,
            15, 145, 255,
        ]
        self.kernel_vars = [
            1, 1, 1, 1
        ]

    def preprocess_image(self, image):
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
        lower_blue = np.array([self.hsv_vars[0], self.hsv_vars[1], self.hsv_vars[2]]) 
        upper_blue = np.array([self.hsv_vars[3], self.hsv_vars[4], self.hsv_vars[5]]) 
        mask = cv2.inRange(hsv, lower_blue, upper_blue) 
        result = cv2.bitwise_and(frame, frame, mask = mask) 

        erode_kernel = np.ones((self.kernel_vars[0], self.kernel_vars[0]), np.uint8) 
        dilate_kernel = np.ones((self.kernel_vars[1], self.kernel_vars[1]), np.uint8) 
                
        result = cv2.erode(result, erode_kernel, iterations=self.kernel_vars[2]) 
        result = cv2.dilate(result, dilate_kernel, iterations=self.kernel_vars[3]) 

        return result

    def hsv_callback(self, val, index):
        self.hsv_vars[index] = val
        
    def kernel_callback(self, val, index):
        self.kernel_vars[index] = (val*2)+1

preprocessor = Preprocessor()
cv2.namedWindow('Image')
cv2.createTrackbar('lower_h', 'Image', preprocessor.hsv_vars[0], 255, lambda x: preprocessor.hsv_callback(x, 0))
cv2.createTrackbar('lower_s', 'Image', preprocessor.hsv_vars[1], 255, lambda x: preprocessor.hsv_callback(x, 1))
cv2.createTrackbar('lower_v', 'Image', preprocessor.hsv_vars[2], 255, lambda x: preprocessor.hsv_callback(x, 2))
cv2.createTrackbar('upper_h', 'Image', preprocessor.hsv_vars[3], 255, lambda x: preprocessor.hsv_callback(x, 3))
cv2.createTrackbar('upper_s', 'Image', preprocessor.hsv_vars[4], 255, lambda x: preprocessor.hsv_callback(x, 4))
cv2.createTrackbar('upper_v', 'Image', preprocessor.hsv_vars[5], 255, lambda x: preprocessor.hsv_callback(x, 5))
# cv2.createTrackbar('erode_kernel', 'Image', preprocessor.kernel_vars[0], 21, lambda x: preprocessor.kernel_callback(x, 0))
# cv2.createTrackbar('erode_iterations', 'Image', preprocessor.kernel_vars[2], 21, lambda x: preprocessor.kernel_callback(x, 2))
# cv2.createTrackbar('dilate_kernel', 'Image', preprocessor.kernel_vars[1], 21, lambda x: preprocessor.kernel_callback(x, 1))
# cv2.createTrackbar('dilate_iterations', 'Image', preprocessor.kernel_vars[3], 21, lambda x: preprocessor.kernel_callback(x, 3))



webcam = cv2.VideoCapture(0) 

while(True):
    ret, frame = webcam.read() 

    frame = preprocessor.preprocess_image(frame)
     
    cv2.imshow('Image', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
webcam.release() 
cv2.destroyAllWindows() 
