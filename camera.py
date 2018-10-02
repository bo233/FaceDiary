
import cv2

clicked = False
def onMouse(event,x,y,flags,param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True

def strokeEdges(src,dst,blurKsize = 9,edgeKsize = 5):
    if blurKsize >= 3:
        blurredSrc = cv2.medianBlur(src,blurKsize)
        graySrc = cv2.cvtColor(blurredSrc,cv2.COLOR_BGR2GRAY)
    else:
        graySrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    cv2.Laplacian(graySrc, cv2.CV_8U, graySrc, ksize=edgeKsize)
    normalizedInverseAlpha = (1.0/255)*(255-graySrc)
    channels = cv2.split(src)
    for channel in channels:
        channel[:] = channel*normalizedInverseAlpha
    cv2.merge(channels, dst)

cameraCapture = cv2.VideoCapture(0)
cameraCapture.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
cameraCapture.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
cv2.namedWindow('MyWindow')
cv2.setMouseCallback('MyWindow', onMouse)
print('Showing camera feed.Click window or press any key to stop')

success, frame = cameraCapture.read()
while success and cv2.waitKey(1) == -1 and not clicked:
    strokeEdges(frame, frame)
    cv2.imshow('MyWindow', frame)
    success, frame = cameraCapture.read()

success, frame = cameraCapture.read()
clicked = False
while success and cv2.waitKey(1) == -1 and not clicked:
    ret, frame = cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 127, 255,  cv2.THRESH_BINARY)
    cv2.imshow('MyWindow', frame)
    success, frame = cameraCapture.read()

success, frame = cameraCapture.read()
clicked = False
while success and cv2.waitKey(1) == -1 and not clicked:
    frame = cv2.Canny(frame, 200, 300)
    cv2.imshow('MyWindow', frame)
    success, frame = cameraCapture.read()

cv2.destroyWindow('MyWindow')
cameraCapture.release()
