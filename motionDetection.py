import cv2
import time
import datetime
import imutils

def motion_detection():
    video_capture = cv2.VideoCapture(0)  # value (0) selects the device's default camera
    time.sleep(2)

    first_frame = None  # instantiate the first frame

    while True:
        frame = video_capture.read()[1]  # gives 2 outputs retval, frame - [1] selects frame
        text = 'Unoccupied'

        greyscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # make each frame grayscale which is needed for threshold

        gaussian_frame = cv2.GaussianBlur(greyscale_frame, (21, 21), 0)
        blur_frame = cv2.blur(gaussian_frame, (5, 5))  # uses a kernel of size(5,5)

        greyscale_image = blur_frame  # grayscale image with blur etc which is the final image ready to be used for threshold and motion detection

        if first_frame is None:
            first_frame = greyscale_image  # first frame is set for background subtraction(BS)
        else:
            pass

        frame = imutils.resize(frame, width=500)
        frame_delta = cv2.absdiff(first_frame, greyscale_image)

        # edit the ** thresh ** depending on the light/dark in the room, change the 100(anything pixel value over 100 will become 255(white))
        thresh = cv2.threshold(frame_delta, 100, 255, cv2.THRESH_BINARY)[1]

        dilate_image = cv2.dilate(thresh, None, iterations=2)

        cnt = cv2.findContours(dilate_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        for c in cnt:
            if cv2.contourArea(c) > 800:  # if contour area is less than 800 non-zero(not-black) pixels(white)
                (x, y, w, h) = cv2.boundingRect(c)

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # (0, 255, 0) = color R,G,B = lime / 2 = thickness
                text = 'Occupied'

        # draw text and timestamp on the security feed
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(frame, '{+} Room Status: %s' % (text),
                    (10, 20), font, 0.5, (0, 0, 255), 2)

        cv2.putText(frame, datetime.datetime.now().strftime('%A %d %B %Y %I:%M:%S%p'),
                    (10, frame.shape[0] - 10), font, 0.35, (0, 0, 255), 1)

        cv2.imshow('Security Feed', frame)
        cv2.imshow('Threshold(foreground mask)', dilate_image)
        cv2.imshow('Frame_delta', frame_delta)

        key = cv2.waitKey(1) & 0xFF
        # if statement; press q to exit the app
        if key == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    motion_detection()
