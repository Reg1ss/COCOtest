import cv2

fps = 15
size = (1920, 1080)
videowriter = cv2.VideoWriter("/home/kuangrx/TopVT/video/test_0003.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)

for i in range(1, 181):
    if(i<10):
        img = cv2.imread('/home/kuangrx/frames/0003_sub/00000000%d.jpg' % i)
    if(9<i<100):
        img = cv2.imread('/home/kuangrx/frames/0003_sub/0000000%d.jpg' % i)
    if(99<i<1000):
        img = cv2.imread('/home/kuangrx/frames/0003_sub/000000%d.jpg' % i)
    videowriter.write(img)