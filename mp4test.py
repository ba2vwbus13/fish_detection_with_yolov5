import cv2

print("---start---")

#動画ファイルを読み込む
video = cv2.VideoCapture("/media/nakahira/GOLDUSB/fishesforNHK/tsunodashi.mp4")

# 幅と高さを取得
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (width, height)

#総フレーム数を取得
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

#フレームレート(1フレームの時間単位はミリ秒)の取得
frame_rate = int(video.get(cv2.CAP_PROP_FPS))

# 保存用
fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
writer = cv2.VideoWriter('./outtest.mp4', fmt, frame_rate, size)

for i in range(frame_count):
    ret, frame = video.read()
    ### ここに加工処理などを記述する ###
    writer.write(frame)

writer.release()
video.release()
cv2.destroyAllWindows()

print("---end---")