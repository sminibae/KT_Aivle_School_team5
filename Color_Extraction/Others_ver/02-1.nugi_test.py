#배경이 흰색인 사진을 누끼를 따준다.
from PIL import Image

# 누끼컷으로 변경할 이미지
img = Image.open('C:\\Users\\user\\amazon\\Detection\\team5\\Color_Extraction\\test_images\AAIRLLENSleekandSturdyInchComputerDeskPerfectforWorkandStudyMultiPurposeTableforWritingDiningandWorkstation.jpg') 
img = img.convert("RGBA")
datas = img.getdata()
newData = []
for item in datas:
    if item[0] == 255 and item[1] == 255 and item[2] == 255:
        newData.append((255, 255, 255, 0))
    else:
        newData.append(item)
img.putdata(newData)


# png 확장자로 저장
img.save("TransparentImage.png", "PNG") 
img.show()