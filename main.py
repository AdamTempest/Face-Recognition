from facenet_pytorch import MTCNN, extract_face, InceptionResnetV1
from PIL import Image, ImageDraw

mtcnn = MTCNN(select_largest=False,keep_all=True)
#      keep_all {bool} -- If True, all detected faces are returned, in the order dictated by the
#          select_largest parameter. If a save_path is specified, the first face is saved to that
#          path and the remaining faces are saved to <save_path>1, <save_path>2 etc.
#          (default: {False})

print("[+] Place your image in images folder.")
filename = input("Enter the name of the image: ")
img = Image.open(f'images/{filename}') # get PIL.Image
boxes, probs, points = mtcnn.detect(img,landmarks=True)
# Draw boxes and save faces
img_draw = img.copy()
draw = ImageDraw.Draw(img_draw)

# i {int}, (box {array}, point {array})
for i, (box,point) in enumerate(zip(boxes, points)):
    draw.rectangle(box.tolist(),width=5)
    for p in point:
        draw.rectangle((p-10).tolist() + (p+10).tolist(), width=1)
    extract_face(img, box, save_path='saved/detected_face_{}.png'.format(i))

img_draw.save('Result.png')

# DOCUMENT
# help(MTCNN)
