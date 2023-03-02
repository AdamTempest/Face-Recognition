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


        # array.tolist()

# DOCUMENT
# help(MTCNN)

# # For a model pretrained on VGGFace2
# model = InceptionResnetV1(pretrained='vggface2').eval()

#  image_path = "image/Picture.jpeg"
# img = Image.open(image_path) # get PIL.Image

# If required, create a face detection pipeline using MTCNN:
# size = img.size()[0] * img.size()[1] # get size in pixel
# margin = 10
# mtcnn = MTCNN(image_size=size, margin=margin

# # Create an inception resnet (in eval mode):
# resnet = InceptionResnetV1(pretrained='vggface2').eval()


# image_path = "image/Picture.jpeg"
# save_path = "saved_pics/"
# img = Image.open(image_path) # get PIL.Image

# Get cropped and prewhitened image tensor
# img_cropped = mtcnn(img, save_path=save_path)

# Calculate embedding (unsqueeze to add batch dimension)
# img_embedding = resnet(img_cropped.unsqueeze(0))

# # Or, if using for VGGFace2 classification
# resnet.classify = True
# img_probs = resnet(img_cropped.unsqueeze(0))