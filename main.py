import PySimpleGUI as sg
import matplotlib.pyplot as plt
from cv2 import cv2

sg.theme('Dark Blue 3')



layout = [[sg.Text('Body Recognition')],
        [sg.Text('Source an Image', size=(15, 1))],
        [sg.InputText(key='001'), sg.FileBrowse()],
        [sg.Text('Select a part of the body')],
        [sg.InputCombo(('Head', 'Neck', 'Hands', 'Foot'), size=(15, 1), key='point')],


        [sg.Submit(), sg.Cancel()]]

window = sg.Window('Body Recognition', layout)

##### Getting the structure #####

file_proto = "content/unzip_files/pose/body/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
file_weight = "content/unzip_files/pose/body/mpi/pose_iter_160000.caffemodel"

#################################

event, values = window.read()
window.close()

image_source = cv2.imread(values['001'])
image_width = image_source.shape[1]
image_height = image_source.shape[0]

model = cv2.dnn.readNetFromCaffe(file_proto, file_weight)

input_height = 368
input_width = int((input_height / image_height) * image_width)

input_blob = cv2.dnn.blobFromImage(image = image_source, scalefactor= 1.0 / 255,
                                    size = (input_height, input_width),
                                    mean = (0, 0, 0), swapRB = False, crop= False)

model.setInput(input_blob)
out = model.forward()

print(values['point'])
if values['point'] == 'Head':
    point = 0
elif values['point'] == 'Neck':
    point = 1
elif values['point'] == 'Hands':
    point = 4
elif values['point'] == 'Foot':
    point = 10

trust_map = out[0, point, :, :]
trust_map = cv2.resize(trust_map, (image_width, image_height))

plt.figure(figsize = [14, 10])
plt.imshow(cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB))
plt.imshow(trust_map, alpha = 0.6)
plt.show()