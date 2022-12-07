import insightface
from insightface.app import FaceAnalysis
import matplotlib.pyplot as plt


def get_embedding(faces):
    embeddings = []
    for index in range(len(faces)):
        embedding = faces[index]['embedding']
        embedding = embedding.reshape(-1, 1)
        embeddings.append(embeddings)
    return embeddings

def get_bbox_insightface(faces):
    res_faces = []
    for index in range(len(faces)):
        res = tuple(faces[index]['bbox'])
        res = tuple(map(round, res))
        res = (max(res[1],0), max(res[3],0), max(res[0], 0), max(res[2],0))
        res_faces.append(res)
    return res_faces

def detect_face_ins(img):
    ins_detector = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    ins_detector.prepare(ctx_id=0, det_size=(640, 640))
    faces = ins_detector.get(img)
    embeddings = get_embedding(faces)
    res_faces = get_bbox_insightface(faces)
    return res_faces, embeddings

def get_roi(image, coordinates):
    # Check negative coordinates
    top = coordinates[0] if coordinates[0] - 5 < 0 else coordinates[0] - 5 
    bottom = coordinates[1] + 5
    left = coordinates[2] if coordinates[2] - 5 < 0 else coordinates[2] - 5
    right = coordinates[3] + 5
    # roi = image[coordinates[0]-5:coordinates[1]+5, coordinates[2]-5:coordinates[3]+5]
    roi = image[top:bottom,left:right]
    return roi

def draw_boundingbox(ax, bbox, names):
    # plot each box
    index = 0
    count = 0 
    dictionary = {}
    # Change here
    for result in bbox:
        if names[index] == 'Others' or names[index] == 'NotFace':
            index += 1
            continue
        # get coordinates
        x, y, width, height = result
        # create the shape
        rect = plt.Rectangle((x, y), width, height, fill=False, color='red')
        # draw the box
        ax.add_patch(rect)
        ax.annotate(count, (x, y), color='white', weight='bold', fontsize=15, ha='center', va='center')
        dictionary[count] = names[index]
        index += 1
        count += 1
    return ax, dictionary
        
