import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image



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
    img = img.convert("RGB")
    faces = ins_detector.get(img)
    embeddings = get_embedding(faces)
    res_faces = get_bbox_insightface(faces)
    return res_faces, embeddings
