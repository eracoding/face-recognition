import cv2
import numpy as np
import onnxruntime
from scipy.spatial.distance import cosine

from facelibuz.model_zoo import model_zoo
from facelibuz.app.common import Face
from facelibuz.utils.sort_tracker import SORT
from facelibuz.utils.trackableobject import TrackableObject

onnxruntime.set_default_logger_severity(3)


class FaceAnalysis:

    def __init__(self, known_people=None, tracking=False):
        self.det_model = model_zoo.get_model('models/det_10g.onnx')
        self.rec_model = model_zoo.get_model('models/adaface.onnx')

        self.known_people=known_people

        self.tracker = None

        if tracking:
            self.tracker = SORT(max_lost=30, iou_threshold=0.3)
            self.trackableObjects = {}


    def prepare(
        self,
        ctx_id,
        det_thresh=0.7,
        rec_thresh=0.4,
        det_size=(640, 640),
    ):
        self.det_thresh = det_thresh
        self.rec_thresh = rec_thresh
        self.det_size = det_size

        self.det_model.prepare(ctx_id, input_size=det_size, det_thresh=det_thresh)
        self.rec_model.prepare(ctx_id)

    def find_face(self, embedding):
        best_matches = []

        for person in self.known_people:
            score = 1 - cosine(person['embedding'], embedding)

            if(score > self.rec_thresh):
                known = {}
                known['name']=person['name']
                known['score']= round(score, 2)
                best_matches.append(known)

        if len(best_matches) == 0:
            return None

        best_matches.sort(key=lambda x: -x['score'])

        return best_matches[0]

    def get(self, img, max_num=0):
        bboxes, kpss = self.det_model.detect(
            img,
            max_num=max_num,
            metric='default',
        )
        pass_point = int(img.shape[0]/3+0.5)
        faces = []
        rects = []
        kps_list = []

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None

            if kpss is not None:
                kps = kpss[i]

            if self.tracker is None:
                face = Face(bbox=bbox, kps=kps, det_score=det_score)
                self.rec_model.get(img, face)
                faces.append(face)
            else:
                landmarks = np.array(kps)
                landmarks = np.transpose(landmarks).reshape(10, -1)
                landmarks = np.transpose(landmarks)[0]

                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

                kps = landmarks.astype('int')
                rects.append([x1, y1, x2, y2])
                kps_list.append(kps)

        if self.tracker:
            objects = self.tracker.update(np.array(rects), np.array(kps_list), np.ones(len(rects)))

            for _, track_obj in self.trackableObjects.items():
                track_obj.live = False

            for obj in objects:
                objectID = obj[1]
                bbox = obj[2:6]
                kps = obj[6]

                facial5points = [[kps[j], kps[j + 5]] for j in range(5)]

                track_obj = self.trackableObjects.get(objectID, None)

                if track_obj is None:
                    track_obj = TrackableObject(objectID, obj)

                face = Face(bbox=np.array(bbox), kps=np.array(facial5points), det_score=1)
                track_obj.bbox = np.array(bbox)
                track_obj.kps = np.array(facial5points)
                track_obj.live = True
                track_obj.lost_count=0

                if not track_obj.recognized:
                    self.rec_model.get(img, face)

                    person = self.find_face(face.embedding)

                    if person:
                        if person['score']>track_obj.score:
                            track_obj.name=person['name']
                            track_obj.score = person['score']
                    box = track_obj.bbox.astype(np.int32)
                    center_y = int((box[1]+box[3])/2 + 0.5)
                    if center_y > pass_point and track_obj.score > 0:
                        track_obj.recognized = True

                self.trackableObjects[objectID] = track_obj

        return faces

    def draw_on(self, img, faces):
        dimg = img.copy()

        if self.tracker is None:
            for i in range(len(faces)):
                face = faces[i]
                box = face.bbox.astype(int)
                color = (0, 0, 255)
                cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
                if face.kps is not None:
                    kps = face.kps.astype(int)
                    #print(landmark.shape)
                    for l in range(kps.shape[0]):
                        color = (0, 0, 255)
                        if l == 0 or l == 3:
                            color = (0, 255, 0)
                        cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color,
                               2)
        else:
            for _, track_obj in self.trackableObjects.items():

                if not track_obj.live:
                    continue

                box = track_obj.bbox.astype(np.int32)

                color = (0, 0, 255)
                cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)

                person_info = [
                    f'objectID: {track_obj.objectID}',
                    track_obj.name,
                    f'score: {track_obj.score}',
                ]

                y0, dy = box[1] - 50, 23

                for i, line in enumerate(person_info):
                    y = y0 + i*dy

                    cv2.putText(
                        dimg,
                        line,
                        (box[0], y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        4,
                    )

        return dimg
