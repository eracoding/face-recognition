import cv2
import logging
import os
import numpy as np

from time import time, sleep
from tqdm import tqdm
from facelibuz.app import FaceAnalysis
from log_handler import TelegramBotHandler, RecognitionScoreFilter

TELEGRAM_ID='7074068104:AAHqQPwzW9bA1D1CfTIStsbqFxQOnuc6X08'
CHAT_ID='5405613345'

KNOWN_PEOPLE = []
RECOGNIZED_LIST = set()

logging.basicConfig(
    level=logging.INFO,
    filename='info.log',
    format='%(name)s => %(levelname)s => %(message)s'
)

main_formatter = logging.Formatter(
    '%(levelname)s => %(message)s => %(asctime)s'
)

root_logger = logging.getLogger()

console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
console.setFormatter(main_formatter)

file = logging.FileHandler(filename="important_logs.txt")
file.setLevel(logging.ERROR)
file.setFormatter(main_formatter)

telegram = TelegramBotHandler(TELEGRAM_ID, CHAT_ID)
telegram.setLevel(logging.INFO)
telegram.setFormatter(main_formatter)
telegram.addFilter(RecognitionScoreFilter())


root_logger.addHandler(console)
root_logger.addHandler(file)
root_logger.addHandler(telegram)


def load_known_faces():
    global KNOWN_PEOPLE

    db_filename = 'db/facedb.npy'

    if os.path.exists(db_filename):
        KNOWN_PEOPLE = list(np.load(db_filename, allow_pickle=True))
        logging.info(f'loaded from file "{db_filename}"')

        return

    app = FaceAnalysis()
    app.prepare(ctx_id=0, det_thresh=0.5, det_size=(640,640))

    known_faces_dir = "known_people/"
    list_images = os.listdir(known_faces_dir)

    for img_name in tqdm(list_images):
        person_name = img_name.split(".")[0]
        image_path = os.path.join(known_faces_dir, img_name)
        image = cv2.imread(image_path)

        try:
            face_encodings = app.get(image)
        except Exception as ex:
            logging.error(img_name)
            logging.exception(ex)

        if len(face_encodings) == 0:
            logging.error("Cannot extract face feature from : {}".format(img_name))
            continue

        person = {}
        person["name"] = person_name
        person["embedding"] = face_encodings[0].embedding
        KNOWN_PEOPLE.append(person)

    np.save('db/facedb.npy', KNOWN_PEOPLE)

def main():
    cap = cv2.VideoCapture(SOURCE)

    while True:
        start_time = time()
        ret, frame = cap.read()
        faces = []

        if not ret:
            cap = cv2.VideoCapture(SOURCE)
            logging.info(f'REINITIALIZED SOURCE: {SOURCE}')
            sleep(3)
            continue
        
        faces = app.get(frame)
        if TRACKING:
            faces = app.trackableObjects
            for i in faces:
                face = faces.get(i, None)

                if face is None:
                    continue

                if (face.objectID not in RECOGNIZED_LIST) and (face.recognized):
                    RECOGNIZED_LIST.add(face.objectID)
                    logging.info(f'Recognized person: {face.name}', extra={'recognition_score': face.score})

                
        end_time = time()
        time_per_frame = end_time - start_time
        fps = 1 / time_per_frame
        logging.debug(f'!!! FPS: {fps}')

        frame = app.draw_on(frame, faces)

        cv2.imshow("test", cv2.resize(frame, (1280,768)))
 
        if ord('q')==cv2.waitKey(1):
            logging.error(f'UNPRECEDENTED ENGING OF THE SERVICE by clicking: {ord("q")}')
            exit(0)

if __name__ == '__main__':
    TRACKING = True
    SOURCE = 0 # built-in camera on ubuntu

    det_size = (640, 640)
    
    load_known_faces()

    app = FaceAnalysis(known_people=KNOWN_PEOPLE, tracking=TRACKING)
    app.prepare(ctx_id=0, rec_thresh=0.35, det_size=det_size)
    main()
