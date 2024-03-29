import numpy as np
from scipy.optimize import linear_sum_assignment
from facelibuz.utils.misc import iou_xywh as iou
from facelibuz.utils.track import KFTrack4DSORT
from facelibuz.utils.centroid_kf_tracker import CentroidKF_Tracker


def assign_tracks2detection_iou(bbox_tracks, bbox_detections, iou_threshold=0.3):
    """
    Assigns detected bounding boxes to tracked bounding boxes using IoU as a distance metric.

    Parameters
    ----------
    bbox_tracks : numpy.ndarray
    bbox_detections : numpy.ndarray
    iou_threshold : float

    Returns
    -------
    tuple :
        Tuple containing the following elements
            - matches: (numpy.ndarray) Array of shape `(n, 2)` where `n` is number of pairs formed after
                matching tracks to detections. This is an array of tuples with each element as matched pair
                of indices`(track_index, detection_index)`.
            - unmatched_detections : (numpy.ndarray) Array of shape `(m,)` where `m` is number of unmatched detections.
            - unmatched_tracks : (numpy.ndarray) Array of shape `(k,)` where `k` is the number of unmatched tracks.
    """

    unmatched_detections, unmatched_tracks = [], []
    for t in range(bbox_tracks.shape[0]):
            unmatched_tracks.append(t)
    if (bbox_detections.size == 0) and (bbox_tracks.size > 0):
        return np.empty((0, 2), dtype=int), np.arange(len(bbox_detections), dtype=int), np.array(unmatched_tracks)
    if (bbox_tracks.size == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(bbox_detections), dtype=int), np.empty((0,), dtype=int)

    if len(bbox_tracks.shape) == 1:
        bbox_tracks = bbox_tracks[None, :]

    if len(bbox_detections.shape) == 1:
        bbox_detections = bbox_detections[None, :]

    iou_matrix = np.zeros((bbox_tracks.shape[0], bbox_detections.shape[0]), dtype=np.float32)
    for t in range(bbox_tracks.shape[0]):
        for d in range(bbox_detections.shape[0]):
            iou_matrix[t, d] = iou(bbox_tracks[t, :], bbox_detections[d, :])

    assigned_tracks, assigned_detections = linear_sum_assignment(-iou_matrix)

    unmatched_detections, unmatched_tracks = [], []
    for d in range(bbox_detections.shape[0]):
        if d not in assigned_detections:
            unmatched_detections.append(d)

    for t in range(bbox_tracks.shape[0]):
        if t not in assigned_tracks:
            unmatched_tracks.append(t)

    # filter out matched with low IOU
    matches = []
    for t, d in zip(assigned_tracks, assigned_detections):
        if iou_matrix[t, d] < iou_threshold:
            unmatched_detections.append(d)
            unmatched_tracks.append(t)
        else:
            matches.append((t, d))

    if len(matches):
        matches = np.array(matches)
    else:
        matches = np.empty((0, 2), dtype=int)

    return matches, np.array(unmatched_detections), np.array(unmatched_tracks)


class SORT(CentroidKF_Tracker):
    """
    SORT - Multi object tracker.

    Parameters
    ----------
    max_lost : int
        Max. number of times a object is lost while tracking.
    tracker_output_format : str
        Output format of the tracker.
    iou_threshold : float
        Intersection over union minimum value.
    process_noise_scale : float or numpy.ndarray
        Process noise covariance matrix of shape (3, 3) or covariance magnitude as scalar value.
    measurement_noise_scale : float or numpy.ndarray
        Measurement noise covariance matrix of shape (1,) or covariance magnitude as scalar value.
    time_step : int or float
        Time step for Kalman Filter.
    """

    def __init__(
            self, max_lost=0,
            tracker_output_format='mot_challenge',
            iou_threshold=0.3,
            process_noise_scale=1.0,
            measurement_noise_scale=1.0,
            time_step=1
    ):
        self.iou_threshold = iou_threshold

        super().__init__(
            max_lost=max_lost, tracker_output_format=tracker_output_format,
            process_noise_scale=process_noise_scale,
            measurement_noise_scale=measurement_noise_scale, time_step=time_step
        )

    def _add_track(self, frame_id, bbox, detection_confidence, class_id, **kwargs):
        self.tracks[self.next_track_id] = KFTrack4DSORT(
            self.next_track_id, frame_id, bbox, detection_confidence, class_id=class_id,
            data_output_format=self.tracker_output_format, process_noise_scale=self.process_noise_scale,
            measurement_noise_scale=self.measurement_noise_scale, kf_time_step=1, **kwargs)
        self.next_track_id += 1

    def update(self, bboxes, kps_list, class_ids):
        self.frame_count += 1
        
        bbox_detections = np.array(bboxes, dtype='int')

        track_ids = list(self.tracks.keys())
        
        bbox_tracks = []
        for track_id in track_ids:
            bb = self.tracks[track_id].predict()
            bbox_tracks.append(bb)

        bbox_tracks = np.array(bbox_tracks)
        
        
        # if len(bbox_tracks)>0 and len(bbox_detections)==0:
        #     matches, unmatched_detections, unmatched_tracks = np.empty((0, 2), dtype=int), np.arange(len(bbox_detections), dtype=int), np.array(unmatched_tracks)
        # else:
        matches, unmatched_detections, unmatched_tracks = assign_tracks2detection_iou(
            bbox_tracks, bbox_detections, iou_threshold=0.3)
        for i in range(matches.shape[0]):
            t, d = matches[i, :]
            track_id = track_ids[t]
            bbox = bboxes[d, :]
            cid = class_ids[d]
            confidence = kps_list[d]
            self._update_track(track_id, self.frame_count, bbox, confidence, cid, lost=0)
            
        for d in unmatched_detections:
            bbox = bboxes[d, :]
            cid = class_ids[d]
            confidence = kps_list[d]
            self._add_track(self.frame_count, bbox, confidence, cid)

        for t in unmatched_tracks:
            track_id = track_ids[t]
            bbox = bbox_tracks[t, :]
            confidence = self.tracks[track_id].detection_confidence
            cid = self.tracks[track_id].class_id
            self._update_track(track_id, self.frame_count, bbox, detection_confidence=confidence, class_id=cid, lost=self.tracks[track_id].lost+1)
            if self.tracks[track_id].lost > self.max_lost:
                self._remove_track(track_id)

        outputs = self._get_tracks(self.tracks)
        return outputs
