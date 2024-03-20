class TrackableObject:
	def __init__(self, objectID, centroid, path_length=0):
		self.objectID = objectID
		self.centroids = [centroid]
		self.path_length=path_length
		self.startY = centroid[1]
		self.name = "unknown"
		self.recognized = False
		self.score = 0
		self.bbox = None
		self.kps = None
		self.lost_count = 0
		self.live = False

		self.counted = False
