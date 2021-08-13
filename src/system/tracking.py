class HandTracker():

    def __init__(self, image_source, detector, estimator):
        self.image_source = image_source
        self.detector = detector
        self.estimator = estimator
        self.keypoints = None

    def track(self):
        while True:
            image = next(self.image_source)
            if self.keypoints is None:
                detections = self.detector.detect(image)

            reject_hand_flag, keypoints = self.estimator.estimate(image, detections)

            if reject_hand_flag:
                self.keypoints = None
            else:
                self.render_image(image, keypoints)
                self.keypoints = keypoints
                detections = self.keypoints_to_rectangle(keypoints)

    def render_image(self, image, keypoints):
        pass

    def keypoints_to_rectangle(self, keypoints):
        pass
