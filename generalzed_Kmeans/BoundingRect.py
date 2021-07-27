class BoundingRect:

    def __init__(self, width, height, angle, anchor):
        self.width = width
        self.height = height
        self.angle = angle
        self.anchor = anchor


class BoundingEllipse:

    def __init__(self, center, width, height, angle):
        self.center = center
        self.width = width
        self.height = height
        self.angle = angle
