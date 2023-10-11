import abc
from collections import defaultdict
import numpy as np

from nuscenes.eval.detection.constants import ATTRIBUTE_NAMES, TP_METRICS

# DETECTION_NAMES = {"Car", "Pedestrian", "Bicycle"}
DETECTION_NAMES = {"Car"}

class EvalBox(abc.ABC):
    """ Abstract base class for data classes used during detection evaluation. Can be a prediction or ground truth."""

    def __init__(self,
                 sample_token="",
                 translation=(0, 0, 0),
                 size=(0, 0, 0),
                 rotation=0,
                 velocity=(0, 0),
                 ego_translation=(0, 0, 0),  # Translation to ego vehicle in meters.
                 num_pts=-1):  # Nbr. LIDAR or RADAR inside the box. Only for gt boxes.

        # Assert data for shape and NaNs.
        assert type(sample_token) == str, 'Error: sample_token must be a string!'

        assert len(translation) == 3, 'Error: Translation must have 3 elements!'
        assert not np.any(np.isnan(translation)), 'Error: Translation may not be NaN!'

        assert len(size) == 3, 'Error: Size must have 3 elements!'
        assert not np.any(np.isnan(size)), 'Error: Size may not be NaN!'

        assert len(rotation) == 1, 'Error: Rotation must have 1 elements!'
        assert not np.any(np.isnan(rotation)), 'Error: Rotation may not be NaN!'

        # Velocity can be NaN from our database for certain annotations.
        assert len(velocity) == 2, 'Error: Velocity must have 2 elements!'

        assert len(ego_translation) == 3, 'Error: Translation must have 3 elements!'
        assert not np.any(np.isnan(ego_translation)), 'Error: Translation may not be NaN!'

        assert type(num_pts) == int, 'Error: num_pts must be int!'
        assert not np.any(np.isnan(num_pts)), 'Error: num_pts may not be NaN!'

        # Assign.
        self.sample_token = sample_token
        self.translation = translation
        self.size = size
        self.rotation = rotation
        self.velocity = velocity
        self.ego_translation = ego_translation
        self.num_pts = num_pts

    @property
    def ego_dist(self) -> float:
        """ Compute the distance from this box to the ego vehicle in 2D. """
        return np.sqrt(np.sum(np.array(self.ego_translation[:2]) ** 2))

    def __repr__(self):
        return str(self.serialize())

    @abc.abstractmethod
    def serialize(self):
        pass

    @classmethod
    @abc.abstractmethod
    def deserialize(cls, content):
        pass

class EvalBoxes:
    """ Data class that groups EvalBox instances by sample. """

    def __init__(self):
        """
        Initializes the EvalBoxes for GT or predictions.
        """
        self.boxes = defaultdict(list)

    def __repr__(self):
        return "EvalBoxes with {} boxes across {} samples".format(len(self.all), len(self.sample_tokens))

    def __getitem__(self, item):
        return self.boxes[item]

    def __eq__(self, other):
        if not set(self.sample_tokens) == set(other.sample_tokens):
            return False
        for token in self.sample_tokens:
            if not len(self[token]) == len(other[token]):
                return False
            for box1, box2 in zip(self[token], other[token]):
                if box1 != box2:
                    return False
        return True

    def __len__(self):
        return len(self.boxes)

    @property
    def all(self):
        """ Returns all EvalBoxes in a list. """
        ab = []
        for sample_token in self.sample_tokens:
            ab.extend(self[sample_token])
        return ab

    @property
    def sample_tokens(self):
        """ Returns a list of all keys. """
        return list(self.boxes.keys())

    def add_boxes(self, sample_token, boxes):
        """ Adds a list of boxes. """
        self.boxes[sample_token].extend(boxes)

    def serialize(self):
        """ Serialize instance into json-friendly format. """
        return {key: [box.serialize() for box in boxes] for key, boxes in self.boxes.items()}

    @classmethod
    def deserialize(cls, content, box_cls, include_cls=None):
        """
        Initialize from serialized content.
        :param content: A dictionary with the serialized content of the box.
        :param box_cls: The class of the boxes, DetectionBox or TrackingBox.
        """
        eb = cls()
        if type(content) is dict:
            for sample_token, boxes in content.items():
                pred_boxes = boxes["pred_boxes"]
                pred_names = boxes["pred_names"]
                pred_scores = boxes["pred_scores"]

                pred_velocity = None
                if "pred_velocity" in boxes:
                    pred_velocity = boxes["pred_velocity"]

                boxes_list = []
                for box_idx in range(len(pred_boxes)):
                    bbox = pred_boxes[box_idx]
                    name = pred_names[box_idx]
                    score = pred_scores[box_idx]
                    if pred_velocity is not None:
                        velocity = pred_velocity[box_idx]

                    if include_cls is not None:
                        if name not in include_cls:
                            continue

                    boxes_list.append({
                        "sample_token": sample_token,
                        "translation": tuple(bbox[:3]),
                        "size": tuple(bbox[3:6]),
                        "rotation": bbox[6:],
                        "velocity": (0, 0) if pred_velocity is None else tuple(velocity[:2]),
                        "detection_name": name,
                        "detection_score": score,
                        "attribute_name": ""
                    })
                eb.add_boxes(sample_token, [box_cls.deserialize(box) for box in boxes_list])
        return eb


class DetectionBox(EvalBox):
    """ Data class used during detection evaluation. Can be a prediction or ground truth."""

    def __init__(self,
                 sample_token="",
                 translation=(0, 0, 0),
                 size=(0, 0, 0),
                 rotation=0,
                 velocity= (0, 0),
                 ego_translation= (0, 0, 0),  # Translation to ego vehicle in meters.
                 num_pts=-1,  # Nbr. LIDAR or RADAR inside the box. Only for gt boxes.
                 detection_name='Car',  # The class name used in the detection challenge.
                 detection_score=-1.0,  # GT samples do not have a score.
                 attribute_name=''):  # Box attribute. Each box can have at most 1 attribute.

        super().__init__(sample_token, translation, size, rotation, velocity, ego_translation, num_pts)

        # if detection_name == 'pedestrian':
        #     detection_name = 'Pedestrian'
        assert detection_name is not None, 'Error: detection_name cannot be empty!'
        assert detection_name in DETECTION_NAMES, 'Error: Unknown detection_name %s' % detection_name

        assert attribute_name in ATTRIBUTE_NAMES or attribute_name == '', \
            'Error: Unknown attribute_name %s' % attribute_name

        assert type(detection_score) == float, 'Error: detection_score must be a float!'
        assert not np.any(np.isnan(detection_score)), 'Error: detection_score may not be NaN!'

        # Assign.
        self.detection_name = detection_name
        self.detection_score = detection_score
        self.attribute_name = attribute_name

    def __eq__(self, other):
        return (self.sample_token == other.sample_token and
                self.translation == other.translation and
                self.size == other.size and
                self.rotation == other.rotation and
                self.velocity == other.velocity and
                self.ego_translation == other.ego_translation and
                self.num_pts == other.num_pts and
                self.detection_name == other.detection_name and
                self.detection_score == other.detection_score and
                self.attribute_name == other.attribute_name)

    def serialize(self):
        """ Serialize instance into json-friendly format. """
        return {
            'sample_token': self.sample_token,
            'translation': self.translation,
            'size': self.size,
            'rotation': self.rotation,
            'velocity': self.velocity,
            'ego_translation': self.ego_translation,
            'num_pts': self.num_pts,
            'detection_name': self.detection_name,
            'detection_score': self.detection_score,
            'attribute_name': self.attribute_name
        }

    @classmethod
    def deserialize(cls, content):
        """ Initialize from serialized content. """
        return cls(sample_token=content['sample_token'],
                   translation=tuple(content['translation']),
                   size=tuple(content['size']),
                   rotation=tuple(content['rotation']),
                   velocity=tuple(content['velocity']),
                   ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
                   else tuple(content['ego_translation']),
                   num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
                   detection_name=content['detection_name'],
                   detection_score=-1.0 if 'detection_score' not in content else float(content['detection_score']),
                   attribute_name=content['attribute_name'])
