import os


class DatasetBase:

    def __init__(self, dataset_path):
        self.dataset_path = str(dataset_path)

    def load_annotations(self, annotation_file_name, train_size):
        if train_size < 0 or train_size > 1:
            raise ValueError("Train_size expected to be in range [0, 1], but got {train_size}.")

        annotations_path = os.path.join(self.dataset_path, annotation_file_name)
        with open(annotations_path, 'r') as f:
            annotations = f.readlines()

        boundary_index = int(len(annotations) * train_size)
        return annotations[:boundary_index], annotations[boundary_index:]
