import glob
import os


class BighandDatasetBase:

    def __init__(self, dataset_path, annotation_folder, test_subject=None, batch_size=16):
        self.dataset_path = dataset_path
        self.test_subject = test_subject
        self.batch_size = batch_size

        self.train_annotation_files, self.test_annotation_files = self._load_annotations(annotation_folder)
        self.train_annotations = self._count_annotations(self.train_annotation_files)
        self.test_annotations = self._count_annotations(self.test_annotation_files)
        self.num_train_batches = int(self.train_annotations // self.batch_size)
        self.num_test_batches = int(self.test_annotations // self.batch_size)

    def _count_annotations(self, annotation_files):
        def file_lines(filename):
            with open(filename) as f:
                for i, l in enumerate(f):
                    pass
            return i + 1

        counts = [file_lines(filename) for filename in annotation_files]
        return sum(counts)

    def _load_annotations(self, annotations_folder):
        subject_dirs = self._get_subject_dirs()
        train_annotation_files = []
        test_annotation_files = []
        for subject_dir in subject_dirs:
            pattern = F"{annotations_folder}/{subject_dir}/[!README]*.txt"
            full_pattern = os.path.join(self.dataset_path, pattern)
            annotation_files = glob.glob(full_pattern)

            if subject_dir == self.test_subject:
                test_annotation_files += annotation_files
            else:
                train_annotation_files += annotation_files
        train_annotation_files = [annot for annot in train_annotation_files if 'Subject_4/76 150' in annot]
        return train_annotation_files, test_annotation_files

    def _get_subject_dirs(self):
        return [f.stem for f in self.dataset_path.iterdir() if f.is_dir()]
