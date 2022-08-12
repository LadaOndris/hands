from src.estimation.blazepose.models.blazepose_full import BlazePoseFull
from src.estimation.blazepose.models.blazepose_light import BlazePoseLight


class ModelCreator:

    @staticmethod
    def create_model(model_name, n_points, n_point_features):
        name_parts = model_name.split('_')
        type = name_parts[0]
        outputs = name_parts[-1]

        if type == 'FULL':
            blaze_pose = BlazePoseFull(n_points, n_point_features)
        elif type == 'LIGHT':
            blaze_pose = BlazePoseLight(n_points, n_point_features)
        else:
            raise ValueError(F"Invalid model type: {type}.")

        if outputs not in ['TWOHEAD', 'HEATMAP', 'REGRESSION']:
            raise ValueError(F"Invalid model outputs type: '{outputs}'")
        return blaze_pose.build_model(outputs)
