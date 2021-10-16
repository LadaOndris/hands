from src.estimation.blazepose.models.model import BlazePose as BlazePoseFull


class ModelCreator():

    @staticmethod
    def create_model(model_name, n_points, n_point_features):
        blaze_pose = BlazePoseFull(n_points, n_point_features)
        if model_name == "SIGMOID_HEATMAP_LINEAR_REGRESS_TWO_HEAD":
            return blaze_pose.build_model("TWO_HEAD")
        elif model_name == "SIGMOID_HEATMAP_LINEAR_REGRESS_HEATMAP":
            return blaze_pose.build_model("HEATMAP")
        elif model_name == "SIGMOID_HEATMAP_LINEAR_REGRESS_REGRESSION":
            return blaze_pose.build_model("REGRESSION")
        raise ValueError(F"Invalid model name: '{model_name}'")

