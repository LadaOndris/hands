from src.estimation.blazepose.models.model import BlazePose as BlazePoseFull


class ModelCreator():

    @staticmethod
    def create_model(model_name, n_points=0):
        if model_name == "SIGMOID_HEATMAP_LINEAR_REGRESS_TWO_HEAD":
            return BlazePoseFull(n_points).build_model("TWO_HEAD")
        elif model_name == "SIGMOID_HEATMAP_LINEAR_REGRESS_HEATMAP":
            return BlazePoseFull(n_points).build_model("HEATMAP")
        elif model_name == "SIGMOID_HEATMAP_LINEAR_REGRESS_REGRESSION":
            return BlazePoseFull(n_points).build_model("REGRESSION")
        raise ValueError(F"Invalid model name: '{model_name}'")

