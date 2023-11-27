from roboflow import Roboflow
rf = Roboflow(api_key="xsWeTDWkKNyQJkNrOZYE")
project = rf.workspace().project("solar-panels-with-faults")
model = project.version(2).model


def infer_local(img: str, visualize: bool = False):
    """Infer on the local image `img` (e.g. "your_image.jpg")."""
    print(model.predict(img, confidence=40, overlap=30).json())
    if visualize:
        model.predict(img, confidence=40, overlap=30).save("prediction.jpg")


# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())
