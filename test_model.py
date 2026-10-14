
from PIL import Image
from image_classifier import classify_image


def test_image_classification():
    image = Image.open("cat.jpg").convert("RGB")
    predictions = classify_image(image)

    assert isinstance(predictions, list)
    assert len(predictions) > 0
    assert "label" in predictions[0]
    assert "score" in predictions[0]

    print("All tests passed!")
    print("Top prediction:", predictions[0]["label"])
    print(f"Score: {predictions[0]['score'] * 100:.2f}%")


if __name__ == "__main__":
    test_image_classification()
