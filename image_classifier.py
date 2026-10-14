
from transformers import pipeline
from PIL import Image

# Load the pretrained image classification model
classifier = pipeline(
    task="image-classification",
    model="google/vit-base-patch16-224"
)


def classify_image(image):
    """
    Predict the objects or categories present in an image.
    """
    results = classifier(image)

    return results


if __name__ == "__main__":
    image_path = input("Enter the image file path: ")

    image = Image.open(image_path).convert("RGB")

    predictions = classify_image(image)

    print("\nImage Recognition Results:")

    for prediction in predictions[:5]:
        label = prediction["label"]
        score = prediction["score"] * 100

        print(f"{label}: {score:.2f}%")
