import pytesseract
from PIL import Image

def extract_text_from_image(image_path):
    """ Uses Tesseract OCR to extract proof text from an image """
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

# Test it
if __name__ == "__main__":
    image_path = "handwritten_proof.png"
    print("Extracted Text:", extract_text_from_image(image_path))
