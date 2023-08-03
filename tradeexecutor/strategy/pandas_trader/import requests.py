import requests
from PIL import Image
from io import BytesIO

# Make a GET request to the image's URL
response = requests.get("https://enzyme-polygon-multipair.tradingstrategy.ai/visualisation?theme=light&type=large")

# Check that the request was successful
if response.status_code == 200:
    # Open the image
    image = Image.open(BytesIO(response.content))
    # Save the image
    image.save("image_large.png")
else:
    print("Failed to download the image, status code:", response.status_code)
