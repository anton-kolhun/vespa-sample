import os

from flask import Flask
from flask import request

from clip_feed import ClipVisualConverter
from flask import Response
app = Flask(__name__)
clip_converter = ClipVisualConverter()


@app.route('/transform_to_tensor', methods=['POST'])
def embed_image():
    image_url = request.get_json()['image_url']
    tensor = clip_converter.image_to_tensor(image_url)
    return Response(tensor, mimetype='application/json')


if __name__ == '__main__':
    app.run(port=os.getenv('PORT', default='8001'), debug=True)
