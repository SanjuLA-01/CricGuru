from flask import Blueprint, request, jsonify
from src.services.processor import *

routes_bp = Blueprint('routes', __name__)


@routes_bp.route('/')
def home():
    data = {'message': 'Welcom to CricGuru API!'}
    response = jsonify(data)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


# Define the endpoint for file uploads
@routes_bp.route('/upload', methods=['POST'])
def upload_file():
    # Get the uploaded file from the request
    file = request.files['file']
    saveFileLocally(file)

    data = {'message': 'File uploaded successfully!', 'name': file.filename}
    response = jsonify(data)
    response.headers.add('Access-Control-Allow-Origin', '*')
    # Return a response to the client
    return response


@routes_bp.route('/process', methods=['POST'])
def process_image():
    hand = request.form.get('hand')
    area = request.form.get('area')

    stanceProper, legProper, shotProper = processVideo(hand, area)
    return jsonify({
        'stance': stanceProper,
        'leg': legProper,
        'shot': shotProper
    })


@routes_bp.route('/images/<path:filename>')
def output_file(filename):
    return loadImage(filename)
