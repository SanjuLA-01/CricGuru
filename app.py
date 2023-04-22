from flask import Flask
from src.routes import routes_bp
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app)
app.register_blueprint(routes_bp)

if __name__ == "__main__":
    app.run(debug=True)
