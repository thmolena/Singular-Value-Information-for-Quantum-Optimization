Website demo

To use the demo locally:
1. Start the Flask server (from the repo root):
   - $env:FLASK_APP = 'src.server'
   - flask run --host=0.0.0.0 --port=5000
2. Serve the website directory:
   - python -m http.server 8000 --directory website
3. Open http://localhost:8000/index.html

Note: `demo.js` sends predictions to `http://localhost:5000/predict` by default. If needed, set `window.API_BASE_URL` before loading `demo.js` (for example, `https://your-host:5000`) to target a different API endpoint.
