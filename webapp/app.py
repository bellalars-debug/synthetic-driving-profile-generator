"""Flask web app: enter your company headcount -> predicted EV charging demand.

Runs the synthetic-driver -> ev-tool charging pipeline behind a single input.
    python app.py    then open http://127.0.0.1:5001
"""
import os
import traceback

from flask import Flask, request, jsonify, send_file

import pipeline

app = Flask(__name__)
_HERE = os.path.dirname(os.path.abspath(__file__))


@app.route("/")
def index():
    return send_file(os.path.join(_HERE, "index.html"))


@app.route("/api/estimate", methods=["POST"])
def api_estimate():
    data = request.get_json(force=True, silent=True) or {}
    try:
        employees = int(data.get("employees", 100))
        adoption = float(data.get("adoption_rate", 0.36))
        if employees < 1:
            return jsonify({"ok": False, "error": "Enter at least 1 employee."}), 400
        result = pipeline.estimate(employees, adoption)
        return jsonify({"ok": True, "result": result})
    except Exception as exc:                                  # pragma: no cover
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    host = os.environ.get("HOST", "127.0.0.1")   # set HOST=0.0.0.0 to share on the LAN
    print(f"\n  EV Charging Demand Estimator running -> http://{host}:{port}\n")
    app.run(host=host, port=port, debug=False)
