from flask import Flask, request

app = Flask(__name__)

scan_triggered = False

@app.route("/scan", methods=["POST"])
def scan():
    global scan_triggered
    scan_triggered = True
    return {"status": "scan received"}, 200

@app.route("/status", methods=["GET"])
def status():
    return {"scan_triggered": scan_triggered}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
