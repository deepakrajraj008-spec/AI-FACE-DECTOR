"""Face recognition attendance - web demo.
Run:  python app.py   then open http://127.0.0.1:5000
Needs: pip install flask opencv-contrib-python==4.10.0.84 numpy==1.26.4
"""
import cv2, os, json, time, platform, threading, shutil
import numpy as np
from flask import Flask, Response, jsonify, request, render_template

DATA, MODEL, NAMES_FILE = "dataset", "model.yml", "names.json"
THRESHOLD = 70      # lower distance = better match; tune after testing
SHOTS = 40          # photos captured per person

os.makedirs(DATA, exist_ok=True)
app = Flask(__name__)
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW) if platform.system() == "Windows" else cv2.VideoCapture(0)
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
lock = threading.Lock()
state = {"enroll": None, "count": 0, "trained": False, "msg": "", "faces": [], "present": {}}


def read_names():
    names = {int(k): v for k, v in json.load(open(NAMES_FILE)).items()} if os.path.exists(NAMES_FILE) else {}
    for d in os.listdir(DATA):
        if d.isdigit() and int(d) not in names:
            names[int(d)] = f"Person {d}"
    return names


def save_names(names):
    json.dump({str(k): v for k, v in names.items()}, open(NAMES_FILE, "w"))


if os.path.exists(MODEL) and read_names():
    recognizer.read(MODEL)
    state["trained"] = True


def frames():
    n = 0
    while True:
        ok, img = cam.read()
        if not ok:
            time.sleep(0.1)
            continue
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        boxes = detector.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))
        found, names = [], read_names()
        for (x, y, w, h) in boxes:
            roi = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
            label, color, conf = "Unknown", (60, 60, 220), None
            e = state["enroll"]
            if e:
                label, color = "Enrolling " + e["name"], (0, 165, 255)
                n += 1
                if n % 3 == 0:
                    folder = os.path.join(DATA, str(e["id"]))
                    os.makedirs(folder, exist_ok=True)
                    cv2.imwrite(os.path.join(folder, f"{int(time.time() * 1000)}.jpg"), roi)
                    state["count"] += 1
                    if state["count"] >= SHOTS:
                        state["enroll"], state["msg"] = None, f"{e['name']} enrolled. Click Train model."
                break  # only the first face while enrolling
            elif state["trained"]:
                with lock:
                    pid, conf = recognizer.predict(roi)
                if conf < THRESHOLD and pid in names:
                    label, color = names[pid], (80, 170, 0)
                    state["present"].setdefault(label, time.strftime("%H:%M:%S"))
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            found.append({"name": label, "conf": None if conf is None else round(float(conf), 1)})
        state["faces"] = found
        ok, buf = cv2.imencode(".jpg", img)
        yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"


@app.route("/video")
def video():
    return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=f")


@app.route("/status")
def status():
    return jsonify({k: state[k] for k in ("count", "trained", "msg", "faces", "present")} | {"enrolling": bool(state["enroll"]), "shots": SHOTS})


def train_model():
    imgs, labels = [], []
    for pid in os.listdir(DATA):
        folder = os.path.join(DATA, pid)
        if not (pid.isdigit() and os.path.isdir(folder)):
            continue
        for f in os.listdir(folder):
            im = cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
            if im is not None:
                imgs.append(im)
                labels.append(int(pid))
    if not imgs:
        return 0
    with lock:
        recognizer.train(imgs, np.array(labels))
        recognizer.write(MODEL)
    state["trained"] = True
    return len(set(labels))


@app.route("/enroll", methods=["POST"])
def enroll():
    names = read_names()
    pid = request.json.get("id")
    if pid:  # add more photos for someone already enrolled
        pid = int(pid)
        if pid not in names:
            return jsonify(ok=False)
        name = names[pid]
    else:
        name = (request.json.get("name") or "").strip()
        if not name:
            return jsonify(ok=False)
        pid = max(names, default=0) + 1
        names[pid] = name
        save_names(names)
    state.update(enroll={"id": pid, "name": name}, count=0, msg="Look at the camera and move your head slightly.")
    return jsonify(ok=True)


@app.route("/train", methods=["POST"])
def train():
    n = train_model()
    state["msg"] = f"Training complete ({n} people)." if n else "No photos yet. Enroll someone first."
    return jsonify(ok=bool(n))


@app.route("/people")
def people():
    out = []
    for pid, name in sorted(read_names().items()):
        folder = os.path.join(DATA, str(pid))
        out.append({"id": pid, "name": name, "photos": len(os.listdir(folder)) if os.path.isdir(folder) else 0})
    return jsonify(out)


@app.route("/rename", methods=["POST"])
def rename():
    pid, name = int(request.json["id"]), (request.json.get("name") or "").strip()
    names = read_names()
    if pid in names and name:
        names[pid] = name
        save_names(names)
    return jsonify(ok=True)


@app.route("/delete", methods=["POST"])
def delete():
    pid = int(request.json["id"])
    names = read_names()
    gone = names.pop(pid, "Person")
    save_names(names)
    shutil.rmtree(os.path.join(DATA, str(pid)), ignore_errors=True)
    state["present"].pop(gone, None)
    if train_model():
        state["msg"] = f"{gone} deleted. Model retrained."
    else:
        state["trained"] = False
        if os.path.exists(MODEL):
            os.remove(MODEL)
        state["msg"] = f"{gone} deleted. No one is enrolled now."
    return jsonify(ok=True)


@app.route("/clear", methods=["POST"])
def clear():
    state["present"] = {}
    return jsonify(ok=True)





@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, threaded=True)