from flask import Flask, render_template, Response, jsonify
from camera import VideoCamera
import os
template_dir = os.path.join('./')
app = Flask(__name__, template_folder=template_dir)

resp = {'Details': '', 'Status': ''}

status = False


def gen(camera):
    global status
    model = camera.train()
    while True:
        frame, verify = camera.get_frame(model)
        if verify == 23:
            status = True
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


x = gen(VideoCamera())


@app.route('/')
def index():
    return render_template('login.ejs')


@app.route('/check_success')
def success():
    if status == True:
        resp['Status'] = 'Success'
        resp['Details'] = 'Face Verified'
        return jsonify(resp)
    resp['Status'] = 'Fail'
    resp['Details'] = 'Face Unverified'
    return jsonify(resp)


@app.route('/video_feed')
def video_feed():
    return Response(x, mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
