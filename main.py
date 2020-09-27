from flask import Flask, render_template, Response
from camera import VideoCamera

app = Flask( __name__ )
video_stream = VideoCamera()


@app.route( '/' )
def index():
    return render_template( 'index.html' )


def gen(camera):
    """

    :type camera: object
    """
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        return 'ok'


@app.route( '/video_feed' )
def video_feed():
    return Response( gen( video_stream ),
                     mimetype='multipart/x-mixed-replace; boundary=frame' )


if __name__ == '__main__':
    # app.run(host='0.0.0.0', debug=True, port=500)
    # app.run(host='0.0.0.0', debug=False, threaded=True)
    app.run(debug=True)
