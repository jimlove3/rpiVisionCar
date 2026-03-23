#!/usr/bin/env python3
"""
Robot Car Web Server
====================
Raspberry Pi differential-drive robot with live camera stream.
Controls two DC motors via an L298N driver using GPIO PWM.
Streams MJPEG video from the Pi Camera via a Flask web server.

Wiring (L298N → Raspberry Pi GPIO BCM numbering):
  IN1 → GPIO 17    (Left motor direction A)
  IN2 → GPIO 27    (Left motor direction B)
  IN3 → GPIO 22    (Right motor direction A)
  IN4 → GPIO 23    (Right motor direction B)
  ENA → GPIO 18    (Left motor PWM speed)
  ENB → GPIO 24    (Right motor PWM speed)
  GND → Pi GND
  Motor power supply (7.4V LiPo or 6×AA) → L298N 12V + GND terminals

Run:
  sudo python3 robot_car_server.py

Access from browser:
  http://10.42.0.1  (when using the Pi hotspot)
  http://<pi-ip>:80  (on a shared network)
"""

import io
import time
import threading
import logging
from threading import Condition
from flask import Flask, Response, render_template_string, jsonify

# --- GPIO & Camera imports (graceful fallback for dev/testing) ---
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("WARNING: RPi.GPIO not found — running in simulation mode")

try:
    from picamera2 import Picamera2
    from picamera2.encoders import JpegEncoder
    from picamera2.outputs import FileOutput
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False
    print("WARNING: picamera2 not found — camera stream disabled")


# ─────────────────────────────────────────────
#  Motor pin configuration (BCM numbering)
# ─────────────────────────────────────────────
LEFT_IN1  = 17   # Left motor direction pin A
LEFT_IN2  = 27   # Left motor direction pin B
LEFT_ENA  = 18   # Left motor PWM (speed)

RIGHT_IN3 = 22   # Right motor direction pin A
RIGHT_IN4 = 23   # Right motor direction pin B
RIGHT_ENB = 24   # Right motor PWM (speed)

DEFAULT_SPEED = 75  # PWM duty cycle 0–100


# ─────────────────────────────────────────────
#  Motor controller
# ─────────────────────────────────────────────
class MotorController:
    def __init__(self):
        self.pwm_left = None
        self.pwm_right = None
        self.speed = DEFAULT_SPEED

        if not GPIO_AVAILABLE:
            return

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        pins = [LEFT_IN1, LEFT_IN2, LEFT_ENA, RIGHT_IN3, RIGHT_IN4, RIGHT_ENB]
        GPIO.setup(pins, GPIO.OUT)

        # PWM at 1 kHz
        self.pwm_left  = GPIO.PWM(LEFT_ENA,  1000)
        self.pwm_right = GPIO.PWM(RIGHT_ENB, 1000)
        self.pwm_left.start(0)
        self.pwm_right.start(0)

    def _set_left(self, forward: bool, speed: int):
        if not GPIO_AVAILABLE:
            return
        GPIO.output(LEFT_IN1, GPIO.HIGH if forward else GPIO.LOW)
        GPIO.output(LEFT_IN2, GPIO.LOW  if forward else GPIO.HIGH)
        self.pwm_left.ChangeDutyCycle(speed)

    def _set_right(self, forward: bool, speed: int):
        if not GPIO_AVAILABLE:
            return
        GPIO.output(RIGHT_IN3, GPIO.HIGH if forward else GPIO.LOW)
        GPIO.output(RIGHT_IN4, GPIO.LOW  if forward else GPIO.HIGH)
        self.pwm_right.ChangeDutyCycle(speed)

    def _stop_motors(self):
        if not GPIO_AVAILABLE:
            return
        GPIO.output([LEFT_IN1, LEFT_IN2, RIGHT_IN3, RIGHT_IN4], GPIO.LOW)
        self.pwm_left.ChangeDutyCycle(0)
        self.pwm_right.ChangeDutyCycle(0)

    # ── Public commands ───────────────────────
    def forward(self):
        self._set_left(True,  self.speed)
        self._set_right(True, self.speed)
        print("MOVE: forward")

    def backward(self):
        self._set_left(False,  self.speed)
        self._set_right(False, self.speed)
        print("MOVE: backward")

    def turn_left(self):
        self._set_left(False, self.speed)
        self._set_right(True, self.speed)
        print("MOVE: turn left")

    def turn_right(self):
        self._set_left(True,  self.speed)
        self._set_right(False, self.speed)
        print("MOVE: turn right")

    def stop(self):
        self._stop_motors()
        print("MOVE: stop")

    def set_speed(self, speed: int):
        self.speed = max(0, min(100, speed))
        print(f"SPEED: {self.speed}%")

    def cleanup(self):
        if not GPIO_AVAILABLE:
            return
        self._stop_motors()
        if self.pwm_left:
            self.pwm_left.stop()
        if self.pwm_right:
            self.pwm_right.stop()
        GPIO.cleanup()


# ─────────────────────────────────────────────
#  Camera stream (MJPEG)
# ─────────────────────────────────────────────
class StreamOutput(io.BufferedIOBase):
    """Thread-safe frame buffer using Condition for reliable first-frame sync.

    Uses JpegEncoder (not MJPEGEncoder) — this is the pattern from the official
    picamera2 examples and avoids a race condition where the generator reads
    output.frame before the first frame has been written.
    """
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()


class CameraStream:
    def __init__(self):
        self.camera = None
        self.output  = None

        if not CAMERA_AVAILABLE:
            return

        self.camera = Picamera2()
        # YUV420 is the native sensor format — JpegEncoder converts it
        # efficiently on-chip. RGB888 would require a software conversion step.
        config = self.camera.create_video_configuration(
            main={"size": (640, 480)},
            controls={"FrameRate": 30}
        )
        self.camera.configure(config)
        self.output = StreamOutput()
        self.camera.start_recording(JpegEncoder(), FileOutput(self.output))

    def get_frame(self):
        """Block until a new frame is available, then return JPEG bytes."""
        if not CAMERA_AVAILABLE or self.output is None:
            return None
        with self.output.condition:
            self.output.condition.wait()
            return self.output.frame

    def stop(self):
        if self.camera:
            self.camera.stop_recording()
            self.camera.close()


# ─────────────────────────────────────────────
#  HTML page (served at /)
# ─────────────────────────────────────────────
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MHS Vision Car Controller</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: Arial, sans-serif;
      background: #1a1a2e;
      color: #eee;
      display: flex;
      flex-direction: column;
      align-items: center;
      min-height: 100vh;
      padding: 20px;
      gap: 20px;
    }
    h1 { font-size: 1.4rem; color: #a78bfa; letter-spacing: 1px; }

    /* Video feed */
    #video-container {
      width: 100%;
      max-width: 640px;
      background: #000;
      border-radius: 10px;
      overflow: hidden;
      border: 2px solid #374151;
    }
    #video-container img {
      width: 100%;
      display: block;
    }
    #no-camera {
      text-align: center;
      padding: 60px 20px;
      color: #6b7280;
    }

    /* Status bar */
    #status {
      font-size: 0.9rem;
      color: #9ca3af;
      min-height: 1.2em;
    }

    /* Speed slider */
    .speed-row {
      display: flex;
      align-items: center;
      gap: 12px;
      font-size: 0.9rem;
    }
    input[type=range] { width: 180px; accent-color: #a78bfa; }

    /* D-pad controls */
    .dpad {
      display: grid;
      grid-template-areas:
        ". up ."
        "left stop right"
        ". down .";
      gap: 8px;
    }
    .btn {
      width: 72px; height: 72px;
      border: none;
      border-radius: 10px;
      font-size: 1.6rem;
      cursor: pointer;
      background: #374151;
      color: #e5e7eb;
      transition: background 0.1s, transform 0.1s;
      user-select: none;
    }
    .btn:active, .btn.active {
      background: #a78bfa;
      transform: scale(0.95);
    }
    .btn-up    { grid-area: up; }
    .btn-left  { grid-area: left; }
    .btn-stop  { grid-area: stop; background: #4b1c1c; }
    .btn-stop:active { background: #dc2626; }
    .btn-right { grid-area: right; }
    .btn-down  { grid-area: down; }

    /* Keyboard hint */
    .hint {
      font-size: 0.75rem;
      color: #4b5563;
      text-align: center;
    }
  </style>
</head>
<body>
  <h1>MHS Vision Car Controller</h1>

  <div id="video-container">
    {% if camera_available %}
      <img src="/video_feed" alt="Live camera feed">
    {% else %}
      <div id="no-camera">Camera not available</div>
    {% endif %}
  </div>

  <div id="status">Press a button or use arrow keys to drive</div>

  <div class="speed-row">
    <span>Speed:</span>
    <input type="range" id="speed" min="20" max="100" value="75"
           oninput="setSpeed(this.value)">
    <span id="speed-label">75%</span>
  </div>

  <div class="dpad">
    <button class="btn btn-up"    id="btn-forward"  ontouchstart="sendCmd('forward')"  onmousedown="sendCmd('forward')"  ontouchend="sendCmd('stop')" onmouseup="sendCmd('stop')">▲</button>
    <button class="btn btn-left"  id="btn-left"     ontouchstart="sendCmd('left')"     onmousedown="sendCmd('left')"     ontouchend="sendCmd('stop')" onmouseup="sendCmd('stop')">◀</button>
    <button class="btn btn-stop"  id="btn-stop"     onclick="sendCmd('stop')">■</button>
    <button class="btn btn-right" id="btn-right"    ontouchstart="sendCmd('right')"    onmousedown="sendCmd('right')"    ontouchend="sendCmd('stop')" onmouseup="sendCmd('stop')">▶</button>
    <button class="btn btn-down"  id="btn-backward" ontouchstart="sendCmd('backward')" onmousedown="sendCmd('backward')" ontouchend="sendCmd('stop')" onmouseup="sendCmd('stop')">▼</button>
  </div>

  <p class="hint">Arrow keys also work · Release key or button to stop</p>

  <script>
    const status = document.getElementById('status');
    const btnMap = {
      forward: 'btn-forward', backward: 'btn-backward',
      left: 'btn-left', right: 'btn-right'
    };
    let activeCmd = null;

    async function sendCmd(cmd) {
      if (cmd === activeCmd) return;
      activeCmd = cmd;

      // Update button highlights
      document.querySelectorAll('.btn').forEach(b => b.classList.remove('active'));
      if (btnMap[cmd]) document.getElementById(btnMap[cmd])?.classList.add('active');

      status.textContent = cmd === 'stop' ? 'Stopped' : `Moving: ${cmd}`;

      try {
        const resp = await fetch('/command/' + cmd, { method: 'POST' });
        if (!resp.ok) status.textContent = 'Command error';
      } catch (e) {
        status.textContent = 'Connection lost';
      }

      if (cmd === 'stop') {
        activeCmd = null;
        document.querySelectorAll('.btn').forEach(b => b.classList.remove('active'));
      }
    }

    async function setSpeed(val) {
      document.getElementById('speed-label').textContent = val + '%';
      await fetch('/speed/' + val, { method: 'POST' });
    }

    // Keyboard control
    const keyMap = {
      ArrowUp: 'forward', ArrowDown: 'backward',
      ArrowLeft: 'left',  ArrowRight: 'right', ' ': 'stop'
    };
    document.addEventListener('keydown', e => {
      if (keyMap[e.key]) { e.preventDefault(); sendCmd(keyMap[e.key]); }
    });
    document.addEventListener('keyup', e => {
      if (keyMap[e.key] && e.key !== ' ') sendCmd('stop');
    });
  </script>
</body>
</html>
"""


# ─────────────────────────────────────────────
#  Flask app
# ─────────────────────────────────────────────
app = Flask(__name__)
motors = MotorController()
camera = CameraStream()

log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)  # Suppress per-request logs


def mjpeg_generator():
    """Yield MJPEG frames as a multipart HTTP stream."""
    while True:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        )


@app.route('/')
def index():
    return render_template_string(HTML_PAGE, camera_available=CAMERA_AVAILABLE)


@app.route('/video_feed')
def video_feed():
    if not CAMERA_AVAILABLE:
        return "Camera not available", 503
    return Response(
        mjpeg_generator(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/command/<cmd>', methods=['POST'])
def command(cmd):
    actions = {
        'forward':  motors.forward,
        'backward': motors.backward,
        'left':     motors.turn_left,
        'right':    motors.turn_right,
        'stop':     motors.stop,
    }
    if cmd in actions:
        actions[cmd]()
        return jsonify(status='ok', command=cmd)
    return jsonify(status='error', message='Unknown command'), 400


@app.route('/speed/<int:speed>', methods=['POST'])
def set_speed(speed):
    motors.set_speed(speed)
    return jsonify(status='ok', speed=speed)


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 50)
    print("  Robot Car Server starting...")
    print(f"  GPIO:   {'enabled' if GPIO_AVAILABLE   else 'SIMULATION MODE'}")
    print(f"  Camera: {'enabled' if CAMERA_AVAILABLE else 'disabled'}")
    print("  Open your browser to: http://10.42.0.1")
    print("=" * 50)
    try:
        app.run(host='0.0.0.0', port=80, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        motors.cleanup()
        camera.stop()
