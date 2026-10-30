#!/usr/bin/env pybricks-micropython
"""Task 5 node-marked contour survey, Pybricks EV3 MicroPython 2.

LEFT motor B; RIGHT C; colour S3; gyro S1. Uses the previously tested
follower without changing its gains/speed. Based on the classmate-shared
contour-following approach, with explicit human node annotations.

Before starting, LEFT selects OUTER / tape on LEFT of sensor;
RIGHT selects INNER / tape on RIGHT. Both start at S, facing towards M.
Place the robot on the appropriate tape edge BEFORE CENTER and gyro check.
No automatic tape crossing in this version. Runs are independent local frames.

At S press CENTER; keep still for the gyro check. S is marked automatically
at the actual starting pose. Match that physical reference on returning to S.
UP at each displayed next node records that node once without stopping.
Hold/release detection prevents one held press from advancing multiple nodes.
The final UP at the last S stops and closes the CSV automatically.
CENTER pauses; CENTER resumes. DOWN stops/saves an incomplete run.
The sequence supplies expected names ONLY: it does not navigate or recognise
nodes. Press UP only after actually observing arrival at the displayed node.
Do not lift or reposition the robot during a run.

Outer sequence adds intermediate S between C and D4 because the lecture
board connects C--S--D4; there is no direct C--D4 edge. Confirm on actual board.
Outer: S M G2 M N K D2 K G1 K N C S D4 S
Inner: S M N C D1 C S

Files task5_nodes_001.csv, 002, ... never overwrite existing names.
node_start/node_manual rows identify visits with route_name and visit_index.
Coordinates/centre travel remain wheel/gyro estimates, not ground truth.
Node lengths include contour manoeuvres and human marking uncertainty;
retain separate directed traversals before combining edge estimates.
"""
import math
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import Motor, ColorSensor, GyroSensor
from pybricks.parameters import Port, Button, Direction
from pybricks.tools import StopWatch, wait

LEFT_PORT = Port.B
RIGHT_PORT = Port.C
WHEEL_RADIUS_MM = 28.0
TARGET = (10.64 + 54.71) / 2.0
BASE_DPS = 120.0
KP = 3.0
ERROR_SLOW = 14.0
MAX_REVERSE = 0.4
ERROR_FILTER_TAU_S = 0.095
CONTROL_MS = 10
LOG_MS = 50
DRIFT_MS = 10000
CROSS_ANGLE_DEG = 50.0
CROSS_SIDE_MM = 35.0  # Must exceed actual tape width; check before crossing.
TURN_TIMEOUT_MS = 8000
CROSS_TIMEOUT_MS = 5000
MAX_WHEEL_DPS = 300.0
OUTER = ('S', 'M', 'G2', 'M', 'N', 'K', 'D2', 'K', 'G1', 'K', 'N', 'C', 'S', 'D4', 'S')
INNER = ('S', 'M', 'N', 'C', 'D1', 'C', 'S')


def clip(x, lo, hi):
    return max(lo, min(hi, x))


def integrate(x, y, old_heading, heading, ds):
    # Integrate the short centre arc using its midpoint heading.
    delta = math.radians(heading - old_heading)
    half = delta / 2.0
    scale = math.sin(half) / half if abs(half) > 1e-8 else 1.0
    middle = math.radians((heading + old_heading) / 2.0)
    return (x + ds * scale * math.cos(middle),
            y + ds * scale * math.sin(middle))


def unique_file():
    index = 1
    while True:
        name = 'task5_nodes_%03d.csv' % index
        try:
            existing = open(name, 'r')
        except OSError:
            return name
        existing.close()
        index += 1


class Survey:
    def __init__(self):
        self.ev3 = EV3Brick()
        self.left = Motor(LEFT_PORT)
        self.right = Motor(RIGHT_PORT)
        self.colour = ColorSensor(Port.S3)
        self.gyro = GyroSensor(Port.S1, Direction.COUNTERCLOCKWISE)
        self.clock = StopWatch()
        self.file = None
        self.lap = 1
        self.tape_left = True
        self.drift = 0.0
        self.x = self.y = self.net = self.travel = 0.0
        self.heading = 0.0
        self.prev_l = self.prev_r = 0
        self.last_log = -LOG_MS
        self.last_flush = 0
        self.last_control = 0
        self.filtered_error = 0.0
        self.push_until = 0
        self.stuck_since = 0
        self.stuck_net = self.stuck_heading = 0.0
        self.u_left = self.u_right = 0.0
        self.reflection = 0
        self.old_node = 'S'
        self.name = ''
        self.route_name = 'outer'
        self.route = OUTER
        self.visit_index = 0
        self.up_was_down = False
        self.last_mark_ms = -1000

    def show(self, *lines):
        self.ev3.screen.clear()
        for line in lines:
            self.ev3.screen.print(line)

    def release(self):
        while self.ev3.buttons.pressed():
            wait(10)

    def hold(self):
        self.left.hold()
        self.right.hold()
        self.u_left = self.u_right = 0.0

    def drive(self, left, right):
        self.u_left = clip(left, -MAX_WHEEL_DPS, MAX_WHEEL_DPS)
        self.u_right = clip(right, -MAX_WHEEL_DPS, MAX_WHEEL_DPS)
        self.left.run(self.u_left)
        self.right.run(self.u_right)

    def sample(self, mode, event='', node=''):
        t = self.clock.time()
        l, r = self.left.angle(), self.right.angle()
        h = self.gyro.angle() - self.drift * t / 1000.0
        dl = math.radians(l - self.prev_l) * WHEEL_RADIUS_MM
        dr = math.radians(r - self.prev_r) * WHEEL_RADIUS_MM
        ds = (dl + dr) / 2.0
        self.x, self.y = integrate(self.x, self.y, self.heading, h, ds)
        self.net += ds
        self.travel += abs(ds)  # Centre path, not wheel turning distance.
        self.prev_l, self.prev_r, self.heading = l, r, h
        self.reflection = self.colour.reflection()
        if self.file and (event or t - self.last_log >= LOG_MS):
            row = (t, self.lap, mode, event, node, int(self.tape_left),
                   self.reflection, TARGET, h, self.x, self.y, self.net,
                   self.travel, l, r, self.u_left, self.u_right, self.drift,
                   self.route_name, self.visit_index)
            self.file.write(','.join(str(v) for v in row) + '\n')
            self.last_log = t
        if self.file and t - self.last_flush >= 1000:
            self.file.flush()
            self.last_flush = t
        return t

    def reset_follow(self):
        self.filtered_error = 0.0
        self.push_until = 0
        self.stuck_since = self.clock.time()
        self.stuck_net = self.net
        self.stuck_heading = self.heading
        self.last_control = self.clock.time()

    def follow(self):
        now = self.sample('follow')
        dt = max(0.001, (now - self.last_control) / 1000.0)
        self.last_control = now
        error = self.reflection - TARGET
        alpha = dt / (ERROR_FILTER_TAU_S + dt)
        self.filtered_error += alpha * (error - self.filtered_error)
        turn = KP * error * (1 if self.tape_left else -1)
        speed = BASE_DPS * max(-MAX_REVERSE,
                               1 - abs(self.filtered_error) / ERROR_SLOW)
        if now - self.stuck_since >= 2000:
            if (abs(self.net - self.stuck_net) < 10 and
                    abs(self.heading - self.stuck_heading) < 20):
                self.push_until = now + 400
                self.sample('follow', 'stuck_push')
            self.stuck_since = now
            self.stuck_net, self.stuck_heading = self.net, self.heading
        if now < self.push_until:
            speed = 0.4 * BASE_DPS
            turn *= 0.5
        self.drive(speed - turn, speed + turn)
        wait(CONTROL_MS)

    def status(self):
        next_node = (self.route[self.visit_index + 1]
                     if self.visit_index + 1 < len(self.route) else 'DONE')
        self.show(self.route_name.upper(), 'Last: ' + self.route[self.visit_index],
                  'NEXT: ' + next_node,
                  'Marks: %d/%d' % (self.visit_index, len(self.route) - 1),
                  'UP mark / DOWN end')

    def mark_next(self):
        now = self.clock.time()
        if now - self.last_mark_ms < 200:
            return False
        if self.visit_index + 1 >= len(self.route):
            return True
        self.visit_index += 1
        self.last_mark_ms = now
        self.sample('follow', 'node_manual', self.route[self.visit_index])
        self.file.flush()
        self.status()
        return self.visit_index == len(self.route) - 1

    def handle_up(self, buttons):
        pressed = Button.UP in buttons
        rising = pressed and not self.up_was_down
        self.up_was_down = pressed
        return self.mark_next() if rising else False

    def select_route(self):
        while True:
            self.show('TASK 5 NODE LOG', 'LEFT outer / RIGHT inner',
                      'Selected: ' + self.route_name.upper(),
                      'Tape on ' + ('LEFT' if self.tape_left else 'RIGHT'),
                      'At S towards M', 'CENTER start / DOWN end')
            while not self.ev3.buttons.pressed():
                wait(20)
            buttons = self.ev3.buttons.pressed()
            self.release()
            if Button.DOWN in buttons:
                return False
            if Button.LEFT in buttons:
                self.route_name, self.route = 'outer', OUTER
                self.tape_left, self.lap = True, 1
            elif Button.RIGHT in buttons:
                self.route_name, self.route = 'inner', INNER
                self.tape_left, self.lap = False, 2
            elif Button.CENTER in buttons:
                return True

    def pause(self):
        self.hold()
        self.sample('pause', 'pause_start')
        self.release()
        self.show('PAUSED', 'Next: ' + self.route[self.visit_index + 1],
                  'CENTER resume', 'DOWN finish', 'UP ignored while paused')
        while True:
            buttons = self.ev3.buttons.pressed()
            if Button.DOWN in buttons:
                return False
            if Button.CENTER in buttons:
                self.release()
                self.reset_follow()
                self.up_was_down = Button.UP in self.ev3.buttons.pressed()
                self.sample('follow', 'resume')
                self.status()
                return True
            self.sample('pause')
            wait(20)

    def run(self):
        if not self.select_route():
            return
        self.hold()
        self.show('Gyro drift check', 'KEEP STILL 10 s')
        wait(1000)
        initial = self.gyro.angle()
        wait(DRIFT_MS)
        self.drift = (self.gyro.angle() - initial) / (DRIFT_MS / 1000.0)
        if abs(self.drift) > 1:
            self.show('Gyro drift too high', 'Reconnect while still', 'Restart program')
            print('Gyro drift rejected:', self.drift)
            return
        self.gyro.reset_angle(0)
        self.left.reset_angle(0)
        self.right.reset_angle(0)
        self.clock.reset()
        self.name = unique_file()
        reason = 'user_finish_incomplete'
        try:
            self.file = open(self.name, 'w')
            self.file.write('time_ms,lap,mode,event,node,tape_left,reflection,target,'
                            'heading_deg,x_mm,y_mm,net_mm,travel_mm,left_deg,right_deg,'
                            'left_cmd_dps,right_cmd_dps,gyro_drift_dps,route_name,visit_index\n')
            self.sample('pause', 'node_start', 'S')
            self.file.flush()
            self.reset_follow()
            self.status()
            while True:
                buttons = self.ev3.buttons.pressed()
                if Button.DOWN in buttons:
                    break
                if Button.CENTER in buttons:
                    if not self.pause():
                        break
                elif self.handle_up(buttons):
                    reason = 'route_complete_manual'
                    break
                self.follow()
        except KeyboardInterrupt:
            reason = 'keyboard_interrupt'
        except Exception as error:
            reason = 'error'
            print('ERROR:', repr(error))
        finally:
            self.hold()
            if self.file:
                try:
                    self.sample('finish', reason)
                    self.file.flush()
                finally:
                    self.file.close()
            self.left.stop()
            self.right.stop()
            self.show('STOPPED', reason, self.name,
                      'Visits %d/%d' % (self.visit_index, len(self.route) - 1),
                      'Copy CSV to laptop')
            print('Saved:', self.name, 'End:', reason)


if __name__ == '__main__':
    Survey().run()
