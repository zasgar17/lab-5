#!/usr/bin/env pybricks-micropython
"""Task 5 contour survey with manual node labels and persistent CSV.

Adapted from the contour-following approach shared by a classmate.
This is NOT an autonomous graph explorer: check actual node/edge coverage.
No lecture example edge lengths are used. PLAN/RUN are subsequent stages.

Motor wiring confirmed by the user:
  LEFT B, RIGHT C, colour S3, gyro S1 (counterclockwise positive).
Change the two motor ports below if your physical wiring differs.

At S face towards M. Put the SENSOR over the south edge of S--M,
just beyond the S pad: tape on sensor's LEFT. Mark sensor start beside tape.
CENTER starts; keep still for gyro drift measurement.

While driving:
  CENTER: hold/pause. CENTER again resumes; LEFT opens node-label menu;
          RIGHT starts automatic crossing to the other tape edge;
          DOWN finishes and saves. Read screen before pressing.
  UP: hold and open node-label menu directly (optional).
  DOWN: finish and save immediately.
At node menu LEFT/RIGHT choose observed label, CENTER records it;
DOWN cancels. Pause at the SAME physical reference each time, preferably
sensor over junction centre. Do not move/lift the robot to make a mark.
After marking, CENTER resumes from the paused menu.

At the first return to the start mark: CENTER then RIGHT crosses the tape
and switches to lap 2 (tape on RIGHT). Crossing requires clear space.
At the second return: DOWN saves. It never detects lap completion itself.
Any missing branches must be surveyed separately; a saved CSV is not proof
of full coverage. Reverses and all crossing movements are explicitly logged.

Output: task5_explore_001.csv, 002, ... (never overwrites an existing file).
Pose is relative to initial ROBOT centre, x forward, y left, mm throughout.
Node labels are human observations, not automatic recognition. Real exits
must be checked on the board; adjacent labels can establish measured edges.
Uses Pybricks EV3 MicroPython 2, not ev3dev2 Python.
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
NODES = ('S', 'C', 'N', 'K', 'M', 'D1', 'D2', 'D4', 'G1', 'G2')


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
        name = 'task5_explore_%03d.csv' % index
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
                   self.travel, l, r, self.u_left, self.u_right, self.drift)
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

    def check_stop(self):
        if Button.DOWN in self.ev3.buttons.pressed():
            raise StopIteration('user_stop')

    def turn(self, degrees):
        self.sample('cross')
        target = self.heading + degrees
        start = self.clock.time()
        while True:
            self.check_stop()
            self.sample('cross')
            error = target - self.heading
            if abs(error) <= 2:
                break
            if self.clock.time() - start > TURN_TIMEOUT_MS:
                raise RuntimeError('turn_timeout: check gyro / motor ports')
            w = clip(4 * abs(error), 60, 200) * (1 if error > 0 else -1)
            self.drive(-w, w)
            wait(CONTROL_MS)
        self.hold()
        self.sample('cross', 'turn_end')

    def cross(self):
        self.sample('cross', 'cross_start')
        angle = CROSS_ANGLE_DEG * (1 if self.tape_left else -1)
        self.turn(angle)
        start_arc, start_ms = self.net, self.clock.time()
        distance = CROSS_SIDE_MM / math.sin(math.radians(CROSS_ANGLE_DEG))
        while self.net - start_arc < distance:
            self.check_stop()
            if self.clock.time() - start_ms > CROSS_TIMEOUT_MS:
                raise RuntimeError('cross_timeout')
            self.drive(150, 150)
            self.sample('cross')
            wait(CONTROL_MS)
        self.hold()
        self.turn(-angle)
        self.lap += 1
        self.tape_left = not self.tape_left
        self.reset_follow()
        self.sample('pause', 'cross_end')

    def mark_node(self):
        index = NODES.index(self.old_node)
        self.release()
        while True:
            self.show('Observed node', NODES[index], 'LEFT/RIGHT select',
                      'CENTER record', 'DOWN cancel')
            while not self.ev3.buttons.pressed():
                self.sample('pause')
                wait(20)
            buttons = self.ev3.buttons.pressed()
            self.release()
            if Button.DOWN in buttons:
                return
            if Button.LEFT in buttons:
                index = (index - 1) % len(NODES)
            elif Button.RIGHT in buttons:
                index = (index + 1) % len(NODES)
            elif Button.CENTER in buttons:
                self.old_node = NODES[index]
                self.sample('pause', 'node_manual', self.old_node)
                self.file.flush()
                return

    def pause(self, node_menu=False):
        self.hold()
        self.sample('pause', 'pause_start')
        self.release()
        if node_menu:
            self.mark_node()
        while True:
            self.show('PAUSED lap %d' % self.lap, 'CENTER resume',
                      'LEFT label node', 'RIGHT cross tape', 'DOWN save/finish')
            while not self.ev3.buttons.pressed():
                self.sample('pause')
                wait(20)
            buttons = self.ev3.buttons.pressed()
            self.release()
            if Button.DOWN in buttons:
                return False
            if Button.LEFT in buttons:
                self.mark_node()
            elif Button.RIGHT in buttons:
                self.cross()
            elif Button.CENTER in buttons:
                self.reset_follow()
                self.sample('follow', 'resume')
                return True

    def run(self):
        self.show('TASK 5 EXPLORE', 'Left B / Right C', 'Colour S3 Gyro S1',
                  'Tape on LEFT', 'CENTER start')
        while Button.CENTER not in self.ev3.buttons.pressed():
            if Button.DOWN in self.ev3.buttons.pressed():
                return
            wait(20)
        self.release()
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
        reason = 'user_finish'
        try:
            self.file = open(self.name, 'w')
            self.file.write('time_ms,lap,mode,event,node,tape_left,reflection,target,'
                            'heading_deg,x_mm,y_mm,net_mm,travel_mm,left_deg,right_deg,'
                            'left_cmd_dps,right_cmd_dps,gyro_drift_dps\n')
            self.sample('pause', 'start_reference')
            self.reset_follow()
            self.show('RUNNING lap 1', 'CENTER pause', 'UP label node', 'DOWN finish')
            while True:
                buttons = self.ev3.buttons.pressed()
                if Button.DOWN in buttons:
                    break
                if Button.CENTER in buttons or Button.UP in buttons:
                    if not self.pause(Button.UP in buttons):
                        break
                    self.show('RUNNING lap %d' % self.lap, 'CENTER pause',
                              'UP label node', 'DOWN finish')
                self.follow()
        except StopIteration:
            reason = 'user_stop_during_cross'
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
            self.show('STOPPED', reason, self.name, 'Copy CSV to laptop')
            print('Saved:', self.name, 'End:', reason)


if __name__ == '__main__':
    Survey().run()
