#!/usr/bin/env pybricks-micropython
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
TURN_TIMEOUT_MS = 60000
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
        name = 'task5_run_%03d.csv' % index
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
        self.state = 'S_TO_N'
        self.expected = 'M'
        self.stage_mark = 0.0
        self.dark_count = 0
        self.n_saved = 0.0
        self.straight_heading = 0.0
        self.goal_reached = False
        self.corner_recovery_done = False
        self.corner_watch_ms = 0
        self.corner_watch_net = 0.0

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
                   self.state, self.expected, self.net-self.stage_mark,
                   SENSOR_AHEAD_MM, SENSOR_RIGHT_MM)
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


# Assumptions and distance gates, in mm. Not lecture example lengths.
SENSOR_AHEAD_MM = 65.0
SENSOR_RIGHT_MM = 60.0
S_M_REFERENCE_MM = 310.57   # inner recorded S -> M signed advance
S_N_REFERENCE_MM = 776.03  # inner recorded S -> N signed advance
N_WINDOW = (S_N_REFERENCE_MM - 100, S_N_REFERENCE_MM + 130)
K_WINDOW = (300.0, 780.0)  # local advance after the N turn
GOAL_WINDOW = (230.0, 850.0)  # local advance after crossing K
BLACK_LIMIT = 20
DARK_CONFIRM = 2
K_COMMIT_START_MM = 500.0
K_COMMIT_END_MM = 580.0
# Trial guard around the observed N--K net costs (about 515--547 mm).
# These bounds are tuning choices, not surveyed D2 branch coordinates.
TURN_TIMEOUT_MS = 60000
NO_PROGRESS_MS = 300000

# Mean included absolute centre-travel costs from task5_plan.py, mm.
EDGES = (('S','M',501.8869),('S','C',693.3320),('S','D4',348.9262),
         ('M','N',691.8660),('M','G2',442.6330),('N','C',1005.9729),
         ('N','K',576.1681),('C','D1',100.4262),('K','D2',89.4307),
         ('K','G1',639.6981))

def planned_route():
    graph = {}
    for a,b,w in EDGES:
        graph.setdefault(a,{})[b] = w
        graph.setdefault(b,{})[a] = w
    distances, parents, done = {'S':0.0}, {}, set()
    while True:
        remaining = [n for n in distances if n not in done]
        if not remaining:
            raise RuntimeError('No planned route')
        node = min(remaining, key=lambda n:distances[n])
        if node == 'G1':
            route=['G1']
            while route[-1]!='S': route.append(parents[route[-1]])
            return route[::-1],distances['G1']
        done.add(node)
        for other,w in graph[node].items():
            cost=distances[node]+w
            if cost < distances.get(other,float('inf')):
                distances[other],parents[other] = cost,node


def heading_error(target, actual):
    return (target-actual+180) % 360 - 180


def check_abort(s):
    if Button.DOWN in s.ev3.buttons.pressed():
        raise StopIteration('user_abort')


def gyro_straight(s, distance, heading, event):
    start=s.net
    watchdog=s.clock.time()
    best=0.0
    while s.net-start < distance:
        check_abort(s)
        s.sample('junction')
        progress=s.net-start
        if progress>best+5:
            best,watchdog=progress,s.clock.time()
        if s.clock.time()-watchdog>NO_PROGRESS_MS:
            raise RuntimeError('no_progress_during_junction')
        turn=clip(3.0*heading_error(heading,s.heading),-70,70)
        s.drive(100-turn,100+turn)
        wait(CONTROL_MS)
    s.hold()
    s.sample('junction',event)


def turn_to(s, target):
    started=s.clock.time()
    while True:
        check_abort(s)
        s.sample('junction')
        error=heading_error(target,s.heading)
        if abs(error)<=2:
            break
        if s.clock.time()-started>TURN_TIMEOUT_MS:
            raise RuntimeError('gyro_turn_timeout')
        w=clip(4*abs(error),45,160)*(1 if error>0 else -1)
        s.drive(-w,w)
        wait(CONTROL_MS)
    s.hold()
    s.sample('junction','turn_end')


def reacquire(s, nominal_heading, sweep=20):
    # A short stationary sweep finds an edge value near the outgoing heading.
    # Score both reflection error and angular departure. DOWN remains active.
    best=(1e9,None)
    for offset in (-sweep,sweep):
        target=nominal_heading+offset
        started=s.clock.time()
        while abs(heading_error(target,s.heading))>2:
            check_abort(s)
            s.sample('acquire')
            score=abs(s.reflection-TARGET)+0.2*abs(heading_error(nominal_heading,s.heading))
            if score<best[0]:best=(score,s.heading)
            if s.clock.time()-started>TURN_TIMEOUT_MS:
                raise RuntimeError('edge_sweep_timeout')
            w=55*(1 if heading_error(target,s.heading)>0 else -1)
            s.drive(-w,w)
            wait(CONTROL_MS)
        s.hold()
    if best[1] is None or best[0]>18:
        raise RuntimeError('outgoing_edge_not_found')
    turn_to(s,best[1])
    s.sample('acquire','edge_reacquired')
    s.reset_follow()


def run_route():
    s=Survey()
    s.tape_left=False
    s.lap=2
    route,cost=planned_route()
    if route != ['S','M','N','K','G1']:
        raise RuntimeError('Executive does not support this planned route')
    s.show('TASK 5 RUN', 'S M N K G1', 'At S towards M',
           'Tape RIGHT of sensor', 'CENTER start', 'DOWN abort')
    while Button.CENTER not in s.ev3.buttons.pressed():
        if Button.DOWN in s.ev3.buttons.pressed():return
        wait(20)
    s.release();s.hold()
    s.show('Gyro drift check','KEEP STILL 10 s')
    wait(1000);initial=s.gyro.angle();wait(DRIFT_MS)
    s.drift=(s.gyro.angle()-initial)/(DRIFT_MS/1000.0)
    if abs(s.drift)>1:
        s.show('Gyro drift too high','Reconnect while still','Restart program')
        return
    s.gyro.reset_angle(0);s.left.reset_angle(0);s.right.reset_angle(0)
    s.clock.reset();s.name=unique_file()
    reason='not_completed';goal_time=None
    try:
        s.file=open(s.name,'w')
        s.file.write('time_ms,lap,mode,event,node,tape_left,reflection,target,'
                     'heading_deg,x_mm,y_mm,net_mm,travel_mm,left_deg,right_deg,'
                     'left_cmd_dps,right_cmd_dps,gyro_drift_dps,state,expected_node,'
                     'stage_net_mm,sensor_ahead_mm,sensor_right_mm\n')
        s.sample('start','run_start','S');s.file.flush();s.reset_follow()
        last_screen=-1000;watchdog=s.clock.time();best=s.net
        while True:
            check_abort(s)
            now=s.sample('run')
            if now-last_screen>300:
                s.show(s.state,'Next: '+s.expected,'t %.1f s'%(now/1000.0),
                       'd %.0f mm'%(s.net-s.stage_mark),'DOWN abort')
                last_screen=now
            if s.net>best+10:
                best,watchdog=s.net,now
            if now-watchdog>NO_PROGRESS_MS:
                raise RuntimeError('no_net_progress')
            if s.state=='S_TO_N':
                if s.expected=='M' and s.net>=S_M_REFERENCE_MM:
                    s.expected='N'
                    s.sample('run','M_odometry_estimate','M')
                inside=(N_WINDOW[0]<=s.net<=N_WINDOW[1] and
                        abs(heading_error(90,s.heading))<30)
                s.dark_count=s.dark_count+1 if inside and s.reflection<BLACK_LIMIT else 0
                if s.net>N_WINDOW[1]:raise RuntimeError('N_not_detected_in_window')
                if s.dark_count>=DARK_CONFIRM:
                    s.hold();s.state='N_RIGHT_TURN';s.sample('junction','N_dark_gate','N')
                    # Turn at INNER N; do not advance to the outer edge.
                    s.sample('junction','N_turn_without_advance')
                    turn_to(s,0)
                    s.tape_left=True
                    reacquire(s,0)
                    s.stage_mark=s.net;s.state='N_TO_K';s.expected='K';s.dark_count=0
                    watchdog=s.clock.time();best=s.net
                    continue
            elif s.state=='N_TO_K':
                progress=s.net-s.stage_mark
                if progress>K_WINDOW[1]:
                    raise RuntimeError('K_alignment_not_found')
                if (progress>=K_COMMIT_START_MM and
                        abs(heading_error(0,s.heading))<18):
                    s.hold();s.state='K_STRAIGHT'
                    s.sample('junction','K_approach_odometry_override')
                    # Align before the D2 peel and suppress the colour follower
                    # through the junction. K is inferred, not optically identified.
                    turn_to(s,0)
                    remaining=max(0.0,K_COMMIT_END_MM-(s.net-s.stage_mark))
                    gyro_straight(s,remaining,0,'K_cross_end')
                    reacquire(s,0,sweep=35)
                    s.corner_watch_ms=s.clock.time();s.corner_watch_net=s.net
                    s.stage_mark=s.net;s.state='K_TO_G1';s.expected='G1';s.dark_count=0
                    s.reset_follow();watchdog=s.clock.time();best=s.net
                    continue
            elif s.state=='K_TO_G1':
                progress=s.net-s.stage_mark
                # One map-directed right-turn recovery at the known G1 corner.
                # Trigger on lack of signed advance, not on a single dark sample.
                if s.net-s.corner_watch_net>=15:
                    s.corner_watch_ms=now;s.corner_watch_net=s.net
                if (not s.corner_recovery_done and
                        now-s.corner_watch_ms>=4000 and
                        15<progress<230 and
                        abs(heading_error(0,s.heading))<50):
                    s.hold();s.state='G1_CORNER_RECOVERY'
                    s.sample('junction','G1_corner_recovery_start')
                    s.corner_recovery_done=True
                    turn_to(s,-90)
                    reacquire(s,-90,sweep=45)
                    s.state='K_TO_G1'
                    s.sample('run','G1_corner_recovery_end')
                    s.corner_watch_ms=s.clock.time();s.corner_watch_net=s.net
                    s.reset_follow();watchdog=s.clock.time();best=s.net
                    continue
                inside=(GOAL_WINDOW[0]<=progress<=GOAL_WINDOW[1] and
                        abs(heading_error(-90,s.heading))<35)
                s.dark_count=s.dark_count+1 if inside and s.reflection<BLACK_LIMIT else 0
                if progress>GOAL_WINDOW[1]:raise RuntimeError('G1_not_detected_in_window')
                if s.dark_count>=DARK_CONFIRM:
                    goal_time=s.clock.time()/1000.0
                    s.hold();s.state='FINISHED';s.expected='G1';s.goal_reached=True
                    s.sample('finish','G1_dark_gate','G1')
                    reason='goal_detected_check_video'
                    break
            s.follow()
    except StopIteration:
        reason='user_abort'
    except KeyboardInterrupt:
        reason='keyboard_interrupt'
    except Exception as error:
        reason='error_'+str(error).replace(',',';').replace('\n',' ')
        print('RUN ERROR:',repr(error))
    finally:
        s.hold()
        if s.file:
            try:
                s.sample('finish',reason)
                s.file.flush()
            finally:s.file.close()
        s.left.stop();s.right.stop()
        s.show('STOPPED', 'G1? Check video' if s.goal_reached else 'NOT COMPLETED',
               '%.2f s'%goal_time if goal_time is not None else reason[:24],s.name)
        print('Route:',route,'Planned contour cost mm:',cost)
        print('End:',reason,'Goal detection s:',goal_time,'CSV:',s.name)


if __name__=='__main__':
    run_route()
