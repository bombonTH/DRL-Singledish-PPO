import numpy as np
import sqlite3

from multiprocessing import shared_memory
from typing import Tuple, Optional, Union, List

import astropy.units as u
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.time import Time, TimeDelta
from astropy.coordinates.erfa_astrom import erfa_astrom, ErfaAstromInterpolator
from astropy.coordinates import golden_spiral_grid

from gymnasium import Env, register, make, make_vec
from gymnasium.spaces import Dict, Box, Discrete
from gymnasium.core import ActType, ObsType, RenderFrame

import cv2
from time import perf_counter

from dataclasses import dataclass
from functools import wraps

import os
ASSET_PATH = os.path.join(os.path.dirname(__file__), 'canvas')

class SingleDish(Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 24,
    }

    def __init__(self, render_mode='rgb_array', max_steps=1024, random_init_step=False, aps=1, rank=None, dilation=1, random=True, random_mag=0.2, hit_margin=0.01, hit_required=10, show=False, default_reward=0.0, num_clusters=4, num_obstacles=0, empty_sky_final=False, fixed_scenario=False):
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_value = 0.0
        self.max_value = 1.0

        self.non_spatial_dim = 7
        self.spatial_dim = 64
        self.num_ch = 6
        self.non_spacial_shape = (self.non_spatial_dim,)
        self.spatial_shape = (self.num_ch, self.spatial_dim, self.spatial_dim)

        self.feature_flatten_shape = (self.num_ch * self.spatial_dim * self.spatial_dim,) + (self.non_spatial_dim,)
        self.feature_original_shape = (self.spatial_shape, self.non_spacial_shape,)

        self.observation_shape = (self.num_ch, self.spatial_dim, self.spatial_dim,)

        self.action_space = Box(low=self.min_action,
                                high=self.max_action,
                                shape=(2,),
                                dtype=np.float32)

        self.observation_space = Box(low=self.min_value,
                                     high=self.max_value,
                                     shape=self.observation_shape,
                                     dtype=np.float32)

        self.reward_range = (-1.0, 1.0)
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.random_init_step = random_init_step
        self.idle_step = 0
        self.aps = aps
        self.rank = rank
        self.dilation = dilation
        self.random = random
        self.random_mag = random_mag
        self.show = show
        self.hit_margin = hit_margin
        self.hit_required = hit_required
        self.default_reward = default_reward
        self.num_clusters = num_clusters
        self.num_obstacles = num_obstacles
        self.empty_sky_final = empty_sky_final
        self.fixed_scenario = fixed_scenario
        noise = [0, 0.1, 0.25, 1]
        self.noise_factor = noise[int(self.rank / 4)]
        self.last_hit = 0
        
        self._seed = 0
        self.cur_episode = 0
        self.cur_step = self.init_step = 0 if not self.random_init_step else int(np.random.uniform(0, 0.8) * self.max_steps)
        self.reward = 0.0
        self.cum_reward = 0.0
        self.video_buffer = []
        self.font = cv2.FONT_HERSHEY_PLAIN

        self.antenna = Antenna(radius=10, random_torque=self.random, random_mag=self.random_mag)
        self.antenna.reset(random=self.random)
        self.obstacles = Obstacle()
        self.targets = None
        self.targets_clusters = None
        self.action = np.array([0, 0]).astype(np.float32)
        ra = []
        dec = []
        ra_center = []
        dec_center = []
        tgt_span = []
        
        for idx in range(self.num_clusters):
            if self.fixed_scenario:
                span = idx + 1
                _dec_center = 10 * idx
                _ra_center = 20 * idx - 180
            else:
                span = np.random.randint(1, 6)
                _dec_center = np.random.randint(-84, 84)
                _ra_center = np.random.randint(-180, 180)
            
            _ra, _dec = np.meshgrid(np.r_[-span:span:1], np.r_[_dec_center-span:_dec_center+span:1], sparse=False)
            _ra = np.array([_ra_center + r / np.cos(np.radians(d)) for r, d in zip(_ra, _dec)], dtype=np.float32)
            ra = np.hstack((ra, _ra.flatten()))
            dec = np.hstack((dec, _dec.flatten()))
            ra_center = np.hstack((ra_center, _ra_center))
            dec_center = np.hstack((dec_center, _dec_center))
            tgt_span = np.hstack((tgt_span, span / 6))
            
        self.targets = Radec(ra=ra, dec=dec, time='2021-08-16 05:00:00', dilation=self.dilation / self.aps, hit_required=self.hit_required)
        self.targets_clusters = Radec(ra=ra_center, dec=dec_center, time='2021-08-16 05:00:00', dilation=self.dilation / self.aps, hit_required=self.hit_required, span=tgt_span)
        self.sun = Sun()
        print(f'Env:{self.rank} num_clusters: {num_clusters} num_obstacles: {num_obstacles}')
        # self.targets = Golden_Spiral(time='2021-08-16 05:00:00', dilation=self.dilation / self.aps, hit_required=self.hit_required, master=self.rank==0, buffer_length=self.max_steps)

        for _ in range(self.num_obstacles):
            _x = np.random.randint(-70, 70)
            _y = np.random.randint(-70, 70)
            span = np.random.randint(10, 20)
            self.obstacles.add_xy(_x-span,_x+span,_y-span,_y+span)
        
        self.obstructed = False
        self.closest_tgt = None
        self.zenith_dist = None
        self.done_count = 0
        self.canvas = np.zeros((self.spatial_dim, self.spatial_dim, 6)).astype(np.float32)
        self.minimap = np.zeros((self.spatial_dim, self.spatial_dim))
        try:
            self.bound = np.load(os.path.join(ASSET_PATH, 'bound64px.npy')).astype(np.uint8)
            self.reward_weight_exp = np.load(os.path.join(ASSET_PATH, 'weight_matrix_exp.npy')).astype(np.float32)
            self.reward_weight_lnr = np.load(os.path.join(ASSET_PATH, 'weight_matrix_lnr.npy')).astype(np.float32)
        except FileNotFoundError as e:
            print(f"Error loading required asset file. Please ensure a 'canvas' directory with the necessary files is in your repository root: {e}")
            raise

    def seed(self, seed):
        self._seed = seed
        np.random.seed(seed)
        
    def cal_reward(self):
        reward = 0
        closest_tgt = self.targets.get_closest_dist(self.antenna.xy, self.obstacles)
        zenith_dist = self.antenna.get_distance_from_zenith() 
        hits, dones, just_hit, just_done = self.targets.check_hit(self.antenna.xy, margin=self.hit_margin)

        if self.obstructed or self.antenna.out_of_bound:
            just_hit = just_done = 0
        if just_hit > 0:
            self.last_hit = self.cur_step
            
        if just_done > 0:
            reward += np.abs(just_done * 1)
            closest_tgt = None      
        else:
            gain_dist = self.closest_tgt - closest_tgt if self.closest_tgt is not None else 0
            reward += gain_dist * 10 if gain_dist > 0 else gain_dist * 11
        
        self.done_count += just_done * 1
        self.closest_tgt = closest_tgt
        self.zenith_dist = zenith_dist
        
        if not self.empty_sky_final and self.targets.sky_count() == 0:
            reward += 0
        reward = min(0, reward) if self.antenna.out_of_bound else reward
        return reward
        

        
    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        reward = self.default_reward
        self.action = np.sin(action * np.pi / 2)
        self.targets.update()
        self.targets_clusters.update()
        self.sun.update()

        self.obstructed = self.obstacles.check_obstruct(self.antenna.elaz, self.antenna.xy)
        
        reward = self.cal_reward()
        
        terminated = self.targets.all_done or (self.targets.sky_count() == 0 and self.empty_sky_final and self.done_count > 0)
        truncated = self.cur_step >= self.max_steps
        
        if terminated:
            reward += min(np.divide(self.max_steps - self.cur_step, 4096), 1) * 10
        
        self.reward = reward      
        self.cum_reward += reward
        
        self.antenna.move(self.action, 1000 / self.aps)
        self.cur_step += 1

        obs = self.render(show=self.show)
        self.idle_step += 1 * (reward < self.default_reward + 0.001)
        
        if self.cur_step % 300 == 0:
            self.obstacles.update()

        info = {'episode_number': self.cur_episode, 
                'init_step': self.init_step,
                'num_step': self.cur_step, 
                'cum_reward': self.cum_reward, 
                'reward': reward, 
                'done': self.done_count,
                'features': np.concatenate((self.antenna.xy, self.targets.get_closest_posit(self.antenna.xy, self.obstacles)), axis=None),
                'antenna': self.antenna.xy,
                'targets': self.targets_clusters.get_target_info(self.obstacles)}
                
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None, random=False) -> Tuple[ObsType, dict]:
        self.cur_episode += 1
        self.cur_step = self.init_step = 0 if not self.random_init_step else int(np.random.uniform(0, 0.8) * self.max_steps)
        self.reward = 0
        self.cum_reward = 0
        self.action = np.array([0.0, 0.0]).astype(np.float32)
        self.idle_step = 0
        self.antenna.reset(random=self.random)
        self.obstacles.update()
        self.last_hit = 0
        self.done_count = 0
        self.targets.reset()
        self.targets_clusters.reset()
        ra = []
        dec = []
        ra_center = []
        dec_center = []
        tgt_span = []
        
        for idx in range(self.num_clusters):
            if self.fixed_scenario:
                span = idx + 1
                _dec_center = 10 * idx
                _ra_center = 20 * idx - 180
            else:
                span = np.random.randint(1, 6)
                _dec_center = np.random.randint(-84, 84) 
                _ra_center = np.random.randint(-180, 180)
            
            _ra, _dec = np.meshgrid(np.r_[-span:span:1], np.r_[_dec_center -span:_dec_center+span:1], sparse=False)
            _ra = np.array([_ra_center + r / np.cos(np.radians(d)) for r, d in zip(_ra, _dec)], dtype=np.float32)
            ra = np.hstack((ra, _ra.flatten()))
            dec = np.hstack((dec, _dec.flatten()))
            ra_center = np.hstack((ra_center, _ra_center))
            dec_center = np.hstack((dec_center, _dec_center))
            tgt_span = np.hstack((tgt_span, span / 6))
            
        self.targets = Radec(ra=ra, dec=dec, time='2021-08-16 05:00:00', dilation=self.dilation / self.aps, hit_required=self.hit_required)
        self.targets_clusters = Radec(ra=ra_center, dec=dec_center, time='2021-08-16 05:00:00', dilation=self.dilation / self.aps, hit_required=self.hit_required, span=tgt_span)
            
        self.closest_tgt = None
        self.zenith_dist = None

        self.canvas = np.zeros((self.spatial_dim, self.spatial_dim, 6)).astype(np.float32)
        obs = self.render(show=False)
        info = {'episode_number': self.cur_episode,
                'init_step': self.init_step,
                'num_step': self.cur_step,
                'features': np.concatenate((self.antenna.xy, self.targets.get_closest_posit(self.antenna.xy, self.obstacles)), axis=None),
                'antenna': self.antenna.xy,
                'targets': self.targets_clusters.get_target_info(self.obstacles)}
        return obs, info

    def render(self, show=False, capture=False) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        self.canvas[:, :, 5] = np.copy(self.canvas[:,:,4]).astype(np.float32)
        self.canvas[:, :, 4] = np.copy(self.canvas[:,:,3]).astype(np.float32)
        self.canvas[:, :,:4] = 0.0
        
        self.canvas[:, :, 0] = self.antenna.draw(self.canvas[:, :, 0])
        self.canvas[:, :, 1] = self.targets.draw(self.canvas[:, :, 1])
        self.canvas[:, :, 2] = self.obstacles.draw(self.canvas[:, :, 2])
        self.canvas[:, :, 2] = self.sun.draw(self.canvas[:, :, 2])
        self.canvas[:, :, 3] = self.targets.draw_relative(self.antenna.xy, self.canvas[:, :, 3], radius = 10, obstacles=self.obstacles, hit_mask=False, scale='fisheye')
        
        action_y = (self.action[0] + 1) / 2
        action_x = (self.action[1] + 1) / 2
        action_y2 = self.canvas[0,0,4]
        action_x2 = self.canvas[0,1,4]
        action_y3 = self.canvas[0,0,5]
        action_x3 = self.canvas[0,1,5]
        
        self.canvas[0,0,0] = min(np.divide(self.max_steps - self.cur_step, 4096), 1)
        self.canvas[0,1,0] = action_y
        self.canvas[0,2,0] = action_x
        self.canvas[0,3,0] = action_y2
        self.canvas[0,4,0] = action_x2
        self.canvas[0,5,0] = action_y3
        self.canvas[0,6,0] = action_x3
        
        self.canvas[0,0,3] = action_y
        self.canvas[0,1,3] = action_x
        
        if show:
            try:
                overlay =  np.load(os.path.join(ASSET_PATH, 'canvas.npy')).astype(np.uint8)
            except FileNotFoundError:
                print("Canvas overlay not found. Using a black canvas for display.")
            
            crosshair = np.zeros((self.spatial_dim * 4, self.spatial_dim * 4, 3)).astype(np.uint8)

            cv2.drawMarker(crosshair, np.divide(crosshair.shape[:2], 2).astype(int), 255, cv2.MARKER_CROSS, 30, 4, cv2.LINE_AA)

            bgr_ch = np.multiply(self.canvas[:, :, :3], 255).astype(np.uint8)
            bgr_ch = cv2.resize(bgr_ch, (self.spatial_dim * 4, self.spatial_dim * 4), interpolation=cv2.INTER_NEAREST)
            b_ch = cv2.cvtColor(bgr_ch[:, :, 0].astype(np.uint8), cv2.COLOR_GRAY2RGB)
            g_ch = cv2.cvtColor(bgr_ch[:, :, 1].astype(np.uint8), cv2.COLOR_GRAY2RGB)
            r_ch = cv2.cvtColor(bgr_ch[:, :, 2].astype(np.uint8), cv2.COLOR_GRAY2RGB)
            bgr_ch = np.maximum(overlay, bgr_ch)
            
            gry_ch = np.multiply(self.canvas[:, :, 3:], 255).astype(np.uint8)
            gry_ch = cv2.resize(gry_ch, (self.spatial_dim * 4, self.spatial_dim * 4), interpolation=cv2.INTER_NEAREST)
            t_ch = cv2.cvtColor(gry_ch[:,:,0].astype(np.uint8), cv2.COLOR_GRAY2RGB)
            o_ch = cv2.cvtColor(gry_ch[:,:,1].astype(np.uint8), cv2.COLOR_GRAY2RGB)
            gry_ch = np.maximum(crosshair, gry_ch)
            
            rgb_img = np.hstack([bgr_ch, b_ch, g_ch, r_ch]).astype(np.uint8)
            gry_img = np.hstack([gry_ch, t_ch, o_ch, o_ch]).astype(np.uint8)
            img = np.vstack([rgb_img, gry_img])
            
            texts = [
                     f'Reward: {self.reward:+06.3f}',
                     f'Cumulative Reward: {self.cum_reward:+06.3f}',
                     f'Dones: {self.done_count}',
                     f'Sq.deg/hr: {(self.done_count * 11309.0673773 / self.cur_step):.2f}',
                     f'Obstructed: {self.obstructed}',
                     f'Out of bound: {self.antenna.out_of_bound}',
                     f'Step: {self.cur_step}',
                     f'Idle: {self.idle_step}',
                     f'EP: {self.cur_episode}',
                     # f'Dilation: {self.dilation}x',
                     f'Action: {self.action[0]:+04.2f}, {self.action[1]:+04.2f}',
                     f'Target Count: {self.targets.sky_count()}',
                     f'Zenith Dist: {self.zenith_dist}',
                     # f'Az: {np.degrees(self.antenna.elaz[1]):+04.2f}, El: {np.degrees(self.antenna.elaz[0]):+04.2f}',
                     # f'X: {np.degrees(self.antenna.xy[0]):+04.2f}, Y: {np.degrees(self.antenna.xy[1]):+04.2f}',
                     ]
                            
            if show:
                for i, text in enumerate(texts):
                    img = cv2.putText(img, text, (10, 20 * (i + 1)), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow(f'Env{self.rank} - Dilation: {self.dilation}x', img)
                cv2.waitKey(1)
            
        return (self.canvas.transpose((2, 0, 1))).astype(np.float32)
    
class Point(object):
    def __init__(self, name, layer, xy: np.ndarray = np.array([0, 0]), radius: float = 2.0, dwell=60):
        self.name = name
        self.layer = layer
        self.xy = xy
        self.elaz = xy2altaz(xy)
        self.radius = radius
        self.dwell = dwell

        self.lim = np.radians([50, 80])

        self.hit = 0
        self.done = False
        self.out_of_bound = self.check_bound()
        self.obstructed = False

        self.az_rate = 0
        self.el_rate = 0
        self.x_rate = 0
        self.y_rate = 0

        self.dif_x = np.pi / 2
        self.dif_y = np.pi / 2

    def update(self, time=1):
        elaz, xy = self.step(time)
        self.az_rate = (elaz[1] - self.elaz[1]) / time if time > 0 else 0
        self.el_rate = (elaz[0] - self.elaz[0]) / time if time > 0 else 0
        self.x_rate = (xy[0] - self.xy[0]) / time if time > 0 else 0
        self.y_rate = (xy[1] - self.xy[1]) / time if time > 0 else 0
        self.elaz, self.xy = elaz, xy
        self.out_of_bound = self.check_bound()
        return self

    def step(self, time):
        return self.elaz, self.xy

    def check_bound(self):
        return not (-self.lim[0] <= self.xy.x <= self.lim[0] and -self.lim[1] <= self.xy.y <= self.lim[1])

    def distance_to(self, other):
        return self.elaz.distance_to(other.elaz)

    def draw(self, canvas):
        if self.done:
            return
        canvas = np.copy(canvas)
        zenith_angle = (np.pi / 2) - self.elaz.el
        shift = 6

        h = canvas.shape[0]
        w = canvas.shape[1]
        hor_h = int(0.5 * h) - 1
        hor_w = int(0.5 * w) - 1
        pix_per_deg = hor_h / 180
        radius = self.radius * pix_per_deg

        y = h / 2 - (zenith_angle * np.cos(self.elaz.az) / np.pi * hor_h)
        x = (zenith_angle * np.sin(self.elaz.az) / np.pi * hor_w) + w / 2

        center = np.multiply(np.array([x, y]), np.power(2, shift)).astype(np.uint)
        radius = np.multiply(radius, np.power(2, shift)).astype(np.uint)

        cv2.circle(canvas, center, radius=radius, color=255, lineType=cv2.LINE_AA, thickness=cv2.FILLED, shift=shift)
        return canvas


class Motor(object):
    def __init__(self, max_rpm=1500, motor_moi=20.5, load_moi=41.0, motor_peak_torque=23.3):
        self.max_rpm = max_rpm
        self.motor_moi = motor_moi
        self.load_moi = load_moi
        self.motor_peak_torque = motor_peak_torque
        self.rpm = 0
        self.ms_per_rpm = 2 * np.pi * (self.motor_moi + self.load_moi) / (600 * self.motor_peak_torque)

    def spin(self, order, time, load_torque=0.0) -> float:
        load_torque = load_torque * self.motor_peak_torque
        ordered_rpm = order * self.max_rpm
        dif_rpm = ordered_rpm - self.rpm
        transition_time = abs(dif_rpm * self.ms_per_rpm * (1 - load_torque if dif_rpm > 0 else 1 + load_torque))
        if time >= transition_time:
            transition_rpm = (ordered_rpm + self.rpm) / 2
            cruising_time = time - transition_time
            self.rpm = ordered_rpm
        else:
            ordered_rpm = self.rpm + dif_rpm * time / transition_time
            transition_rpm = (ordered_rpm + self.rpm) / 2
            transition_time = time
            cruising_time = 0
            self.rpm = ordered_rpm
        return (transition_rpm * transition_time + ordered_rpm * cruising_time) / 60000

    def get_rpm(self):
        return self.rpm / self.max_rpm

    def set_rpm(self, rpm):
        self.rpm = rpm * self.max_rpm


class Antenna(object):
    SGMG_13A2A = {'max_rpm': 1500, 'motor_moi': 20.5, 'load_moi': 41, 'motor_peak_torque': 23.3}

    def __init__(self, xy: np.ndarray[2] = np.array([0.0, 0.0]), name='burn_lab', radius: float = 1.5, random_torque=True, random_mag=0.1):
        self.lim = np.radians([80, 80])
        self.y_motor = Motor(**Antenna.SGMG_13A2A)
        self.x_motor = Motor(**Antenna.SGMG_13A2A)
        self.y_gear_ratio = 1.0 / 30000
        self.x_gear_ratio = 1.0 / 59400
        self.y_rad_per_rev = np.radians(360 * self.y_gear_ratio)
        self.x_rad_per_rev = np.radians(360 * self.x_gear_ratio)
        self.random_torque = random_torque
        self.random_mag = random_mag
        self.load_torque = np.random.uniform(-self.random_mag, self.random_mag) * self.random_torque
        self.noise1 = np.random.uniform(1-self.random_mag, 1+self.random_mag)
        self.noise2 = np.random.uniform(1-self.random_mag, 1+self.random_mag)
        self.xy = np.array(xy).astype(np.float32)
        self.elaz = xy2altaz(self.xy)
        self.name = name
        self.radius = radius
        self.req_speed = np.array([0.0, 0.0]).astype(np.float32)
        self.cur_speed = np.array([0.0, 0.0]).astype(np.float32)
        self.out_of_bound = False

    def move(self, req_speed, time):
        self.xy[0] += self.x_motor.spin(req_speed[0], time, load_torque=self.load_torque) * self.x_rad_per_rev * self.noise1
        self.xy[1] += self.y_motor.spin(req_speed[1], time, load_torque=self.load_torque) * self.y_rad_per_rev * self.noise2

        min_x, max_x, min_y, max_y = self.check_bound()

        self.req_speed = req_speed
        self.cur_speed = np.array([self.x_motor.get_rpm() * (not (min_x or max_x)),
                                   self.y_motor.get_rpm() * (not (min_y or max_y))])
        self.x_motor.set_rpm(self.cur_speed[0])
        self.y_motor.set_rpm(self.cur_speed[1])

        self.xy = np.clip(self.xy, -self.lim, self.lim)

        self.elaz = xy2altaz(self.xy)
        self.out_of_bound = np.logical_or(np.logical_or(min_x, max_x), np.logical_or(min_y, max_y))
        
        return

    def check_bound(self) -> tuple[bool, bool, bool, bool]:
        min_x = np.less(self.xy[0], -self.lim[0])
        max_x = np.greater(self.xy[0], self.lim[0])
        min_y = np.less(self.xy[1], -self.lim[1])
        max_y = np.greater(self.xy[1], self.lim[1])
        return min_x, max_x, min_y, max_y

    def get_cur_speed(self):
        return self.cur_speed
        
    def get_distance_from_zenith(self):
        return np.sqrt(np.sum(np.square(self.xy))) / 1.97461463918

    def reset(self, random=False):
        self.xy = np.array([0.0, 0.0]).astype(np.float32)
        self.out_of_bound = False
        self.req_speed = self.cur_speed = np.array([0.0, 0.0]).astype(np.float32)
        self.load_torque = np.random.uniform(-self.random_mag, self.random_mag) * self.random_torque
        self.noise1 = np.random.uniform(1-self.random_mag, 1+self.random_mag)
        self.noise2 = np.random.uniform(1-self.random_mag, 1+self.random_mag)
        if random:
            self.xy = np.random.uniform(low=-self.lim, high=self.lim)
        return self

    def draw(self, canvas):
        assert len(canvas.shape) == 2
        canvas = np.copy(canvas).astype(np.float32)
       
        h = canvas.shape[0]
        w = canvas.shape[1]
        hor_h = (0.5 * h) - 1
        hor_w = (0.5 * w) - 1
        center_h = (0.5 * h) - 0.5
        center_w = (0.5 * w) - 0.5
        radius = self.radius / 180 * np.pi
        grid_x = np.arange(-5, 5+1) / 5 * radius
        grid_y = np.arange(-5, 5+1) / 5 * radius
        xx, yy = np.meshgrid(grid_x, grid_y)
        corners = np.vstack([xx.flatten(), yy.flatten()])
        xys = np.add(self.xy[:, np.newaxis], corners)
        elaces = xy2altaz(xys)
        zenith_angle = np.subtract(np.divide(np.pi, 2), elaces[0])
        
        y = np.multiply(zenith_angle, np.cos(elaces[1]))
        y = np.divide(y, np.pi)
        y = np.multiply(y, hor_h)
        y = np.subtract(center_h, y)
        
        x = np.multiply(zenith_angle, np.sin(elaces[1]))
        x = np.divide(x, np.pi)
        x = np.multiply(x, hor_w)
        x = np.add(center_w, x)
        
        xy = np.vstack([y, x])
        
        xy00 = np.vstack([y, x]).astype(int)
        xy01 = np.vstack([y, np.add(x, 1)]).astype(int)
        xy10 = np.vstack([np.add(y, 1), x]).astype(int)
        xy11 = np.vstack([np.add(y, 1), np.add(x, 1)]).astype(int)
        residual = xy - xy00

        try:
            canvas[*xy00] = np.clip(canvas[*xy00] + 0.5 * np.subtract(1, residual[0]) + 0.5 * np.subtract(1, residual[1]), 0.0, 1.0)
            canvas[*xy10] = np.clip(canvas[*xy10] + 0.5 * residual[0] + 0.5 * np.subtract(1, residual[1]), 0.0, 1.0)
            canvas[*xy01] = np.clip(canvas[*xy01] + 0.5 * np.subtract(1, residual[0]) + 0.5 * residual[1], 0.0, 1.0)
            canvas[*xy11] = np.clip(canvas[*xy11] + 0.5 * residual[0] + 0.5 * residual[1], 0.0, 1.0)
        except IndexError:
            print(np.max(x), np.max(y))
        return canvas



def memoize(func):
    cache = {}

    @wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs) + str(args[-1].shape)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    return wrapper


class Obstacle(object):
    def __init__(self):
        self.ellipses = []
        self.xys = []
        self.polygons = []

    def add_altaz(self, fm_az, to_az, fm_el, to_el):
        self.ellipses.append({'fm_az': fm_az,
                              'to_az': to_az,
                              'fm_el': fm_el,
                              'to_el': to_el})
        return self

    def add_xy(self, fm_x, to_x, fm_y, to_y):
        fm_x, to_x = (fm_x, to_x) if to_x > fm_x else (to_x, fm_x)
        fm_y, to_y = (fm_y, to_y) if to_y > fm_y else (to_y, fm_y)
        self.xys.append({'fm_x': fm_x,
                         'to_x': to_x,
                         'fm_y': fm_y,
                         'to_y': to_y})
        return self

    def add_polygon(self, pts):
        self.polygons.append(pts)
        pass
        
    def update(self):
        for idx, item in enumerate(self.xys):
            bias_x = np.random.randint(-10, 10)
            bias_y = np.random.randint(-10, 10)
            span = np.random.randint(10, 20)
            _x = np.clip((((item['fm_x'] + item['to_x']) / 2) + bias_x), -70, 70)
            _y = np.clip((((item['fm_y'] + item['to_y']) / 2) + bias_y), -70, 70)
            self.xys[idx] = {'fm_x': _x-span, 'to_x': _x+span, 'fm_y': _y-span, 'to_y': _y+span}
            
    # @memoize
    def draw(self, canvas):
        assert len(canvas.shape) == 2
        canvas = np.copy(canvas).astype(np.uint8)
        shift = 0
        factor = np.power(2, shift)
        draw_box = np.subtract(np.multiply(np.array([np.min(canvas.shape), np.min(canvas.shape)]), 0.5), 1)
        center = np.multiply(np.divide(canvas.shape, 2), factor).astype(np.uint)[::-1, np.newaxis]
        pixel_per_rad = np.min(canvas.shape) / np.pi / 2 - 1

        for ellipse in self.ellipses:
            _canvas = np.zeros_like(canvas)
            inner = np.multiply(np.multiply(np.divide(np.subtract(90, ellipse['to_el']), 180), draw_box), factor).astype(np.uint)
            outer = np.multiply(np.multiply(np.divide(np.subtract(90, ellipse['fm_el']), 180), draw_box), factor).astype(np.uint)
            cv2.ellipse(_canvas, center.T[0], outer, -90, ellipse['fm_az'], ellipse['to_az'], color=200, thickness=cv2.FILLED,
                        lineType=cv2.LINE_AA, shift=shift)
            cv2.ellipse(_canvas, center.T[0], inner, -90, 0, 360, color=0, thickness=cv2.FILLED, lineType=cv2.LINE_AA,
                        shift=shift)
            canvas = np.maximum(canvas, _canvas)

        for item in self.xys:
            _canvas = np.zeros_like(canvas)
            xlim = np.arange(item['fm_x'], item['to_x'])
            ylim = np.arange(item['fm_y'], item['to_y'])

            pts = np.hstack([np.vstack([np.full_like(ylim, xlim[0]), ylim]),
                             np.vstack([xlim, np.full_like(xlim, ylim[-1])]),
                             np.vstack([np.full_like(ylim, xlim[-1]), ylim[::-1]]),
                             np.vstack([xlim[::-1], np.full_like(xlim, ylim[0])])])

            elaces = xy2altaz(np.radians(pts))

            zenith_angle = np.subtract(np.divide(np.pi, 2), elaces[0])

            sincos = np.array([np.sin(elaces[1]), np.cos(elaces[1])])
            sign = np.array([[1], [-1]])
            xy = np.multiply(sign, np.multiply(np.multiply(zenith_angle, sincos), pixel_per_rad))
            xy = np.multiply(np.add(center, xy), factor).astype(int)
            xy = xy.T.reshape((1, -1, 2))
            cv2.fillPoly(img=_canvas, pts=xy, color=200, lineType=cv2.LINE_AA, shift=shift)
            canvas = np.maximum(canvas, _canvas)
        
        return np.divide(canvas, 255.0).astype(np.float32)

    def check_obstruct(self, ref_altaz, ref_xy):
        ref_altaz = np.degrees(ref_altaz)
        ref_xy = np.degrees(ref_xy)
        obs_altaz = any([altaz['fm_el'] < ref_altaz[0] < altaz['to_el'] and altaz['fm_az'] < ref_altaz[1] < altaz['to_az'] for altaz in self.ellipses])
        obs_xy = any([xy['fm_x'] < ref_xy[0] < xy['to_x'] and xy['fm_y'] < ref_xy[1] < xy['to_y'] for xy in self.xys])
        obstructed = obs_altaz or obs_xy
        return obstructed
        
    def get_xy_obstructed_mask(self, ref_xy):
        mask = np.full(ref_xy.shape[1], False).astype(bool)
        for xy in self.xys:
            obs_x = np.logical_and(np.radians(xy['fm_x']) < ref_xy[0, :], ref_xy[0, :] < np.radians(xy['to_x']))
            obs_y = np.logical_and(np.radians(xy['fm_y']) < ref_xy[1, :], ref_xy[1, :] < np.radians(xy['to_y']))
            obstructed = np.logical_and(obs_x, obs_y)
            mask = np.logical_or(mask, obstructed)
        return mask

def altaz2xy(altaz: np.ndarray):
    alt, az = altaz[0], altaz[1]
    x = np.arcsin(np.multiply(np.sin(az), np.cos(alt)))
    y = np.arctan2(np.multiply(np.cos(az), np.cos(alt)), np.sin(alt))
    return np.array([x, y])


def xy2altaz(xy: np.ndarray):
    x, y = xy[0], xy[1]
    alt = np.arcsin(np.multiply(np.cos(y), np.cos(x)))
    az = np.arctan2(np.tan(x), np.sin(y))
    return np.array([alt, az], dtype=np.float32)


class Radec(object):
    def __init__(self, time: str = '2021-08-16 05:00:00', ra=0, dec=0, dilation: float = 1.0, hit_required=60, span=None):
        self.spos = SkyCoord(ra=ra, dec=dec, unit=u.deg)
        self.location = EarthLocation(lat=13.7309711 * u.deg, lon=100.7873937 * u.deg, height=15 * u.m)
        self.dilation = dilation
        self.hit_required = hit_required
        self.span = span
        self._time = Time(time)
        self.time = Time(time)
        
        self.timegrid = self.time + np.arange(0, 1 * self.dilation, self.dilation) * u.second
        self.frame = AltAz(obstime=self.timegrid, location=self.location)
        
        self.dones = np.full_like(self.spos, False, dtype=bool)
        self.hit_count = np.zeros_like(self.spos, dtype=np.uint8)
        
        altaz = self.spos[:, np.newaxis].transform_to(self.frame[np.newaxis])
        self.elaces = np.array([altaz.alt.radian, altaz.az.radian], dtype=np.float32)
        self.xys = altaz2xy(self.elaces)
        self.elaz = self.elaces[:, :, 0]
        self.xy = self.xys[:, :, 0]
        
        self.xlim = np.radians(80)
        self.ylim = np.radians(80)
        self.layer = 1
        self.all_done = False
        self.cur_step = 0
        self.prep_elaz(512)
        
    def prep_elaz(self, period=128):
        self.timegrid = self.time + np.arange(0, period * self.dilation, self.dilation) * u.second
        self.frame = AltAz(obstime=self.timegrid, location=self.location)
        
        with erfa_astrom.set(ErfaAstromInterpolator(1000 * u.s)):
            altaz = self.spos[:, np.newaxis].transform_to(self.frame[np.newaxis])
            self.elaces = np.concatenate([self.elaces, np.array([altaz.alt.radian, altaz.az.radian], dtype=np.float32)], axis=-1)
            self.xys = altaz2xy(self.elaces)

    def update(self, time=1):
        self.elaz, self.xy = self.step(time)
        self.all_done = np.all(self.dones)
        return self

    def step(self, time=1, obstacles=None):
        self.time += time * u.second * self.dilation
        if self.elaces is None or self.elaces.shape[-1] == 0 or self.cur_step >= self.elaces.shape[-1]:
            self.prep_elaz(5120)
        elaz= self.elaces[:, :, self.cur_step]
        xy = self.xys[:, :, self.cur_step]
        self.cur_step += 1
        return elaz, xy
        
    def get_closest_dist(self, ref, obstacles=None, radius=85):
        try:
            not_obstructed_mask = np.logical_not(obstacles.get_xy_obstructed_mask(self.xy)) if obstacles else True
            not_dones_mask = np.logical_not(self.dones)
            x_within_lim_mask = np.abs(self.xy[0,:]) < self.xlim
            y_within_lim_mask = np.abs(self.xy[1,:]) < self.ylim
            valid_masks = not_obstructed_mask & not_dones_mask & x_within_lim_mask & y_within_lim_mask
            rel_pos = np.subtract(self.xy[:, valid_masks], ref[:, np.newaxis])
            rel_dist = np.sqrt(np.sum(np.square(rel_pos), axis=0))
            rel_dist = rel_dist / np.radians(radius)
            rel_dist = rel_dist[rel_dist < np.radians(radius)]
            min_dist = np.min(rel_dist) / np.radians(radius)
        except ValueError:
            return 1
        return min_dist if min_dist <= 1 else None     

    def get_target_info(self, obstacles=None):
        target_info = np.zeros((8, 3)) + np.array([2, 2, 0])
        x_within_lim_mask = np.abs(self.xy[0,:]) < np.radians(180)
        y_within_lim_mask = np.abs(self.xy[1,:]) < np.radians(180)
        obstructed_mask = obstacles.get_xy_obstructed_mask(self.xy) if obstacles else False    
        not_obstructed_mask = np.logical_not(obstructed_mask)    
        valid_masks = x_within_lim_mask & y_within_lim_mask
        _span = np.copy(self.span)
        _span[obstructed_mask] = -6
        _xy = np.copy(self.xy)
        target_info[:self.xy[:,valid_masks].shape[1],:3] = np.vstack((self.xy[:,valid_masks], _span[valid_masks] / 6.0)).T
        return target_info
        
    def get_closest_posit(self, ref, obstacles=None):
        try:
            not_obstructed_mask = np.logical_not(obstacles.get_xy_obstructed_mask(self.xy)) if obstacles else True
            not_dones_mask = np.logical_not(self.dones)
            x_within_lim_mask = np.abs(self.xy[0,:]) < self.xlim
            y_within_lim_mask = np.abs(self.xy[1,:]) < self.ylim
            valid_masks = not_obstructed_mask & not_dones_mask
            try:
                pos_valid = self.xy[:, valid_masks]
                rel_pos_valid = np.subtract(pos_valid, ref[:, np.newaxis])
                rel_dist_valid = np.sqrt(np.sum(np.square(rel_pos_valid), axis=0))
                min_dist_valid_id = np.argmin(rel_dist_valid)
                closest_notobstructed_notdone_posit =  self.xy[:, min_dist_valid_id]
            except UnboundLocalError:
                closest_notobstructed_notdone_posit = [0, 0]
            
            try:
                pos_notdone = self.xy[:, not_dones_mask]
                rel_pos_notdone = np.subtract(pos_notdone, ref[:, np.newaxis])
                rel_dist_notdone = np.sqrt(np.sum(np.square(rel_pos_notdone), axis=0))
                min_dist_notdone_id = np.argmin(rel_dist_notdone)
                closest_notdone_posit = self.xy[:, min_dist_valid_id]
            except ValueError:
                closest_notdone_posit = [0, 0]
        except ValueError:
            return [0, 0, 0, 0]
        return np.concatenate((closest_notobstructed_notdone_posit, closest_notdone_posit), axis=None)

    def check_hit(self, ref, margin=0.001) -> tuple[np.ndarray, np.ndarray, bool, bool]:
        ref = ref[:, np.newaxis]
        not_dones = np.logical_not(self.dones)
        targets = self.xy
        rel_pos = np.subtract(targets, ref)
        distance = np.sqrt(np.sum(np.square(rel_pos), axis=0))
        hit_mask = np.logical_and(np.less_equal(distance, margin), not_dones)
        self.hit_count[np.logical_not(hit_mask)] = 0
        self.hit_count[hit_mask] += 1
        self.hit_count[self.dones] = self.hit_required
        
        done_mask = self.hit_count >= self.hit_required
        just_hit = np.sum(hit_mask) > 0
        just_done = (np.sum(done_mask) - np.sum(self.dones)) > 0

        self.dones = done_mask
        
        return hit_mask, self.dones, just_hit, just_done

    def draw(self, canvas, intensity=1/16):
        not_dones = np.logical_not(self.dones)
        valid_elaz = self.elaz[:, not_dones]
        valid_xy = self.xy[:, not_dones]
        zenith_angle = np.subtract(np.divide(np.pi, 2), valid_elaz[0])

        h = canvas.shape[0]
        w = canvas.shape[1]
        hor_h = (0.5 * h) - 1
        hor_w = (0.5 * w) - 1
        
        center_h = (h / 2) - 0.5
        center_w = (w / 2) - 0.5
        
        y = np.multiply(zenith_angle, np.cos(valid_elaz[1]))
        y = np.divide(y, np.pi)
        y = np.multiply(y, hor_h)
        y = np.subtract(center_h, y)
        
        x = np.multiply(zenith_angle, np.sin(valid_elaz[1]))
        x = np.divide(x, np.pi)
        x = np.multiply(x, hor_w)
        x = np.add(center_w, x)
                
        x_mask = np.logical_and(np.less_equal(x, w - 1), np.greater_equal(x, 0))
        y_mask = np.logical_and(np.less_equal(y, h - 1), np.greater_equal(y, 0))
        mask = np.logical_and(x_mask, y_mask)
        mask = np.vstack([mask, mask])
        xy = np.vstack([y, x])
        
        xy00 = np.vstack([y, x]).astype(int)
        xy01 = np.vstack([y, np.add(x, 1)]).astype(int)
        xy10 = np.vstack([np.add(y, 1), x]).astype(int)
        xy11 = np.vstack([np.add(y, 1), np.add(x, 1)]).astype(int)
        residual = xy - xy00

        try:
            canvas[*xy00] = np.clip(canvas[*xy00] + intensity * np.subtract(1, residual[0]) + intensity * np.subtract(1, residual[1]), 0.0, 1.0)
            canvas[*xy10] = np.clip(canvas[*xy10] + intensity * residual[0] + intensity * np.subtract(1, residual[1]), 0.0, 1.0)
            canvas[*xy01] = np.clip(canvas[*xy01] + intensity * np.subtract(1, residual[0]) + intensity * residual[1], 0.0, 1.0)
            canvas[*xy11] = np.clip(canvas[*xy11] + intensity * residual[0] + intensity * residual[1], 0.0, 1.0)
        except IndexError:
            print(np.max(x), np.max(y))
        return canvas
    
    def draw_relative(self, ref, canvas, radius=5, obstacles=None, hit_mask=False, scale='linear'):
        not_obstructed_mask = np.logical_not(obstacles.get_xy_obstructed_mask(self.xy)) if obstacles else True
        not_dones_mask = np.logical_not(self.dones)
        x_within_lim_mask = np.abs(self.xy[0,:]) < self.xlim
        y_within_lim_mask = np.abs(self.xy[1,:]) < self.ylim
        valid_masks = not_obstructed_mask & not_dones_mask & x_within_lim_mask & y_within_lim_mask
        if hit_mask:
            non_complete_hit_mask = self.hit_count < self.hit_required
            valid_masks = valid_masks & non_complete_hit_mask
        rel_pos = np.divide(np.subtract(self.xy[:, valid_masks], ref[:, np.newaxis]), np.radians(radius))
        hit_count = self.hit_count[valid_masks]
        in_scope_mask = np.max(np.abs(rel_pos), axis=0) <= 1
        valid_pos = rel_pos[:, in_scope_mask].reshape([2, -1])
        hit_count = hit_count[in_scope_mask]
        
        h = canvas.shape[0]
        w = canvas.shape[1]
        hor_h = (0.5 * h) - 1
        hor_w = (0.5 * w) - 1                                              
     
        if scale == 'fisheye':
            distance = np.sqrt(np.sum(np.square(valid_pos), axis=0))
            angle = np.arctan2(valid_pos[0,:], valid_pos[1,:])
            xs = np.sin(angle) * np.tanh(distance * 5)
            ys = np.cos(angle) * np.tanh(distance * 5)
            valid_pos = np.array([xs, ys])
            
        if scale == 'log':
            y = np.multiply(np.sign(y), np.sqrt(np.abs(y)))
        
        x = valid_pos[0]
        y = valid_pos[1]
        
        y = np.multiply(y, hor_h)
        y = np.subtract(np.divide(h - 1, 2), y)
           
        x = np.multiply(x, hor_w)
        x = np.add(np.divide(w - 1, 2), x)        

        xy = np.vstack([y, x])

        xy00 = np.vstack([y, x]).astype(int)
        xy01 = np.vstack([y, np.add(x, 1)]).astype(int)
        xy10 = np.vstack([np.add(y, 1), x]).astype(int)
        xy11 = np.vstack([np.add(y, 1), np.add(x, 1)]).astype(int)
        residual = xy - xy00
        
        pixel_intensity = 0.25 if not hit_mask else 0.25 + (self.hit_required - hit_count) * 0.05

        try:
            canvas[*xy00] = np.clip(canvas[*xy00] + pixel_intensity * np.subtract(1, residual[0]) + pixel_intensity * np.subtract(1, residual[1]), 0.0, 1.0)
            canvas[*xy10] = np.clip(canvas[*xy10] + pixel_intensity * residual[0] + pixel_intensity * np.subtract(1, residual[1]), 0.0, 1.0)
            canvas[*xy01] = np.clip(canvas[*xy01] + pixel_intensity * np.subtract(1, residual[0]) + pixel_intensity * residual[1], 0.0, 1.0)
            canvas[*xy11] = np.clip(canvas[*xy11] + pixel_intensity * residual[0] + pixel_intensity * residual[1], 0.0, 1.0)
        except IndexError:
            print(np.max(x), np.max(y), np.max(valid_pos))
        return canvas
        
    def sky_count(self):
        not_dones_mask = np.logical_not(self.dones)
        x_within_lim_mask = np.abs(self.xy[0,:]) < self.xlim
        y_within_lim_mask = np.abs(self.xy[1,:]) < self.ylim
        valid_masks = not_dones_mask & x_within_lim_mask & y_within_lim_mask
        return self.xy[:, valid_masks].shape[-1]

    def reset(self):
        self._time = self._time + 64 * u.second * self.dilation
        self.time = self._time
        self.cur_step=0
        self.xys=self.xys[:,:,64:]
        self.elaces=self.elaces[:,:,64:]
        self.dones = np.full_like(self.spos, False, dtype = bool)
        self.hit_count = np.zeros_like(self.spos, dtype=np.uint8)
        return self
        
class Sun(object):
    def __init__(self, time='2021-08-16 05:00:00', radius = 2.0, dilation: float = 100.0, hit_required=60,):
        self.connection = sqlite3.connect("db/sun.db")
        self.cursor = self.connection.cursor()
        
        self.location = EarthLocation(lat=13.7309711 * u.deg, lon=100.7873937 * u.deg, height=15 * u.m)
        self.radius = radius
        self.dilation = dilation
        self.hit_required = hit_required
        self._time = Time(time)
        self.time = Time(time)
        
        self.elaz = np.array([0, 0], dtype=np.float32)
        self.xy = altaz2xy(self.elaz)
        
        self.timegrid = self.time + np.arange(0, 1 * self.dilation, self.dilation) * u.second
        self.frame = AltAz(obstime=self.timegrid, location=self.location)
        
        # Initialize position
        self.update(0)
        
    def update(self, time=1):
        self.elaz, self.xy = self.step(time)
        return self
        
    def step(self, time):
        """Advances time and calculates the sun's new position."""
        self.time += time * u.second * self.dilation
        self.frame = AltAz(obstime=self.time, location=self.location)
        return self._get_sun_position()
        
        
    def _get_sun_position(self):
        """Reads sun position from cache (SQLite) or calculates it using Astropy."""
        self.cursor = self.connection.cursor()
        time_str = str(self.time)
        sun_position = self._read_sun(time_str)

        if sun_position is None:
            # Cache miss: calculate position
            # print('Not found - calculating position') # debug print
            sun_az_deg, sun_el_deg = self._cal_sun(self.time)
            elaz = np.array([np.radians(sun_el_deg), np.radians(sun_az_deg)], dtype=np.float32)
            xy = altaz2xy(self.elaz)
            self._write_sun(time_str, elaz[1], elaz[0], xy[1], xy[0])
        else:
            # Cache hit: retrieve and convert to numpy array
            # Format: (time, az_rad, elv_rad, y_rad, x_rad)
            time_str, az_rad, elv_rad, y_rad, x_rad = sun_position 
            elaz = np.array([elv_rad, az_rad], dtype=np.float32)
            xy = np.array([x_rad, y_rad], dtype=np.float32)
        return elaz, xy
            
    def _read_sun(self, time_str):
        self.cursor.execute("SELECT * FROM sun WHERE time = ?", (time_str,))
        return self.cursor.fetchone()

    def _cal_sun(self, time):
        """Calculates Sun's AltAz position in degrees."""
        sun = get_sun(time).transform_to(self.frame)
        az = sun.az.deg
        alt = sun.alt.deg
        return az, alt

    def _write_sun(self, time_str, az_rad, elv_rad, y_rad, x_rad):
        self.cursor.execute("INSERT INTO sun VALUES (?, ?, ?, ?, ?)",
                            (time_str, az_rad, elv_rad, y_rad, x_rad))
        self.connection.commit()
        
    def draw(self, canvas, intensity=1/16):
        h = canvas.shape[0]
        w = canvas.shape[1]
        hor_h = (0.5 * h) - 1
        hor_w = (0.5 * w) - 1
        center_h = (0.5 * h) - 0.5
        center_w = (0.5 * w) - 0.5
        radius = self.radius / 180 * np.pi
        grid_x = np.arange(-5, 5+1) / 5 * radius
        grid_y = np.arange(-5, 5+1) / 5 * radius
        xx, yy = np.meshgrid(grid_x, grid_y)
        corners = np.vstack([xx.flatten(), yy.flatten()])
        xys = np.add(self.xy[:, np.newaxis], corners)
        elaces = xy2altaz(xys)
        zenith_angle = np.subtract(np.divide(np.pi, 2), elaces[0])
        
        y = np.multiply(zenith_angle, np.cos(elaces[1]))
        y = np.divide(y, np.pi)
        y = np.multiply(y, hor_h)
        y = np.subtract(center_h, y)
        
        x = np.multiply(zenith_angle, np.sin(elaces[1]))
        x = np.divide(x, np.pi)
        x = np.multiply(x, hor_w)
        x = np.add(center_w, x)
        
        xy = np.vstack([y, x])
        
        xy00 = np.vstack([y, x]).astype(int)
        xy01 = np.vstack([y, np.add(x, 1)]).astype(int)
        xy10 = np.vstack([np.add(y, 1), x]).astype(int)
        xy11 = np.vstack([np.add(y, 1), np.add(x, 1)]).astype(int)
        residual = xy - xy00

        try:
            canvas[*xy00] = np.clip(canvas[*xy00] + 0.5 * np.subtract(1, residual[0]) + 0.5 * np.subtract(1, residual[1]), 0.0, 1.0)
            canvas[*xy10] = np.clip(canvas[*xy10] + 0.5 * residual[0] + 0.5 * np.subtract(1, residual[1]), 0.0, 1.0)
            canvas[*xy01] = np.clip(canvas[*xy01] + 0.5 * np.subtract(1, residual[0]) + 0.5 * residual[1], 0.0, 1.0)
            canvas[*xy11] = np.clip(canvas[*xy11] + 0.5 * residual[0] + 0.5 * residual[1], 0.0, 1.0)
        except IndexError:
            print(np.max(x), np.max(y))
        return canvas
        
        
        
class Golden_Spiral(object):
    def __init__(self, time: str = '2021-08-16 05:00:00', n_points=41253, dilation: float = 1.0, hit_required=2, master=False, buffer_length=3600):
        self.n_points = n_points
        coord = golden_spiral_grid(self.n_points)
        self.mask = (-1.05 * u.rad < coord.lat) & (1.05 * u.rad > coord.lat)
        self.spos = SkyCoord(coord)
        self.location = EarthLocation(lat=13.7309711 * u.deg, lon=100.7873937 * u.deg, height=15 * u.m)
        self.dilation = dilation
        self.hit_required = hit_required
        self._time = Time(time)
        self.time = Time(time)
        self.master = master
        self.buffer_length = buffer_length
                
        self.dones = np.full_like(self.spos, False, dtype=bool)
        self.hit_count = np.zeros_like(self.spos, dtype=np.uint8)
        
        self.xlim = np.radians(80)
        self.ylim = np.radians(80)
        self.layer = 1
        self.all_done = False
        self.cur_step = 0
        self.shm_elaces = shared_memory.SharedMemory(name='elaces')
        self.shm_xys = shared_memory.SharedMemory(name='xys')
        self.elaces = np.ndarray((2, 41253, self.buffer_length), dtype=np.float32, buffer=self.shm_elaces.buf)
        self.xys = np.ndarray((2, 41253, self.buffer_length), dtype=np.float32, buffer=self.shm_xys.buf)
        if self.master:
            self.prep_elaz(self.buffer_length)
        self.elaz = self.elaces[:, :, 0]
        self.xy = self.xys[:, :, 0]
        
    def prep_elaz(self, period=128):
        self.timegrid = self.time + np.arange(0, period * self.dilation, self.dilation) * u.second
        self.frame = AltAz(obstime=self.timegrid, location=self.location)
        with erfa_astrom.set(ErfaAstromInterpolator(1000 * u.s)):
            altaz = self.spos[:, np.newaxis].transform_to(self.frame[np.newaxis])
            self.elaces[:] = np.array([altaz.alt.radian, altaz.az.radian], dtype=np.float32)[:]
            self.xys[:] = altaz2xy(self.elaces)[:]            

    def update(self, time=1):
        self.elaz, self.xy = self.step(time)
        self.all_done = np.all(self.dones)
        return self

    def step(self, time=1):
        self.time += time * u.second * self.dilation
        if self.elaces is None or self.elaces.shape[-1] == 0 or self.cur_step >= self.elaces.shape[-1]:
            self.reset()
            self.step()
            exit()
        elaz = self.elaces[:, :, self.cur_step]
        xy = self.xys[:, :, self.cur_step]
        self.cur_step += 1
        return elaz, xy

    def check_hit(self, ref, margin=0.001) -> tuple[np.ndarray, np.ndarray, bool, bool]:
        ref = ref[:, np.newaxis]
        not_dones = np.logical_not(self.dones)
        targets = self.xy
        rel_pos = np.subtract(targets, ref)
        distance = np.sqrt(np.sum(np.square(rel_pos), axis=0))
        hit_mask = np.logical_and(np.less_equal(distance, margin), not_dones)
        self.hit_count[np.logical_not(hit_mask)] = 0
        self.hit_count[hit_mask] += 1
        self.hit_count[self.dones] = self.hit_required
        
        done_mask = self.hit_count >= self.hit_required
        just_hit = np.sum(hit_mask) > 0
        just_done = (np.sum(done_mask) - np.sum(self.dones)) > 0

        self.dones = done_mask
        
        return hit_mask, self.dones, just_hit, just_done

    def draw(self, canvas):
        not_dones = np.logical_not(self.dones)
        valid_elaz = self.elaz[:, not_dones]
        valid_xy = self.xy[:, not_dones]
        zenith_angle = np.subtract(np.divide(np.pi, 2), valid_elaz[0])

        h = canvas.shape[0]
        w = canvas.shape[1]
        hor_h = (0.5 * h) - 1
        hor_w = (0.5 * w) - 1
        
        center_h = (h / 2) - 0.5
        center_w = (w / 2) - 0.5
        
        y = np.multiply(zenith_angle, np.cos(valid_elaz[1]))
        y = np.divide(y, np.pi)
        y = np.multiply(y, hor_h)
        y = np.subtract(center_h, y)
        
        x = np.multiply(zenith_angle, np.sin(valid_elaz[1]))
        x = np.divide(x, np.pi)
        x = np.multiply(x, hor_w)
        x = np.add(center_w, x)
                
        x_mask = np.logical_and(np.less_equal(x, w - 1), np.greater_equal(x, 0))
        y_mask = np.logical_and(np.less_equal(y, h - 1), np.greater_equal(y, 0))
        mask = np.logical_and(x_mask, y_mask)
        mask = np.vstack([mask, mask])
        xy = np.vstack([y, x])
        
        xy00 = np.vstack([y, x]).astype(int)
        xy01 = np.vstack([y, np.add(x, 1)]).astype(int)
        xy10 = np.vstack([np.add(y, 1), x]).astype(int)
        xy11 = np.vstack([np.add(y, 1), np.add(x, 1)]).astype(int)
        residual = xy - xy00

        try:
            canvas[*xy00] = np.clip(canvas[*xy00] + 0.25 * np.subtract(1, residual[0]) + 0.25 * np.subtract(1, residual[1]), 0.0, 1.0)
            canvas[*xy10] = np.clip(canvas[*xy10] + 0.25 * residual[0] + 0.25 * np.subtract(1, residual[1]), 0.0, 1.0)
            canvas[*xy01] = np.clip(canvas[*xy01] + 0.25 * np.subtract(1, residual[0]) + 0.25 * residual[1], 0.0, 1.0)
            canvas[*xy11] = np.clip(canvas[*xy11] + 0.25 * residual[0] + 0.25 * residual[1], 0.0, 1.0)
        except IndexError:
            print('Index Error while drawing:', np.max(x), np.max(y))
        return canvas
    
    def draw_relative(self, ref, canvas, radius=5, obstacles=None, hit_mask=False, scale='linear'):
        not_obstructed_mask = np.logical_not(obstacles.get_xy_obstructed_mask(self.xy)) if obstacles else True
        not_dones_mask = np.logical_not(self.dones)
        x_within_lim_mask = np.abs(self.xy[0,:]) < self.xlim
        y_within_lim_mask = np.abs(self.xy[1,:]) < self.ylim
        valid_masks = not_obstructed_mask & not_dones_mask & x_within_lim_mask & y_within_lim_mask
        if hit_mask:
            non_complete_hit_mask = self.hit_count < self.hit_required
            valid_masks = valid_masks & non_complete_hit_mask
        rel_pos = np.divide(np.subtract(self.xy[:, valid_masks], ref[:, np.newaxis]), np.radians(radius))
        hit_count = self.hit_count[valid_masks]
        in_scope_mask = np.max(np.abs(rel_pos), axis=0) <= 1
        valid_pos = rel_pos[:, in_scope_mask].reshape([2, -1])
        hit_count = hit_count[in_scope_mask]
            
        h = canvas.shape[0]
        w = canvas.shape[1]
        hor_h = (0.5 * h) - 1
        hor_w = (0.5 * w) - 1                                              
     
        if scale == 'fisheye':
            distance = np.sqrt(np.sum(np.square(valid_pos), axis=0))
            angle = np.arctan2(valid_pos[0,:], valid_pos[1,:])
            xs = np.sin(angle) * np.tanh(distance * 5)
            ys = np.cos(angle) * np.tanh(distance * 5)
            valid_pos = np.array([xs, ys])
            
        if scale == 'log':
            y = np.multiply(np.sign(y), np.sqrt(np.abs(y)))
            
        print(np.min(valid_pos), np.max(valid_pos))
        
        x = valid_pos[0]
        y = valid_pos[1]
        
        y = np.multiply(y, hor_h)
        y = np.subtract(np.divide(h - 1, 2), y)
           
        x = np.multiply(x, hor_w)
        x = np.add(np.divide(w - 1, 2), x)        

        xy = np.vstack([y, x])
        
        print(np.min(xy), np.max(xy))

        xy00 = np.vstack([y, x]).astype(int)
        xy01 = np.vstack([y, np.add(x, 1)]).astype(int)
        xy10 = np.vstack([np.add(y, 1), x]).astype(int)
        xy11 = np.vstack([np.add(y, 1), np.add(x, 1)]).astype(int)
        residual = xy - xy00
        
        pixel_intensity = 0.25 if not hit_mask else 0.1 + (self.hit_required - hit_count) * 0.025

        try:
            canvas[*xy00] = np.clip(canvas[*xy00] + pixel_intensity * np.subtract(1, residual[0]) + pixel_intensity * np.subtract(1, residual[1]), 0.0, 1.0)
            canvas[*xy10] = np.clip(canvas[*xy10] + pixel_intensity * residual[0] + pixel_intensity * np.subtract(1, residual[1]), 0.0, 1.0)
            canvas[*xy01] = np.clip(canvas[*xy01] + pixel_intensity * np.subtract(1, residual[0]) + pixel_intensity * residual[1], 0.0, 1.0)
            canvas[*xy11] = np.clip(canvas[*xy11] + pixel_intensity * residual[0] + pixel_intensity * residual[1], 0.0, 1.0)
        except IndexError:
            print(np.max(x), np.max(y), np.max(valid_pos))
        return canvas

    def reset(self):
        self.time = self._time
        self.cur_step=0
        self.dones = np.full_like(self.spos, False, dtype = bool)
        self.hit_count = np.zeros_like(self.spos, dtype=np.uint8)
        return self


register(
    # unique identifier for the env `name-version`
    id="SingleDish-v4",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point=SingleDish,
)

if __name__ == '__main__':
    import gymnasium as gym
    
    def chase(env):
        """ Proportional Controller. Move toward the closest above-horizon targets with speed proportional to the distance"""
        not_dones = np.logical_not(env.targets.dones)
        valid_targets = np.all(np.less(np.abs(env.targets.xy), np.radians([[80], [80]])), axis=0) 
        targets = env.targets.xy[:, valid_targets & not_dones]
        rel_pos = np.subtract(targets, env.antenna.xy[:, np.newaxis])
        distance = np.sqrt(np.sum(np.square(rel_pos), axis=0))
        try:
            closest_target_id = np.argmin(distance)
            closest_target = rel_pos[:,closest_target_id]
            closest_distance = distance[closest_target_id]
        except ValueError:
            closest_target = [0, 0]
        action = np.sign(closest_target) * np.square(np.clip([-0.05, 0] + np.multiply((closest_target) , [384, 196]), -1.0, 1.0)).astype(np.float32)
        return action

    env = SingleDish(rank=0, show=True)
    _obs, _info = env.reset()
    
    for _ in range(1000):
        env.step(chase(env))
        img = env.canvas[:,:,:3]
        bore = env.canvas[:,:,4]
        env.render(show=True)
        # img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_NEAREST) #Uncomment for bigger zoom
        # cv2.imshow('obs', img)
        cv2.waitKey(1)
