import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import random

class SimpleRocketEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, render_mode='human'):
        super().__init__()
        self.render_mode = render_mode
        self.dt, self.m, self.g = 0.025, 3.0, 9.81
        self.F_main, self.F_side = 400.0, 200.0

        self.screen_w, self.screen_h = 960, 480
        self.floor_y = 10.0

        self.w, self.h = 30.0, 60.0
        self.I = (1/12) * self.m * (self.w**2 + self.h**2)

        self.b_linear  = 0.1
        self.b_angular = 0.15

        tx_init = int(random.random() * 500)
        self.target_pos = np.array([300.0 + tx_init, 40.0], np.float32)
        self.target_w, self.target_h = 240.0, 80.0
        self.target_min_x = 300.0
        self.target_max_x = 800.0
        self.target_vx    = 50.0

        self.launch_pad_pos = np.array([100.0, 50.0], dtype=np.float32)
        self.pad_w, self.pad_h = 40.0, 40.0

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(11,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        self.norm_pos = np.array([self.screen_w, self.screen_h], dtype=np.float32)
        self.norm_vel = 120.0
        self.norm_omega = 20.0
        self.norm_dist = 800.0

        self.render_mode = render_mode
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
            pygame.display.set_caption("Simple Rocket Env")
            self.clock = pygame.time.Clock()
            self.font  = pygame.font.SysFont("Arial", 16)

            base = os.path.dirname(__file__)
            load = lambda fn: pygame.image.load(os.path.join(base, fn)).convert_alpha()
            self.rocket_img = pygame.transform.scale(load("rocket.png"), (int(self.w), int(self.h)))
            self.target_img = pygame.transform.scale(load("target.png"),
                                                     (int(self.target_w), int(self.target_h)))

        self.state = None
        self.last_action = 0
        self.max_steps = 800
        self.step_count = 0
        
        self.prev_dist = None
        self.prev_angle_error = None
        self.prev_velocity_mag = None
        self.prev_omega = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        x, y = 100.0, 50.0
        vx, vy = 0.0, 60.0
        theta = 0.0
        omega = 0.0
        
        tx_offset = random.random() * 500
        self.target_pos[0] = 300.0 + tx_offset
        self.target_vx = abs(self.target_vx) * random.choice([1, -1])
        
        dx = x - self.target_pos[0]
        dy = y - self.target_pos[1]
        dist = np.sqrt(dx**2 + dy**2)
        speed = np.sqrt(vx**2 + vy**2)
        
        self.prev_dist = dist
        self.prev_angle_error = abs(theta)
        self.prev_velocity_mag = speed
        self.prev_omega = omega
        
        # State Normalization
        state = np.array([
            x / self.norm_pos[0] - 0.5,
            y / self.norm_pos[1] - 0.5,
            vx / self.norm_vel,
            vy / self.norm_vel,
            np.sin(theta),
            np.cos(theta),
            omega / self.norm_omega,
            dx / self.norm_pos[0],
            dy / self.norm_pos[1],
            dist / self.norm_dist,
            speed / self.norm_vel,
        ], dtype=np.float32)
        
        self.state = state
        self.last_action = 0
        self.step_count = 0
        return self.state, {}

    def _normalize_state(self, x, y, vx, vy, theta, omega, dx, dy, dist, speed):
        return np.array([
            x / self.norm_pos[0] - 0.5,
            y / self.norm_pos[1] - 0.5,
            vx / self.norm_vel,
            vy / self.norm_vel,
            np.sin(theta),
            np.cos(theta),
            omega / self.norm_omega,
            dx / self.norm_pos[0],
            dy / self.norm_pos[1],
            dist / self.norm_dist,
            speed / self.norm_vel,
        ], dtype=np.float32)

    def step(self, action):
        self.step_count += 1

        state_denorm = self.state.copy()
        x = (state_denorm[0] + 0.5) * self.norm_pos[0]
        y = (state_denorm[1] + 0.5) * self.norm_pos[1]
        vx = state_denorm[2] * self.norm_vel
        vy = state_denorm[3] * self.norm_vel
        sin_theta, cos_theta = state_denorm[4], state_denorm[5]
        omega = state_denorm[6] * self.norm_omega
        
        theta = np.arctan2(sin_theta, cos_theta)
        Fx = Fy = torque = 0.0

        if action == 1:
            Fx = self.F_main * np.sin(theta)
            Fy = self.F_main * np.cos(theta)
        elif action in (2, 3):
            s = 1 if action == 2 else -1
            torque = s * self.F_side * (self.h/2)
            Fx += -s * self.F_side * np.cos(theta)
            Fy += -s * self.F_side * np.sin(theta)

        Fx += -self.b_linear * vx
        Fy += -self.b_linear * vy
        torque += -self.b_angular * omega

        vy += ((Fy/self.m) - self.g) * self.dt
        vx += (Fx/self.m) * self.dt
        y  += vy * self.dt
        x  += vx * self.dt

        omega += (torque / self.I) * self.dt
        theta += omega * self.dt
        sin_theta, cos_theta = np.sin(theta), np.cos(theta)

        theta = (theta + np.pi) % (2*np.pi) - np.pi
        vx = np.clip(vx, -120, 120)
        vy = np.clip(vy, -120, 120)
        omega = np.clip(omega, -20, 20)

        tvx = random.random() * self.target_vx
        tx = self.target_pos[0] + tvx * self.dt
        if tx < self.target_min_x:
            tx = self.target_min_x
            self.target_vx = -self.target_vx
        elif tx > self.target_max_x:
            tx = self.target_max_x
            self.target_vx = -self.target_vx
        self.target_pos[0] = tx

        dx = x - self.target_pos[0]
        dy = y - self.target_pos[1]
        dist = np.sqrt(dx**2 + dy**2)
        speed = np.sqrt(vx**2 + vy**2)

        landed = y <= self.floor_y
        if landed:
            y, vy = self.floor_y, 0.0

        tx, ty = self.target_pos
        half_w, half_h = self.target_w/2, self.target_h/2
        target_collide = (tx-half_w <= x <= tx+half_w) and (ty-half_h <= y <= ty+half_h)

        terminated = False
        reward = 0.0

        angle_error = abs(theta)

        if target_collide:
            angle_quality = max(0, 1.0 - (angle_error / (np.pi/6)))
            angle_quality = angle_quality ** 3
            
            ideal_landing_speed = 10.0
            if speed <= ideal_landing_speed:
                speed_quality = 1.0
            else:
                speed_quality = max(0, 1.0 - (speed - ideal_landing_speed) / 30.0)
            speed_quality = speed_quality ** 2
            
            vertical_ratio = abs(vy) / (speed + 1e-6)
            vertical_quality = vertical_ratio ** 2
            
            horizontal_quality = max(0, 1.0 - abs(vx) / 10.0)
            horizontal_quality = horizontal_quality ** 2
            
            omega_quality = max(0, 1.0 - abs(omega) / 3.0)
            omega_quality = omega_quality ** 2
            
            landing_precision = max(0, 1.0 - abs(dx) / (self.target_w / 2))
            
            base_landing_reward = 250.0
            
            quality_bonus = 550.0 * (
                0.50 * angle_quality +
                0.20 * speed_quality +
                0.10 * vertical_quality +
                0.08 * horizontal_quality +
                0.07 * omega_quality +
                0.05 * landing_precision
            )
            
            landing_reward = base_landing_reward + quality_bonus
            
            if angle_error < 0.087 and speed < 12 and abs(omega) < 1.0:
                landing_reward += 100.0
            
            reward = landing_reward
            terminated = True
            
        elif x < 0 or x > self.screen_w or y > self.screen_h:
            reward = -100.0
            terminated = True
            
        elif landed:
            x_dist_to_target = abs(x - self.target_pos[0])
            proximity_bonus = max(0, 20.0 * (1.0 - x_dist_to_target / 200.0))
            reward = -50.0 + proximity_bonus
            terminated = True

        if not terminated:
            dist_improvement = self.prev_dist - dist
            reward += 8.0 * dist_improvement
            
            proximity_factor = np.exp(-dist / 80.0)
            reward += 12.0 * proximity_factor
            
            upright_reward = 4.0 * (1.0 - angle_error / np.pi)
            if dist < 200:
                upright_reward *= (1.0 + 3.0 * proximity_factor)
                
                angle_improvement = self.prev_angle_error - angle_error
                reward += 5.0 * angle_improvement
                
                omega_improvement = self.prev_omega - abs(omega)
                reward += 8.0 * omega_improvement
            
            reward += upright_reward
            
            target_speed = max(5.0, dist / 15.0)
            
            if dist < 180:
                if speed > target_speed:
                    speed_penalty = -1.5 * (speed - target_speed) / 10.0
                    reward += speed_penalty
                    
                    velocity_improvement = self.prev_velocity_mag - speed
                    reward += 2.5 * velocity_improvement
                else:
                    reward += 1.5
            
            if dist < 150 and speed > 1.0:
                vertical_component = abs(vy) / (speed + 1e-6)
                if vertical_component > 0.7:
                    reward += 2.0 * vertical_component
            
            if dist < 120:
                stability_reward = 0.0
                
                if abs(omega) < 2.0:
                    stability_reward += 6.0 * (1.0 - abs(omega) / 2.0)
                else:
                    stability_reward += -5.0 * (abs(omega) - 2.0)
                
                if dy > 0 and dy < 100:
                    altitude_bonus = 3.0 * (1.0 - abs(dy - 40) / 40.0)
                    stability_reward += altitude_bonus
                
                if abs(dx) < 50:
                    alignment_bonus = 2.0 * (1.0 - abs(dx) / 50.0)
                    stability_reward += alignment_bonus
                
                reward += stability_reward * proximity_factor
            
            if dist < 200:
                x_alignment = 1.0 - min(1.0, abs(dx) / 120.0)
                reward += 2.5 * x_alignment * proximity_factor
                
                if abs(dx) < 30 and dy > 10:
                    reward += 3.0 * proximity_factor
            
            omega_penalty = -3.0 * (abs(omega) / self.norm_omega)
            if dist < 150:
                omega_penalty *= (1.0 + 6.0 * proximity_factor)
            reward += omega_penalty
            
            if dist < 180:
                if y < 50:
                    reward += -3.0
                elif 60 <= y <= 140:
                    reward += 0.8
                elif y > 200:
                    reward += -0.5
            
            if dist < 100:
                if action == 0:
                    if speed < 15 and angle_error < 0.3 and abs(dx) < 40 and abs(omega) < 1.0:
                        reward += 0.5
                elif action in (2, 3):
                    if abs(dx) < 15 and abs(omega) < 1.0:
                        reward += -3.0
                    if abs(omega) > 3.0:
                        reward += -2.0
            
            reward += -0.02

        self.prev_dist = dist
        self.prev_angle_error = angle_error
        self.prev_velocity_mag = speed
        self.prev_omega = abs(omega)

        truncated = self.step_count >= self.max_steps

        self.state = self._normalize_state(x, y, vx, vy, theta, omega, dx, dy, dist, speed)
        self.last_action = action

        return self.state, reward, terminated, truncated, {}

    def render(self, mode='human'):
        if mode != 'human' or self.render_mode != 'human':
            return
            
        x = (self.state[0] + 0.5) * self.norm_pos[0]
        y = (self.state[1] + 0.5) * self.norm_pos[1]
        vx = self.state[2] * self.norm_vel
        vy = self.state[3] * self.norm_vel
        sin_theta, cos_theta = self.state[4], self.state[5]
        omega = self.state[6] * self.norm_omega
        
        theta = np.arctan2(sin_theta, cos_theta)
        y_flip = self.screen_h - y
        
        self.screen.fill((0,0,0))

        deg = -np.degrees(theta)
        rocket = pygame.transform.rotate(self.rocket_img, deg)
        rect = rocket.get_rect(center=(x, y_flip))
        self.screen.blit(rocket, rect.topleft)

        flame = []
        if self.last_action == 1:
            flame = [(-5,self.h/2), (5,self.h/2), (0,self.h/2+20)]
        elif self.last_action in (2,3):
            s = -1 if self.last_action==2 else 1
            flame = [(s*(self.w/2+5),0), (s*self.w/2,-5), (s*self.w/2,5)]
            
        if flame:
            c, s_ = np.cos(theta), np.sin(theta)
            pts = [(int(x + c*px - s_*py), int(y_flip + s_*px + c*py))
                   for px,py in flame]
            pygame.draw.polygon(self.screen, (255,140,70), pts)

        fy = self.screen_h - self.floor_y
        pygame.draw.line(self.screen,(200,200,200),(0,fy),(self.screen_w,fy),2)

        tx, ty = self.target_pos
        tx_screen, ty_screen = tx-self.target_w/2, (self.screen_h-ty)-self.target_h/2
        self.screen.blit(self.target_img, (int(tx_screen), int(ty_screen)))

        px, py = self.launch_pad_pos
        screen_py = self.screen_h - py
        pad_rect = pygame.Rect(
            int(px - self.pad_w/2), int(screen_py),
            int(self.pad_w), int(self.pad_h)
        )
        pygame.draw.rect(self.screen, (80, 120, 80), pad_rect)

        dist = self.state[9] * self.norm_dist
        speed = self.state[10] * self.norm_vel
        
        angle_deg = abs(np.degrees(theta))
        if angle_deg < 5:
            orient_status, orient_color = "PERFECT", (0, 255, 0)
        elif angle_deg < 10:
            orient_status, orient_color = "GOOD", (100, 255, 100)
        elif angle_deg < 20:
            orient_status, orient_color = "OK", (255, 255, 0)
        elif angle_deg < 45:
            orient_status, orient_color = "TILTED", (255, 150, 0)
        else:
            orient_status, orient_color = "BAD", (255, 0, 0)
        
        info1 = f"Pos: ({x:.1f}, {y:.1f}) | Vel: ({vx:.1f}, {vy:.1f}) | Speed: {speed:.1f}"
        self.screen.blit(self.font.render(info1, True, (255,255,255)), (10,10))
        
        info2 = f"Angle: {np.degrees(theta):.1f}° | ω: {omega:.2f} | Status: {orient_status}"
        self.screen.blit(self.font.render(info2, True, orient_color), (10,30))
        
        info3 = f"Target Dist: {dist:.1f} | Target: ({self.target_pos[0]:.0f}, {self.target_pos[1]:.0f})"
        self.screen.blit(self.font.render(info3, True, (255,255,255)), (10,50))
        
        arrow_length = 40
        arrow_x = self.screen_w - 60
        arrow_y = 60
        
        end_x = arrow_x + arrow_length * np.sin(theta)
        end_y = arrow_y - arrow_length * np.cos(theta)
        
        pygame.draw.circle(self.screen, (50, 50, 50), (arrow_x, arrow_y), 45)
        pygame.draw.circle(self.screen, (100, 100, 100), (arrow_x, arrow_y), 42, 2)
        
        pygame.draw.line(self.screen, orient_color, (arrow_x, arrow_y), 
                         (int(end_x), int(end_y)), 4)
        pygame.draw.circle(self.screen, orient_color, (int(end_x), int(end_y)), 6)
        
        pygame.draw.line(self.screen, (150, 150, 150), 
                         (arrow_x, arrow_y), (arrow_x, arrow_y - 35), 1)
        
        label = self.font.render("UP", True, (200, 200, 200))
        self.screen.blit(label, (arrow_x - 15, arrow_y + 50))

        pygame.display.flip()
        self.clock.tick(50)

    def close(self):
        if self.render_mode == 'human':
            pygame.quit()