import math
import random
import pyray as rl
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.widgets import Widget

# ── Map ──────────────────────────────────────────────────────────────────────
# Wall types: 0=empty, 1=stone, 2=brick, 3=tech, 4=toxic, 5=door
MAP_W, MAP_H = 24, 24
WORLD_MAP = [
  [1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3],
  [1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
  [1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
  [1,0,0,0,0,0,5,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,3],
  [1,0,0,0,0,0,1,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,3],
  [1,0,0,0,0,0,1,0,0,0,0,0,2,0,0,0,0,0,3,3,3,0,0,3],
  [1,1,1,5,1,1,1,0,0,0,0,0,2,2,5,2,2,0,3,0,0,0,0,3],
  [4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,0,0,0,3],
  [4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,0,0,0,3],
  [4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
  [4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
  [4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,0,0,0,3],
  [4,4,4,0,4,4,4,0,0,0,1,1,1,1,1,1,1,0,3,3,5,3,3,3],
  [4,0,0,0,0,0,4,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,3],
  [4,0,0,0,0,0,4,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,3],
  [4,0,0,0,0,0,4,0,0,0,5,0,0,0,0,0,1,0,0,0,0,0,0,3],
  [4,0,0,0,0,0,4,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,3],
  [4,0,0,0,0,0,4,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,3],
  [4,4,4,5,4,4,4,0,0,0,1,1,1,5,1,1,1,0,0,0,0,0,0,3],
  [2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,4,0,0,3],
  [2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,3],
  [2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,3],
  [2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
  [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3],
]

# Wall colors: (lit_side, dark_side) for N/S vs E/W faces
WALL_COLORS = {
  1: (rl.Color(160, 160, 160, 255), rl.Color(110, 110, 110, 255)),  # Stone
  2: (rl.Color(180, 70, 50, 255),   rl.Color(130, 50, 35, 255)),    # Brick
  3: (rl.Color(60, 100, 170, 255),  rl.Color(40, 70, 125, 255)),    # Tech
  4: (rl.Color(50, 155, 50, 255),   rl.Color(35, 110, 35, 255)),    # Toxic
  5: (rl.Color(160, 130, 60, 255),  rl.Color(120, 95, 40, 255)),    # Door
}

CEIL_COLOR = rl.Color(40, 40, 50, 255)
FLOOR_COLOR = rl.Color(70, 60, 50, 255)

FOV = math.pi / 3
MAX_DEPTH = 24.0
MOVE_SPEED = 3.5
TURN_SPEED = 2.8
COLLISION_RADIUS = 0.25
SHOOT_COOLDOWN = 0.35
ENEMY_ATTACK_RANGE = 1.2
ENEMY_DAMAGE = 12
ENEMY_SPEED = 1.8

NUM_RAYS = 720  # cast this many rays, each draws a ~3px wide column


# ── Entities ─────────────────────────────────────────────────────────────────

class Enemy:
  __slots__ = ['x', 'y', 'hp', 'alive', 'speed', 'hurt_timer', 'attack_cd', 'kind']

  def __init__(self, x: float, y: float, kind: int = 0):
    self.x = x
    self.y = y
    self.kind = kind  # 0=imp, 1=demon
    self.hp = 3 if kind == 0 else 6
    self.alive = True
    self.speed = ENEMY_SPEED if kind == 0 else ENEMY_SPEED * 0.7
    self.hurt_timer = 0.0
    self.attack_cd = 0.0


class Pickup:
  __slots__ = ['x', 'y', 'kind', 'active']

  def __init__(self, x: float, y: float, kind: str):
    self.x = x
    self.y = y
    self.kind = kind  # 'health' or 'ammo'
    self.active = True


# ── Game Widget ──────────────────────────────────────────────────────────────

class DoomGame(Widget):
  def __init__(self):
    super().__init__()
    self._init_game()

  def _init_game(self):
    # Player
    self.px, self.py = 2.5, 2.5
    self.pa = math.pi / 4  # face SE
    self.health = 100
    self.ammo = 30
    self.kills = 0
    self.score = 0

    # Timers & effects
    self.shoot_cd = 0.0
    self.muzzle_flash = 0.0
    self.damage_flash = 0.0
    self.hit_marker = 0.0
    self.bob_phase = 0.0

    # State
    self.game_over = False
    self.game_won = False

    # Enemies
    self.enemies = [
      Enemy(10.5, 2.5, 0),
      Enemy(20.5, 4.5, 0),
      Enemy(4.5, 9.5, 0),
      Enemy(14.5, 9.5, 1),   # demon
      Enemy(2.5, 15.5, 0),
      Enemy(13.5, 15.5, 0),
      Enemy(21.5, 14.5, 0),
      Enemy(20.5, 20.5, 1),  # demon
      Enemy(8.5, 20.5, 0),
      Enemy(13.5, 21.5, 0),
    ]

    # Pickups
    self.pickups = [
      Pickup(5.5, 8.5, 'health'),
      Pickup(2.5, 20.5, 'ammo'),
      Pickup(9.5, 15.5, 'health'),
      Pickup(20.5, 9.5, 'ammo'),
      Pickup(14.5, 2.5, 'ammo'),
      Pickup(8.5, 14.5, 'health'),
    ]

    # Z-buffer for sprite rendering
    self.z_buffer = []

  # ── Input ────────────────────────────────────────────────────────────────

  def _handle_input(self, dt: float):
    cos_a = math.cos(self.pa)
    sin_a = math.sin(self.pa)

    # Turn
    if rl.is_key_down(rl.KeyboardKey.KEY_LEFT):
      self.pa -= TURN_SPEED * dt
    if rl.is_key_down(rl.KeyboardKey.KEY_RIGHT):
      self.pa += TURN_SPEED * dt
    # Also allow Q/E for turning
    if rl.is_key_down(rl.KeyboardKey.KEY_Q):
      self.pa -= TURN_SPEED * dt
    if rl.is_key_down(rl.KeyboardKey.KEY_E):
      self.pa += TURN_SPEED * dt

    # Movement
    dx, dy = 0.0, 0.0
    if rl.is_key_down(rl.KeyboardKey.KEY_W) or rl.is_key_down(rl.KeyboardKey.KEY_UP):
      dx += cos_a * MOVE_SPEED * dt
      dy += sin_a * MOVE_SPEED * dt
    if rl.is_key_down(rl.KeyboardKey.KEY_S) or rl.is_key_down(rl.KeyboardKey.KEY_DOWN):
      dx -= cos_a * MOVE_SPEED * dt
      dy -= sin_a * MOVE_SPEED * dt
    if rl.is_key_down(rl.KeyboardKey.KEY_A):
      dx += sin_a * MOVE_SPEED * dt
      dy -= cos_a * MOVE_SPEED * dt
    if rl.is_key_down(rl.KeyboardKey.KEY_D):
      dx -= sin_a * MOVE_SPEED * dt
      dy += cos_a * MOVE_SPEED * dt

    # Apply movement with collision
    self._move_player(dx, dy)

    # Walk bob
    if abs(dx) > 0.001 or abs(dy) > 0.001:
      self.bob_phase += dt * 8.0
    else:
      self.bob_phase *= 0.9

    # Shoot
    if (rl.is_key_down(rl.KeyboardKey.KEY_SPACE) or rl.is_key_down(rl.KeyboardKey.KEY_LEFT_CONTROL)) and self.shoot_cd <= 0 and self.ammo > 0:
      self._shoot()

  def _move_player(self, dx: float, dy: float):
    r = COLLISION_RADIUS
    # X movement
    nx = self.px + dx
    if WORLD_MAP[int(self.py)][int(nx + r)] == 0 and WORLD_MAP[int(self.py)][int(nx - r)] == 0:
      self.px = nx
    # Y movement
    ny = self.py + dy
    if WORLD_MAP[int(ny + r)][int(self.px)] == 0 and WORLD_MAP[int(ny - r)][int(self.px)] == 0:
      self.py = ny

  def _shoot(self):
    self.ammo -= 1
    self.shoot_cd = SHOOT_COOLDOWN
    self.muzzle_flash = 0.15

    # Raycast from player center to find hit
    best_dist = MAX_DEPTH
    best_enemy = None

    for e in self.enemies:
      if not e.alive:
        continue
      edx = e.x - self.px
      edy = e.y - self.py
      dist = math.sqrt(edx * edx + edy * edy)
      if dist < 0.1:
        continue

      # Check angle to enemy
      angle_to_enemy = math.atan2(edy, edx)
      angle_diff = angle_to_enemy - self.pa
      # Normalize
      while angle_diff > math.pi:
        angle_diff -= 2 * math.pi
      while angle_diff < -math.pi:
        angle_diff += 2 * math.pi

      # Enemy width in angle space (bigger when closer)
      enemy_half_width = math.atan2(0.4, dist)

      if abs(angle_diff) < enemy_half_width and dist < best_dist:
        # Check if wall is blocking
        wall_dist = self._cast_single_ray(self.pa)
        if dist < wall_dist:
          best_dist = dist
          best_enemy = e

    if best_enemy is not None:
      best_enemy.hp -= 1
      best_enemy.hurt_timer = 0.2
      self.hit_marker = 0.2
      if best_enemy.hp <= 0:
        best_enemy.alive = False
        self.kills += 1
        self.score += 100 if best_enemy.kind == 0 else 250

  def _cast_single_ray(self, angle: float) -> float:
    """Cast a single ray and return wall distance."""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    # DDA setup
    map_x, map_y = int(self.px), int(self.py)

    delta_x = abs(1.0 / cos_a) if cos_a != 0 else 1e30
    delta_y = abs(1.0 / sin_a) if sin_a != 0 else 1e30

    if cos_a < 0:
      step_x = -1
      side_x = (self.px - map_x) * delta_x
    else:
      step_x = 1
      side_x = (map_x + 1.0 - self.px) * delta_x

    if sin_a < 0:
      step_y = -1
      side_y = (self.py - map_y) * delta_y
    else:
      step_y = 1
      side_y = (map_y + 1.0 - self.py) * delta_y

    for _ in range(64):
      if side_x < side_y:
        side_x += delta_x
        map_x += step_x
        side = 0
      else:
        side_y += delta_y
        map_y += step_y
        side = 1

      if 0 <= map_x < MAP_W and 0 <= map_y < MAP_H:
        if WORLD_MAP[map_y][map_x] > 0:
          if side == 0:
            return side_x - delta_x
          else:
            return side_y - delta_y

    return MAX_DEPTH

  # ── Enemy AI ─────────────────────────────────────────────────────────────

  def _update_enemies(self, dt: float):
    for e in self.enemies:
      if not e.alive:
        continue

      e.hurt_timer = max(0.0, e.hurt_timer - dt)
      e.attack_cd = max(0.0, e.attack_cd - dt)

      # Distance to player
      edx = self.px - e.x
      edy = self.py - e.y
      dist = math.sqrt(edx * edx + edy * edy)

      if dist < 0.5:
        dist = 0.5

      # Simple chase: move toward player if within detection range
      if dist < 12.0:
        # Check line of sight
        if self._has_line_of_sight(e.x, e.y, self.px, self.py):
          # Move toward player
          move_x = (edx / dist) * e.speed * dt
          move_y = (edy / dist) * e.speed * dt

          nx = e.x + move_x
          ny = e.y + move_y

          if 0 <= int(ny) < MAP_H and 0 <= int(nx) < MAP_W and WORLD_MAP[int(e.y)][int(nx)] == 0:
            e.x = nx
          if 0 <= int(ny) < MAP_H and 0 <= int(nx) < MAP_W and WORLD_MAP[int(ny)][int(e.x)] == 0:
            e.y = ny

          # Attack if close
          if dist < ENEMY_ATTACK_RANGE and e.attack_cd <= 0:
            self.health -= ENEMY_DAMAGE
            self.damage_flash = 0.3
            e.attack_cd = 1.0 + random.random() * 0.5
            if self.health <= 0:
              self.health = 0
              self.game_over = True

  def _has_line_of_sight(self, x0: float, y0: float, x1: float, y1: float) -> bool:
    """Simple raycast line-of-sight check."""
    dx = x1 - x0
    dy = y1 - y0
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 0.01:
      return True

    steps = int(dist * 4)
    for i in range(steps):
      t = i / max(steps, 1)
      cx = x0 + dx * t
      cy = y0 + dy * t
      mx, my = int(cx), int(cy)
      if 0 <= mx < MAP_W and 0 <= my < MAP_H:
        if WORLD_MAP[my][mx] > 0:
          return False
    return True

  # ── Pickups ──────────────────────────────────────────────────────────────

  def _update_pickups(self):
    for p in self.pickups:
      if not p.active:
        continue
      dx = self.px - p.x
      dy = self.py - p.y
      if dx * dx + dy * dy < 0.5:
        p.active = False
        if p.kind == 'health':
          self.health = min(100, self.health + 25)
          self.score += 50
        else:
          self.ammo += 15
          self.score += 25

  # ── Timers ───────────────────────────────────────────────────────────────

  def _update_timers(self, dt: float):
    self.shoot_cd = max(0.0, self.shoot_cd - dt)
    self.muzzle_flash = max(0.0, self.muzzle_flash - dt)
    self.damage_flash = max(0.0, self.damage_flash - dt)
    self.hit_marker = max(0.0, self.hit_marker - dt)

  # ── Rendering ────────────────────────────────────────────────────────────

  def _draw_ceiling_floor(self, rect: rl.Rectangle):
    hw = int(rect.height / 2)
    # Ceiling
    rl.draw_rectangle(int(rect.x), int(rect.y), int(rect.width), hw, CEIL_COLOR)
    # Floor - gradient from dark to lighter
    rl.draw_rectangle_gradient_v(int(rect.x), int(rect.y) + hw, int(rect.width), hw,
                                  rl.Color(50, 45, 38, 255), FLOOR_COLOR)

  def _cast_rays(self, rect: rl.Rectangle):
    """Cast all rays and render walls."""
    vw = int(rect.width)
    vh = int(rect.height)
    vx = int(rect.x)
    vy = int(rect.y)
    col_w = max(vw / NUM_RAYS, 1.0)

    self.z_buffer = [MAX_DEPTH] * NUM_RAYS

    for i in range(NUM_RAYS):
      # Ray angle
      ray_angle = self.pa - FOV / 2 + (i / NUM_RAYS) * FOV
      cos_a = math.cos(ray_angle)
      sin_a = math.sin(ray_angle)

      # DDA algorithm
      map_x = int(self.px)
      map_y = int(self.py)

      delta_x = abs(1.0 / cos_a) if cos_a != 0 else 1e30
      delta_y = abs(1.0 / sin_a) if sin_a != 0 else 1e30

      if cos_a < 0:
        step_x = -1
        side_x = (self.px - map_x) * delta_x
      else:
        step_x = 1
        side_x = (map_x + 1.0 - self.px) * delta_x

      if sin_a < 0:
        step_y = -1
        side_y = (self.py - map_y) * delta_y
      else:
        step_y = 1
        side_y = (map_y + 1.0 - self.py) * delta_y

      wall_type = 0
      side = 0

      for _ in range(64):
        if side_x < side_y:
          side_x += delta_x
          map_x += step_x
          side = 0
        else:
          side_y += delta_y
          map_y += step_y
          side = 1

        if 0 <= map_x < MAP_W and 0 <= map_y < MAP_H:
          if WORLD_MAP[map_y][map_x] > 0:
            wall_type = WORLD_MAP[map_y][map_x]
            break

      # Perpendicular distance (fixes fisheye)
      if side == 0:
        perp_dist = side_x - delta_x
      else:
        perp_dist = side_y - delta_y

      # Fix for cos correction
      angle_diff = ray_angle - self.pa
      perp_dist *= math.cos(angle_diff)

      if perp_dist < 0.01:
        perp_dist = 0.01

      self.z_buffer[i] = perp_dist

      # Wall height
      wall_h = vh / perp_dist
      if wall_h > vh * 3:
        wall_h = vh * 3

      # Draw position
      draw_start = int(vh / 2 - wall_h / 2)
      draw_end = int(vh / 2 + wall_h / 2)

      # Get color
      if wall_type in WALL_COLORS:
        color = WALL_COLORS[wall_type][side]
      else:
        color = rl.Color(200, 200, 200, 255)

      # Distance fog
      fog = min(perp_dist / MAX_DEPTH, 0.8)
      r = int(color.r * (1 - fog))
      g = int(color.g * (1 - fog))
      b = int(color.b * (1 - fog))
      color = rl.Color(r, g, b, 255)

      cx = int(vx + i * col_w)
      cw = int(col_w) + 1
      rl.draw_rectangle(cx, vy + draw_start, cw, draw_end - draw_start, color)

  # ── Sprites ──────────────────────────────────────────────────────────────

  def _draw_sprites(self, rect: rl.Rectangle):
    """Draw enemies and pickups as sprites."""
    vw = int(rect.width)
    vh = int(rect.height)
    vx = int(rect.x)
    vy = int(rect.y)
    col_w = max(vw / NUM_RAYS, 1.0)

    sprites = []

    # Add enemies
    for e in self.enemies:
      if not e.alive:
        continue
      dx = e.x - self.px
      dy = e.y - self.py
      dist = math.sqrt(dx * dx + dy * dy)
      angle = math.atan2(dy, dx)
      sprites.append((dist, angle, 'enemy', e))

    # Add pickups
    for p in self.pickups:
      if not p.active:
        continue
      dx = p.x - self.px
      dy = p.y - self.py
      dist = math.sqrt(dx * dx + dy * dy)
      angle = math.atan2(dy, dx)
      sprites.append((dist, angle, 'pickup', p))

    # Sort far to near
    sprites.sort(key=lambda s: -s[0])

    for dist, angle, kind, obj in sprites:
      if dist < 0.3:
        continue

      # Angle relative to player view
      angle_diff = angle - self.pa
      while angle_diff > math.pi:
        angle_diff -= 2 * math.pi
      while angle_diff < -math.pi:
        angle_diff += 2 * math.pi

      # Skip if outside FOV (with some margin)
      if abs(angle_diff) > FOV / 2 + 0.15:
        continue

      # Screen X position
      screen_x = (0.5 + angle_diff / FOV) * vw
      sprite_h = vh / dist
      sprite_w = sprite_h * 0.6

      sx = int(vx + screen_x - sprite_w / 2)
      sy = int(vy + vh / 2 - sprite_h / 2)
      sw = int(sprite_w)
      sh = int(sprite_h)

      # Clip to screen
      if sx + sw < vx or sx > vx + vw:
        continue

      # Z-buffer clipping (column by column)
      for col in range(max(0, sx - vx), min(vw, sx - vx + sw)):
        ray_idx = int(col / col_w)
        if ray_idx < 0 or ray_idx >= NUM_RAYS:
          continue
        if dist >= self.z_buffer[ray_idx]:
          continue

        # This column is visible - draw a 1px wide strip
        strip_x = vx + col
        # Calculate how far into the sprite this column is
        frac = (col - (sx - vx)) / max(sw, 1)

        if kind == 'enemy':
          e = obj
          # Body color
          if e.hurt_timer > 0:
            base_color = rl.Color(255, 255, 255, 255)
          elif e.kind == 0:  # imp
            base_color = rl.Color(200, 80, 60, 255)
          else:  # demon
            base_color = rl.Color(160, 60, 160, 255)

          # Distance fog
          fog = min(dist / MAX_DEPTH, 0.8)
          cr = int(base_color.r * (1 - fog))
          cg = int(base_color.g * (1 - fog))
          cb = int(base_color.b * (1 - fog))

          # Draw body
          body_top = sy + int(sh * 0.1)
          body_bot = sy + sh
          rl.draw_rectangle(strip_x, body_top, 1, body_bot - body_top, rl.Color(cr, cg, cb, 255))

          # Eyes (in upper portion)
          eye_y = sy + int(sh * 0.2)
          eye_h = max(int(sh * 0.08), 2)
          if 0.25 < frac < 0.38 or 0.62 < frac < 0.75:
            rl.draw_rectangle(strip_x, eye_y, 1, eye_h, rl.Color(255, 255, 0, 255))

          # Horns for demons
          if e.kind == 1:
            if 0.2 < frac < 0.3 or 0.7 < frac < 0.8:
              horn_y = sy
              horn_h = int(sh * 0.12)
              rl.draw_rectangle(strip_x, horn_y, 1, horn_h, rl.Color(cr, cg, cb, 255))

        elif kind == 'pickup':
          p = obj
          # Floating bobbing effect
          bob_offset = int(math.sin(rl.get_time() * 3.0 + p.x) * sh * 0.05)
          item_top = sy + int(sh * 0.3) + bob_offset
          item_bot = sy + int(sh * 0.7) + bob_offset

          if p.kind == 'health':
            # Green cross
            cross_center = (item_top + item_bot) // 2
            cross_half = (item_bot - item_top) // 2
            if 0.35 < frac < 0.65:
              rl.draw_rectangle(strip_x, item_top, 1, item_bot - item_top, rl.Color(0, 220, 0, 255))
            if 0.2 < frac < 0.8:
              rl.draw_rectangle(strip_x, cross_center - cross_half // 2, 1, cross_half, rl.Color(0, 220, 0, 255))
          else:
            # Yellow ammo box
            if 0.2 < frac < 0.8:
              rl.draw_rectangle(strip_x, item_top, 1, item_bot - item_top, rl.Color(220, 200, 50, 255))
              # Bullet shape in center
              if 0.4 < frac < 0.6:
                rl.draw_rectangle(strip_x, item_top + 2, 1, max((item_bot - item_top) // 2, 1),
                                  rl.Color(180, 140, 30, 255))

  # ── HUD & Overlays ──────────────────────────────────────────────────────

  def _draw_crosshair(self, rect: rl.Rectangle):
    cx = int(rect.x + rect.width / 2)
    cy = int(rect.y + rect.height / 2)
    size = 12
    thick = 2
    color = rl.Color(255, 255, 255, 180)
    rl.draw_rectangle(cx - size, cy - thick // 2, size - 4, thick, color)
    rl.draw_rectangle(cx + 4, cy - thick // 2, size - 4, thick, color)
    rl.draw_rectangle(cx - thick // 2, cy - size, thick, size - 4, color)
    rl.draw_rectangle(cx - thick // 2, cy + 4, thick, size - 4, color)

  def _draw_gun(self, rect: rl.Rectangle):
    """Draw a stylized gun at the bottom center."""
    cx = int(rect.x + rect.width / 2)
    by = int(rect.y + rect.height)

    # Walk bob
    bob_x = int(math.sin(self.bob_phase) * 8)
    bob_y = int(abs(math.cos(self.bob_phase)) * 6)

    # Recoil when shooting
    recoil_y = 0
    if self.shoot_cd > SHOOT_COOLDOWN * 0.5:
      recoil_y = 20

    gx = cx - 60 + bob_x
    gy = by - 180 + bob_y + recoil_y

    # Gun barrel
    rl.draw_rectangle(gx + 25, gy, 12, 80, rl.Color(80, 80, 80, 255))
    rl.draw_rectangle(gx + 23, gy + 10, 16, 60, rl.Color(65, 65, 65, 255))
    # Gun body
    rl.draw_rectangle(gx + 10, gy + 70, 42, 35, rl.Color(90, 85, 75, 255))
    rl.draw_rectangle(gx + 12, gy + 72, 38, 31, rl.Color(75, 70, 62, 255))
    # Grip
    rl.draw_rectangle(gx + 20, gy + 105, 22, 55, rl.Color(60, 55, 45, 255))
    rl.draw_rectangle(gx + 22, gy + 107, 18, 51, rl.Color(70, 65, 55, 255))
    # Trigger guard
    rl.draw_rectangle(gx + 12, gy + 100, 8, 25, rl.Color(70, 70, 65, 255))
    # Barrel tip
    rl.draw_rectangle(gx + 27, gy - 5, 8, 10, rl.Color(60, 60, 60, 255))

    # Muzzle flash
    if self.muzzle_flash > 0:
      flash_intensity = self.muzzle_flash / 0.15
      flash_size = int(40 * flash_intensity)
      flash_color = rl.Color(255, 220, 50, int(200 * flash_intensity))
      rl.draw_circle(gx + 31, gy - 10, flash_size, flash_color)
      rl.draw_circle(gx + 31, gy - 10, flash_size // 2, rl.Color(255, 255, 200, int(255 * flash_intensity)))

  def _draw_hud(self, rect: rl.Rectangle):
    """Draw the HUD bar at the bottom."""
    hud_h = 70
    hud_y = int(rect.y + rect.height - hud_h)
    hud_x = int(rect.x)
    hud_w = int(rect.width)

    # Dark background bar
    rl.draw_rectangle(hud_x, hud_y, hud_w, hud_h, rl.Color(20, 20, 25, 220))
    rl.draw_line_ex(rl.Vector2(hud_x, hud_y), rl.Vector2(hud_x + hud_w, hud_y), 2, rl.Color(80, 80, 90, 255))

    font = gui_app.font(FontWeight.BOLD)
    pad = 30

    # Health
    health_color = rl.Color(220, 50, 50, 255) if self.health < 30 else rl.Color(50, 200, 50, 255)
    rl.draw_text_ex(font, "HEALTH", rl.Vector2(hud_x + pad, hud_y + 8), 24, 0, rl.Color(150, 150, 150, 255))
    rl.draw_text_ex(font, str(self.health), rl.Vector2(hud_x + pad, hud_y + 34), 32, 0, health_color)
    # Health bar
    bar_x = hud_x + pad + 120
    bar_w = 200
    bar_h = 16
    bar_y = hud_y + 38
    rl.draw_rectangle(bar_x, bar_y, bar_w, bar_h, rl.Color(60, 20, 20, 255))
    rl.draw_rectangle(bar_x, bar_y, int(bar_w * self.health / 100), bar_h, health_color)

    # Ammo
    ammo_x = hud_x + hud_w - 300
    ammo_color = rl.Color(220, 180, 50, 255) if self.ammo > 5 else rl.Color(220, 50, 50, 255)
    rl.draw_text_ex(font, "AMMO", rl.Vector2(ammo_x, hud_y + 8), 24, 0, rl.Color(150, 150, 150, 255))
    rl.draw_text_ex(font, str(self.ammo), rl.Vector2(ammo_x, hud_y + 34), 32, 0, ammo_color)

    # Score & Kills
    score_x = hud_x + hud_w // 2 - 80
    rl.draw_text_ex(font, f"KILLS: {self.kills}", rl.Vector2(score_x, hud_y + 8), 24, 0, rl.Color(200, 200, 200, 255))
    rl.draw_text_ex(font, f"SCORE: {self.score}", rl.Vector2(score_x, hud_y + 34), 28, 0, rl.Color(255, 220, 100, 255))

    # Controls hint (top of screen)
    hint_font = gui_app.font(FontWeight.NORMAL)
    rl.draw_text_ex(hint_font, "WASD: Move  Arrows: Turn  SPACE: Shoot  ESC: Quit",
                    rl.Vector2(int(rect.x + 20), int(rect.y + 10)), 22, 0, rl.Color(180, 180, 180, 120))

  def _draw_minimap(self, rect: rl.Rectangle):
    """Draw a small minimap in the top-right corner."""
    map_size = 160
    tile = map_size // MAP_W
    mx = int(rect.x + rect.width - map_size - 15)
    my = int(rect.y + 40)

    # Background
    rl.draw_rectangle(mx - 2, my - 2, map_size + 4, map_size + 4, rl.Color(0, 0, 0, 180))

    for row in range(MAP_H):
      for col in range(MAP_W):
        if WORLD_MAP[row][col] > 0:
          wt = WORLD_MAP[row][col]
          if wt in WALL_COLORS:
            c = WALL_COLORS[wt][0]
            c = rl.Color(c.r // 2, c.g // 2, c.b // 2, 200)
          else:
            c = rl.Color(100, 100, 100, 200)
          rl.draw_rectangle(mx + col * tile, my + row * tile, tile, tile, c)

    # Player dot
    ppx = int(mx + self.px * tile)
    ppy = int(my + self.py * tile)
    rl.draw_circle(ppx, ppy, 3, rl.Color(0, 255, 0, 255))
    # Direction line
    dx = int(math.cos(self.pa) * 8)
    dy = int(math.sin(self.pa) * 8)
    rl.draw_line_ex(rl.Vector2(ppx, ppy), rl.Vector2(ppx + dx, ppy + dy), 2, rl.Color(0, 255, 0, 200))

    # Enemy dots
    for e in self.enemies:
      if e.alive:
        ex = int(mx + e.x * tile)
        ey = int(my + e.y * tile)
        c = rl.Color(255, 60, 60, 200) if e.kind == 0 else rl.Color(200, 60, 200, 200)
        rl.draw_circle(ex, ey, 2, c)

    # Pickup dots
    for p in self.pickups:
      if p.active:
        px = int(mx + p.x * tile)
        py = int(my + p.y * tile)
        c = rl.Color(0, 200, 0, 200) if p.kind == 'health' else rl.Color(200, 200, 0, 200)
        rl.draw_circle(px, py, 2, c)

  # ── Game Over / Win ────────────────────────────────────────────────────

  def _draw_game_over(self, rect: rl.Rectangle):
    rl.draw_rectangle(int(rect.x), int(rect.y), int(rect.width), int(rect.height), rl.Color(80, 0, 0, 200))
    font = gui_app.font(FontWeight.BOLD)
    cx = int(rect.x + rect.width / 2)
    cy = int(rect.y + rect.height / 2)

    rl.draw_text_ex(font, "YOU DIED", rl.Vector2(cx - 180, cy - 60), 80, 0, rl.Color(255, 50, 50, 255))
    rl.draw_text_ex(font, f"Score: {self.score}   Kills: {self.kills}/{len(self.enemies)}",
                    rl.Vector2(cx - 200, cy + 40), 36, 0, rl.Color(200, 200, 200, 255))

    norm_font = gui_app.font(FontWeight.NORMAL)
    rl.draw_text_ex(norm_font, "Press R to restart  |  ESC to quit",
                    rl.Vector2(cx - 200, cy + 100), 28, 0, rl.Color(180, 180, 180, 200))

  def _draw_win_screen(self, rect: rl.Rectangle):
    rl.draw_rectangle(int(rect.x), int(rect.y), int(rect.width), int(rect.height), rl.Color(0, 40, 0, 200))
    font = gui_app.font(FontWeight.BOLD)
    cx = int(rect.x + rect.width / 2)
    cy = int(rect.y + rect.height / 2)

    rl.draw_text_ex(font, "LEVEL COMPLETE!", rl.Vector2(cx - 280, cy - 60), 80, 0, rl.Color(50, 255, 50, 255))
    rl.draw_text_ex(font, f"Score: {self.score}   Kills: {self.kills}/{len(self.enemies)}",
                    rl.Vector2(cx - 200, cy + 40), 36, 0, rl.Color(200, 200, 200, 255))

    norm_font = gui_app.font(FontWeight.NORMAL)
    rl.draw_text_ex(norm_font, "Press R to restart  |  ESC to quit",
                    rl.Vector2(cx - 200, cy + 100), 28, 0, rl.Color(180, 180, 180, 200))

  # ── Main Render ──────────────────────────────────────────────────────────

  def _render(self, rect: rl.Rectangle):
    dt = rl.get_frame_time()
    dt = min(dt, 0.05)

    # Black background
    rl.draw_rectangle(int(rect.x), int(rect.y), int(rect.width), int(rect.height), rl.BLACK)

    if self.game_over:
      self._draw_game_over(rect)
      if rl.is_key_pressed(rl.KeyboardKey.KEY_R):
        self._init_game()
      if rl.is_key_pressed(rl.KeyboardKey.KEY_ESCAPE):
        gui_app.pop_widget()
      return

    if self.game_won:
      self._draw_win_screen(rect)
      if rl.is_key_pressed(rl.KeyboardKey.KEY_R):
        self._init_game()
      if rl.is_key_pressed(rl.KeyboardKey.KEY_ESCAPE):
        gui_app.pop_widget()
      return

    # Update
    self._handle_input(dt)
    self._update_enemies(dt)
    self._update_pickups()
    self._update_timers(dt)

    # Check win
    if all(not e.alive for e in self.enemies):
      self.game_won = True

    # Draw world
    self._draw_ceiling_floor(rect)
    self._cast_rays(rect)
    self._draw_sprites(rect)

    # Overlays
    self._draw_gun(rect)
    self._draw_crosshair(rect)
    self._draw_hud(rect)
    self._draw_minimap(rect)

    # Damage flash
    if self.damage_flash > 0:
      alpha = int(min(self.damage_flash / 0.3, 1.0) * 120)
      rl.draw_rectangle(int(rect.x), int(rect.y), int(rect.width), int(rect.height),
                        rl.Color(255, 0, 0, alpha))

    # Hit marker
    if self.hit_marker > 0:
      cx = int(rect.x + rect.width / 2)
      cy = int(rect.y + rect.height / 2)
      s = 18
      rl.draw_line_ex(rl.Vector2(cx - s, cy - s), rl.Vector2(cx + s, cy + s), 3, rl.Color(255, 50, 50, 255))
      rl.draw_line_ex(rl.Vector2(cx + s, cy - s), rl.Vector2(cx - s, cy + s), 3, rl.Color(255, 50, 50, 255))

    # ESC to quit
    if rl.is_key_pressed(rl.KeyboardKey.KEY_ESCAPE):
      gui_app.pop_widget()
