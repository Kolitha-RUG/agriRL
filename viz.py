import pygame
import numpy as np
import gymnasium as gym
from vine_env import VineEnv  # Import your custom env class

# ==== CONFIGURATION ====
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 800
FPS = 5

# Colors
COLLECTION_COLOR = (255, 215, 0)  # gold
VINE_COLOR = (34, 139, 34)        # green
HUMAN_COLOR = (30, 144, 255)      # blue
DRONE_COLOR = (220, 20, 60)       # red
BACKGROUND = (240, 240, 240)      # light grey

# Maps world coords -> screen
def world_to_screen(pos, field_size):
    return (
        int(pos[0] / field_size[0] * SCREEN_WIDTH),
        int(pos[1] / field_size[1] * SCREEN_HEIGHT),
    )

# ---- Initialize pygame ----
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Vine Environment Visualization")
clock = pygame.time.Clock()

# ---- Load environment ----
env = gym.make("VineEnv-v0", render_mode=None)
base_env = env.unwrapped  # unwrap so we can inspect internals
obs, info = base_env.reset()

running = True
step = 0

while running:
    # ---- handle quit event ----
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # ---- draw background ----
    screen.fill(BACKGROUND)

    # ---- draw collection point ----
    cp = base_env.collection_point
    cp_screen = world_to_screen(cp, base_env.field_size)
    pygame.draw.circle(screen, COLLECTION_COLOR, cp_screen, 10)

    # ---- draw vines ----
    font = pygame.font.SysFont(None, 20)
    for i, v in enumerate(base_env.vines):
        vine_screen = world_to_screen(v.position, base_env.field_size)
        pygame.draw.rect(screen, VINE_COLOR, (*vine_screen, 12, 12))
        text = font.render(f"{v.boxes_remaining}/{v.queued_boxes}", True, (0, 0, 0))
        screen.blit(text, (vine_screen[0] - 10, vine_screen[1] + 15))

    # ---- draw humans ----
    for i, h in enumerate(base_env.humans):
        human_screen = world_to_screen(h.position, base_env.field_size)
        pygame.draw.circle(screen, HUMAN_COLOR, human_screen, 8)
        label = font.render(f"H{i}", True, (0, 0, 0))
        screen.blit(label, (human_screen[0] - 5, human_screen[1] - 20))

    # ---- draw drones ----
    for i, d in enumerate(base_env.drones):
        drone_screen = world_to_screen(d.position, base_env.field_size)
        # triangle for drone
        points = [
            (drone_screen[0], drone_screen[1] - 8),
            (drone_screen[0] - 8, drone_screen[1] + 8),
            (drone_screen[0] + 8, drone_screen[1] + 8),
        ]
        pygame.draw.polygon(screen, DRONE_COLOR, points)
        label = font.render(f"D{i}", True, (0, 0, 0))
        screen.blit(label, (drone_screen[0] - 5, drone_screen[1] - 25))

    pygame.display.flip()
    clock.tick(FPS)

    # ---- step the environment ----
    # (using random actions for initial visualization)
    action = base_env.action_space.sample()
    obs, reward, terminated, truncated, info = base_env.step(action)
    step += 1

    if terminated or truncated or step >= base_env.max_steps:
        obs, info = base_env.reset()
        step = 0

pygame.quit()
