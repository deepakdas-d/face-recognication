# app/session/challenge_generator.py
import random

CHALLENGES = ["turn_left", "turn_right", "look_up", "look_down", "smile", "blink"]

def generate_challenge():
    return random.choice(CHALLENGES)