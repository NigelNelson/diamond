from typing import Dict, List, Tuple

# import gymnasium
import pygame


ActionNames = List[str]
Keymap = Dict[Tuple[int], int]


def get_keymap_and_action_names(name: str) -> Tuple[Keymap, ActionNames]:
    if name == "empty":
        return EMPTY_KEYMAP, EMPTY_ACTION_NAMES

    if name == "dataset_mode":
        return DATASET_MODE_KEYMAP, DATASET_MODE_ACTION_NAMES

    if name == "atari":
        return ATARI_KEYMAP, ATARI_ACTION_NAMES

    if name == "surgical":
        return SURGICAL_KEYMAP, SURGICAL_ACTION_NAMES

    assert name.startswith("atari/")
    env_id = name.split("atari/")[1]
    action_names = [x.lower() for x in gymnasium.make(env_id).unwrapped.get_action_meanings()]
    keymap = {}
    for key, value in ATARI_KEYMAP.items():
        if ATARI_ACTION_NAMES[value] in action_names:
            keymap[key] = action_names.index(ATARI_ACTION_NAMES[value])
    return keymap, action_names


ATARI_ACTION_NAMES = [
    "noop",
    "fire",
    "up",
    "right",
    "left",
    "down",
    "upright",
    "upleft",
    "downright",
    "downleft",
    "upfire",
    "rightfire",
    "leftfire",
    "downfire",
    "uprightfire",
    "upleftfire",
    "downrightfire",
    "downleftfire",
]

ATARI_KEYMAP = {
    (pygame.K_SPACE,): 1,
    (pygame.K_w,): 2,
    (pygame.K_d,): 3,
    (pygame.K_a,): 4,
    (pygame.K_s,): 5,

    (pygame.K_w, pygame.K_d): 6,
    (pygame.K_w, pygame.K_a): 7,
    (pygame.K_s, pygame.K_d): 8,
    (pygame.K_s, pygame.K_a): 9,

    (pygame.K_w, pygame.K_SPACE): 10,
    (pygame.K_d, pygame.K_SPACE): 11,
    (pygame.K_a, pygame.K_SPACE): 12,
    (pygame.K_s, pygame.K_SPACE): 13,

    (pygame.K_w, pygame.K_d, pygame.K_SPACE): 14,
    (pygame.K_w, pygame.K_a, pygame.K_SPACE): 15,
    (pygame.K_s, pygame.K_d, pygame.K_SPACE): 16,
    (pygame.K_s, pygame.K_a, pygame.K_SPACE): 17,

}

DATASET_MODE_ACTION_NAMES = [
    "noop",
    "previous",
    "next",
    "previous_10",
    "next_10",
]

DATASET_MODE_KEYMAP = {
    (pygame.K_LEFT,): 1,
    (pygame.K_RIGHT,): 2,
    (pygame.K_PAGEDOWN,): 3,
    (pygame.K_PAGEUP,): 4,
}

EMPTY_ACTION_NAMES = [
    "noop",
]

EMPTY_KEYMAP = {}

SURGICAL_ACTION_NAMES = [
    "noop",
    "left_move_x_pos", "left_move_x_neg",
    "left_move_y_pos", "left_move_y_neg",
    "left_move_z_pos", "left_move_z_neg",
    "left_rotate_x_pos", "left_rotate_x_neg",
    "left_rotate_y_pos", "left_rotate_y_neg",
    "left_rotate_z_pos", "left_rotate_z_neg",
    "left_gripper_open", "left_gripper_close",
    "right_move_x_pos", "right_move_x_neg",
    "right_move_y_pos", "right_move_y_neg",
    "right_move_z_pos", "right_move_z_neg",
    "right_rotate_x_pos", "right_rotate_x_neg",
    "right_rotate_y_pos", "right_rotate_y_neg",
    "right_rotate_z_pos", "right_rotate_z_neg",
    "right_gripper_open", "right_gripper_close",
]

SURGICAL_KEYMAP = {
    # Left MTM controls
    (pygame.K_q,): 1,  # left_move_x_pos
    (pygame.K_a,): 2,  # left_move_x_neg
    (pygame.K_w,): 3,  # left_move_y_pos
    (pygame.K_s,): 4,  # left_move_y_neg
    (pygame.K_e,): 5,  # left_move_z_pos
    (pygame.K_d,): 6,  # left_move_z_neg
    (pygame.K_r,): 7,  # left_rotate_x_pos
    (pygame.K_f,): 8,  # left_rotate_x_neg
    (pygame.K_t,): 9,  # left_rotate_y_pos
    (pygame.K_g,): 10,  # left_rotate_y_neg
    (pygame.K_y,): 11,  # left_rotate_z_pos
    (pygame.K_h,): 12,  # left_rotate_z_neg
    (pygame.K_z,): 13,  # left_gripper_open
    (pygame.K_x,): 14,  # left_gripper_close

    # Right MTM controls
    (pygame.K_u,): 15,  # right_move_x_pos
    (pygame.K_j,): 16,  # right_move_x_neg
    (pygame.K_i,): 17,  # right_move_y_pos
    (pygame.K_k,): 18,  # right_move_y_neg
    (pygame.K_o,): 19,  # right_move_z_pos
    (pygame.K_l,): 20,  # right_move_z_neg
    (pygame.K_p,): 21,  # right_rotate_x_pos
    (pygame.K_SEMICOLON,): 22,  # right_rotate_x_neg
    (pygame.K_LEFTBRACKET,): 23,  # right_rotate_y_pos
    (pygame.K_QUOTE,): 24,  # right_rotate_y_neg
    (pygame.K_RIGHTBRACKET,): 25,  # right_rotate_z_pos
    (pygame.K_BACKSLASH,): 26,  # right_rotate_z_neg
    (pygame.K_n,): 27,  # right_gripper_open
    (pygame.K_m,): 28,  # right_gripper_close
}
