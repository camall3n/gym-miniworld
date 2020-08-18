import numpy as np
import math
from gym import spaces
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box, Key

class LockBox(MiniWorldEnv):
    """
    Multi-room environment with a locked door
    The agent must:
        - pick up the key,
        - use the key to open the door
        - then collect the treasure
    """

    def __init__(self, **kwargs):
        super().__init__(
            max_episode_steps=250,
            **kwargs
        )

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward+1)

    def _gen_world(self):
        # Top-left room
        room0 = self.add_rect_room(
            min_x=-7, max_x=-1,
            min_z=1 , max_z=7,
            wall_tex='marble',
        )
        # Top-right room
        room1 = self.add_rect_room(
            min_x=1, max_x=7,
            min_z=1, max_z=7,
            wall_tex='brick_wall',
        )
        # Bottom-right room
        room2 = self.add_rect_room(
            min_x=1 , max_x=7,
            min_z=-7, max_z=-1,
            wall_tex='wood',
        )
        # Bottom-left room
        room3 = self.add_rect_room(
            min_x=-7, max_x=-1,
            min_z=-7, max_z=-1,
            wall_tex='rock',
        )

        # Add openings to connect the rooms together
        self.connect_rooms(room0, room1, min_z=3, max_z=5, max_y=2.2)
        self.connect_rooms(room1, room2, min_x=3, max_x=5, max_y=2.2)
        self.connect_rooms(room2, room3, min_z=-5, max_z=-3, max_y=2.2)
        self.connect_rooms(room3, room0, min_x=-5, max_x=-3, max_y=2.2)

        self.key = self.place_entity(
                        Key(color='yellow'),
                        room=room2,
                        pos=(5,.8,-5),
                        # pos=(-4,.8,4),
                    )
        self.box = self.place_entity(
                        Box(color='red', size=1.0),
                        room=room3,
                        pos=(-5,0,-5),)

        self.place_agent(room=room0, min_x=-5, max_x=-5, min_z=5, max_z=5, dir=math.pi/4)

        self.ignore_carrying_intersect = True

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.agent.carrying:
            if self.near(self.key, self.box):
                self.entities.remove(self.agent.carrying)
                self.agent.carrying = None
                print('unlocked!')
                reward += self._reward()
                done = True

        return obs, reward, done, info
