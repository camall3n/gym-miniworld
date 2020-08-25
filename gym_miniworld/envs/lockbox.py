import numpy as np
import math
from gym import spaces
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box, Key, Door

class LockBox(MiniWorldEnv):
    """
    Multi-room environment with a locked door
    The agent must:
        - navigate to the key
        - pick up the key
        - navigate to the door
        - use the key to open the door
        - navigate to the treasure
        - collect the treasure
    """

    def __init__(self, **kwargs):
        super().__init__(
            max_episode_steps=250,
            **kwargs
        )

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward+1)

    def _gen_world(self):
        # Bottom-left room
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
        # Top-left room
        room3 = self.add_rect_room(
            min_x=-7, max_x=-1,
            min_z=-7, max_z=-1,
            wall_tex='rock',
        )
        # Treasure room
        room4 = self.add_rect_room(
            min_x=-7, max_x=-1,
            min_z=-15, max_z=-9,
            wall_tex='stucco',
        )

        # Add openings to connect the rooms together
        self.connect_rooms(room0, room1, min_z=3, max_z=5, max_y=2.2)
        self.connect_rooms(room1, room2, min_x=3, max_x=5, max_y=2.2)
        self.connect_rooms(room2, room3, min_z=-5, max_z=-3, max_y=2.2)
        self.connect_rooms(room3, room0, min_x=-5, max_x=-3, max_y=2.2)
        self.connect_rooms(room4, room3, min_x=-5, max_x=-3, max_y=2.2)

        self.key = self.place_entity(
            Key(color='yellow'),
            room=room2,
            pos=np.array([5,.8,-5]),
        )

        self.door = self.place_entity(
            Door(),
            pos=np.array([-4,0,-8]),
            dir=math.pi/2,
        )

        self.gold = self.place_entity(
            Box(color='gold', size=0.5),
            room=room4,
            pos=np.array([-4,.9,-12]),
        )

        self.place_agent(room=room0, min_x=-5, max_x=-5, min_z=5, max_z=5, dir=math.pi/4)

        self.ignore_carrying_intersect = True

    def step(self, action):
        experiences = {'obs': [], 'reward': [], 'done': [], 'info': [], 'skills_valid': []}
        for action in self.get_next_skill_action(action):
            obs, reward, done, info = super().step(action)

            if self.agent.carrying is self.key:
                if self.near(self.key, self.door) and action == self.actions.toggle:
                    self.agent.carrying = None
                    self.entities.remove(self.key)
                    self.entities.remove(self.door)
            if self.agent.carrying is self.gold:
                reward += self._reward()
                done = True

            experiences['obs'].append(obs)
            experiences['reward'].append(reward)
            experiences['done'].append(done)
            experiences['info'].append(info)

            if done:
                break

        obs, reward, done = experiences['obs'][-1], sum(experiences['reward']), experiences['done'][-1]
        if len(experiences['obs']) == 1:
            info = experiences['info']
        else:
            info = experiences
        return obs, reward, done, info

    def get_next_skill_action(self, skill):
        if skill == self.actions.skill1:
            for i in range(10):
                yield self.actions.move_forward
        else:
            yield skill
