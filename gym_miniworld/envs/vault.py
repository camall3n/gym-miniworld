import numpy as np
import math
from gym import spaces
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box, Key, Door

class Vault(MiniWorldEnv):
    """
    Multi-room environment with treasure behind a locked door
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
        self.room0 = self.add_rect_room(
            min_x=-7, max_x=-1,
            min_z=1 , max_z=7,
            wall_tex='marble',
            name='room0',
        )
        # Bottom-right room
        self.room1 = self.add_rect_room(
            min_x=1, max_x=7,
            min_z=1, max_z=7,
            wall_tex='brick_wall',
            name='room1',
        )
        # Top-right room
        self.room2 = self.add_rect_room(
            min_x=1 , max_x=7,
            min_z=-7, max_z=-1,
            wall_tex='wood',
            name='room2',
        )
        # Top-left room
        self.room3 = self.add_rect_room(
            min_x=-7, max_x=-1,
            min_z=-7, max_z=-1,
            wall_tex='rock',
            name='room3',
        )
        # Treasure room
        self.room4 = self.add_rect_room(
            min_x=-7, max_x=-1,
            min_z=-15, max_z=-9,
            wall_tex='stucco',
            name='room4'
        )
        # self.main_rooms = [self.room0, self.room1, self.room2, self.room3, self.room4]

        # Add openings to connect the rooms together
        self.connect_rooms(self.room0, self.room1, min_z=3, max_z=5, max_y=2.2, name='hall_0_1')
        self.connect_rooms(self.room1, self.room2, min_x=3, max_x=5, max_y=2.2, name='hall_1_2')
        self.connect_rooms(self.room2, self.room3, min_z=-5, max_z=-3, max_y=2.2, name='hall_2_3')
        self.connect_rooms(self.room3, self.room0, min_x=-5, max_x=-3, max_y=2.2, name='hall_3_4')
        self.connect_rooms(self.room4, self.room3, min_x=-5, max_x=-3, max_y=2.2, name='vault_door')

        self.key = self.place_entity(
            Key(color='yellow'),
            room=self.room2,
            pos=np.array([5,.8,-5]),
        )

        self.door = self.place_entity(
            Door(),
            pos=np.array([-4,0,-8]),
            dir=np.pi/2,
        )

        self.gold = self.place_entity(
            Box(color='gold', size=0.5),
            room=self.room4,
            pos=np.array([-4,.9,-12]),
        )

        self.place_agent(room=self.room0, min_x=-5, max_x=-5, min_z=5, max_z=5, dir=np.pi/4)

        self.ignore_carrying_intersect = True

    def step(self, action):
        experiences = {'obs': [], 'reward': [], 'done': [], 'info': [], 'skills_valid': []}
        for action in self.get_next_skill_action(action):
            obs, reward, done, info = super().step(action)

            if action == self.actions.drop and self.key.pos[1] < .5:
                self.key.pos[1] = .8
            if (action == self.actions.toggle and self.agent.carrying is self.key
                and self.near(self.key, self.door)):
                    self.agent.carrying = None
                    self.entities.remove(self.key)
                    self.entities.remove(self.door)
            if action == self.actions.pickup and self.agent.carrying is self.gold:
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
        elif skill == self.actions.skill2:
            for timestep in range(40):
                yield self.skill_go_to_center_of_current_room()
        elif skill == self.actions.skill3:
            for timestep in range(40):
                yield self.skill_go_to_hallway(direction='clockwise')
            for timestep in range(3):
                yield self.actions.move_forward
        elif skill == self.actions.skill4:
            for timestep in range(40):
                yield self.skill_go_to_hallway(direction='counterclockwise')
            for timestep in range(3):
                yield self.actions.move_forward
        elif skill == self.actions.skill5:
            for timestep in range(40):
                yield self.skill_go_to_vault_door()
        else:
            yield skill

    def skill_go_to_center_of_current_room(self):
        rooms = [room for room in self.rooms if 'room' in room.name]
        room_centers = [self.get_room_center(room) for room in rooms]
        dist_to_centers = [self.dist(self.agent.pos, room_ctr) for room_ctr in room_centers]
        current_room = np.argmin(dist_to_centers)
        return self.skill_go_to_position(room_centers[current_room])

    def skill_go_to_position(self, target_pos):
        vector_to_center = target_pos - self.agent.pos
        dist_to_center = np.linalg.norm(vector_to_center)

        if dist_to_center > 0.4:
            angle_to_center = self.vector2angle(vector_to_center)
            angle_to_face_center = self.relative_angle(angle_to_center)
            if np.abs(angle_to_face_center) < .2:
                return self.actions.move_forward
            elif angle_to_face_center > 0:
                return self.actions.turn_left
            else:
                return self.actions.turn_right
        else:
            return None

    def skill_go_to_hallway(self, direction='clockwise'):
        hallways = [room for room in self.rooms if 'hall' in room.name]
        hallway_centers = [self.get_room_center(hallway) for hallway in hallways]
        agent_angle = self.vector2angle(self.agent.pos)
        hallway_angles = [self.vector2angle(pos) for pos in hallway_centers]
        if direction in ['clockwise', 'cw']:
            next_hallway = np.argmin([np.mod(angle-agent_angle,2*np.pi) for angle in hallway_angles])
        else:
            next_hallway = np.argmin([np.mod(agent_angle-angle,2*np.pi) for angle in hallway_angles])
        return self.skill_go_to_position(hallway_centers[next_hallway])

    def skill_go_to_vault_door(self):
        rooms = [room for room in self.rooms if 'room' in room.name]
        room_centers = [self.get_room_center(room) for room in rooms]
        dist_to_centers = [self.dist(self.agent.pos, room_ctr) for room_ctr in room_centers]
        current_room = rooms[np.argmin(dist_to_centers)]
        if current_room.name != 'room3':
            return False
        else:
            vault_door = [room for room in self.rooms if room.name == 'vault_door'][0]
            return self.skill_go_to_position(self.get_room_center(vault_door))


    @staticmethod
    def get_room_center(room):
        return np.array([room.mid_x, 0, room.mid_z])

    @staticmethod
    def vector2angle(vector):
        dx, _, dy = vector
        return np.arctan2(-dy, dx)

    def relative_angle(self, angle):
        return self.wrap_angle(angle - self.agent.dir)

    @staticmethod
    def dist(pos1, pos2):
        return np.linalg.norm(pos1 - pos2)

    @staticmethod
    def wrap_angle(angle):
        return np.angle(np.exp(1j*angle))
