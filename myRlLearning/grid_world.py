'''
Created on Mar 13, 2017

@author: Yury
'''

import numpy as np

REWARD_GOAL = 10
REWARD_PIT = -10
REWARD_HANG = -100

class EnvironmentFactory:
    class EnvironmentType:
        Deterministic = 0
        RandomPlayer = 1
        RandomPlayerAndGoal = 2
        RandomPlayerGoalAndPit = 3
        AllRandom = 4
        
    def __init__(self, env_type):
        self.env_type = env_type
        
    def create_environment(self, state=None):
        if self.env_type == EnvironmentFactory.EnvironmentType.Deterministic:
            cls = DeterministicEnvironment
        elif self.env_type == EnvironmentFactory.EnvironmentType.RandomPlayer:
            cls = RandomPlayerEnvironment
        elif self.env_type == EnvironmentFactory.EnvironmentType.RandomPlayerAndGoal:
            cls = RandomGoalAndPlayerEnvironment
        elif self.env_type == EnvironmentFactory.EnvironmentType.RandomPlayerGoalAndPit:
            cls = RandomGoalPlayerAndPitEnvironment
        elif self.env_type == EnvironmentFactory.EnvironmentType.AllRandom:
            cls = FullyRandomEnvironment
        else:
            cls = None
        
        if state == None:
            env = cls()
        else:
            env = cls.from_state(state)
        return env

class EnvironmentBase:
    size = 4
    grid_size = size * size
    grid_size_square = grid_size ** 2
    grid_size_cube = grid_size ** 3
    
    def __init__(self, player, goal, pit, wall, state):
        # Assume that all parameters are valid
        self.player_starting_point = player
        self.goal = goal
        self.goal_cartesian = (int(self.goal / self.size), int(self.goal % self.size))
        self.pit = pit
        self.pit_cartesian = (int(self.pit / self.size), int(self.pit % self.size))
        self.wall = wall
        self.wall_cartesian = (int(self.wall / self.size), int(self.wall % self.size))
        self.state = state
    
    def __str__(self):
        return "state %d, player %d, goal %d, pit %d, wall %d" % (self.state, self.player_starting_point, self.goal, self.pit, self.wall)
      
    def reward(self):
        player_pos = self.player_abs_from_state(self.state)
        if player_pos == self.pit:
            return -10
        elif player_pos == self.goal:
            return 10
        else:
            return -100
        
    def is_done(self):
        player_pos = self.player_abs_from_state(self.state)
        return player_pos == self.pit or player_pos == self.goal
    
    def player_cartesian(self):
        player_abs_pos = self.player_abs_from_state(self.state)
        return (int(player_abs_pos / self.size), player_abs_pos % self.size)
    
    def update_state(self, player_loc_cartesian):
        player_abs_pos = int(player_loc_cartesian[0] * self.size + player_loc_cartesian[1])
        self.state = self.player_abs_to_state(player_abs_pos)
        
    def get_state(self, player_loc_cartesian):
        player_abs_pos = int(player_loc_cartesian[0] * self.size + player_loc_cartesian[1])
        return self.player_abs_to_state(player_abs_pos)
            
    def show(self):
        grid = np.zeros((self.size, self.size), dtype='<U2')

        for i in range(0, self.size):
            for j in range(0, self.size):
                grid[i, j] = ' '
    

        grid[self.player_cartesian()] = 'P'  # player starting point
        grid[self.wall_cartesian] = 'W'  # wall
        grid[self.goal_cartesian] = '+'  # goal
        grid[self.pit_cartesian] = '-'  # pit
    
        print(grid)
        
    @classmethod
    def from_state(cls, state):
        player = cls.player_abs_from_state(state)
        goal = cls.goal_abs_from_state(state)
        pit = cls.pit_abs_from_state(state)
        wall = cls.wall_abs_from_state(state)
        
        # Check validity
        if player in [goal, pit, wall] or goal in [pit, wall] or pit in [wall]:
            return None
        
        return cls(player, goal, pit, wall, state)

class DeterministicEnvironment(EnvironmentBase):
    def __init__(self, player=None, goal=None, pit=None, wall=None, state=None):
        self.num_states = self.grid_size
        
        if state != None:
            super(DeterministicEnvironment, self).__init__(player, goal, pit, wall, state)
        else:
            self.player_starting_point = 0
            self.wall = 10
            self.wall_cartesian = (2, 2)
            self.goal = 15
            self.goal_cartesian = (3, 3)
            self.pit = 5
            self.pit_cartesian = (1, 1)
            
            self.state = self.player_abs_to_state(self.player_starting_point)
    
    def player_abs_to_state(self, player_abs):
        # In this environment everything initialized deterministically. Player can change position so it's location represent the state of the world.
        return player_abs
    
    @classmethod
    def player_abs_from_state(cls, state):
        # State represent's player position
        return state
    
    @classmethod
    def goal_abs_from_state(cls, state):
        # In this environment goal is fixed
        return 15
    
    @classmethod
    def pit_abs_from_state(cls, state):
        # In this environment pit is fixed
        return 5
    
    @classmethod
    def wall_abs_from_state(cls, state):
        # In this environment wall is fixed
        return 10

class RandomPlayerEnvironment(DeterministicEnvironment):
    def __init__(self, player=None, goal=None, pit=None, wall=None, state=None):
        self.num_states = self.grid_size
        
        if state != None:
            super(RandomPlayerEnvironment, self).__init__(player, goal, pit, wall, state)
        else:
            self.wall = 10
            self.wall_cartesian = (2, 2)
            self.goal = 15
            self.goal_cartesian = (3, 3)
            self.pit = 5
            self.pit_cartesian = (1, 1)
            
            # Initialize player random location
            self.player_starting_point = np.random.choice(self.grid_size)
            while self.player_starting_point in [self.wall, self.pit, self.goal]:
                self.player_starting_point = np.random.choice(self.grid_size)
            
            self.state = self.player_abs_to_state(self.player_starting_point)

class RandomGoalAndPlayerEnvironment(EnvironmentBase):
    def __init__(self, player=None, goal=None, pit=None, wall=None, state=None):
        if state != None:
            super(RandomGoalAndPlayerEnvironment, self).__init__(player, goal, pit, wall, state)
        else:
            self.num_states = self.grid_size ** 2
            self.wall = 10
            self.wall_cartesian = (2, 2)
            self.pit = 5
            self.pit_cartesian = (1, 1)
    
            # Initialize goal random location
            self.goal = np.random.choice(self.grid_size)
            while self.goal in [self.wall, self.pit]:
                self.goal = np.random.choice(self.grid_size)
            self.goal_cartesian = (int(self.goal / self.size), int(self.goal % self.size))
            
            # Initialize player random location
            self.player_starting_point = np.random.choice(self.grid_size)
            while self.player_starting_point in [self.wall, self.pit, self.goal]:
                self.player_starting_point = np.random.choice(self.grid_size)
    
            self.state = self.player_abs_to_state(self.player_starting_point)
    
    def player_abs_to_state(self, player_abs):
        # We represent state as linear combination of (player and goal) were coordinates are (y,x) accordingly
        # So state = y*a + x where y is player coordinate and x - goal 
        return int(player_abs * self.grid_size + self.goal)
    
    @classmethod
    def player_abs_from_state(cls, state):
        # We need to find y coordinate from state = y*a + x so it just state/a
        return int(state / cls.grid_size)
    
    @classmethod
    def goal_abs_from_state(cls, state):
        # We need to find x coordinate from state = y*a + x so it just state mod a
        return int(state % cls.grid_size)
    
    @classmethod
    def pit_abs_from_state(cls, state):
        # In this environment pit is fixed
        return 5
    
    @classmethod
    def wall_abs_from_state(cls, state):
        # In this environment wall is fixed
        return 10

class RandomGoalPlayerAndPitEnvironment(EnvironmentBase):    
    def __init__(self, player=None, goal=None, pit=None, wall=None, state=None):
        if state != None:
            super(RandomGoalPlayerAndPitEnvironment, self).__init__(player, goal, pit, wall, state)
        else:
            self.num_states = self.grid_size ** 3
            self.wall = 10
            self.wall_cartesian = (2, 2)
            
            # Initialize goal random location
            self.pit = np.random.choice(self.grid_size)
            while self.pit in [self.wall]:
                self.pit = np.random.choice(self.grid_size)
            self.pit_cartesian = (int(self.pit / self.size), int(self.pit % self.size))
    
            # Initialize goal random location
            self.goal = np.random.choice(self.grid_size)
            while self.goal in [self.wall, self.pit]:
                self.goal = np.random.choice(self.grid_size)
            self.goal_cartesian = (int(self.goal / self.size), int(self.goal % self.size))
    
            # Initialize player random location
            self.player_starting_point = np.random.choice(self.grid_size)
            while self.player_starting_point in [self.wall, self.pit, self.goal]:
                self.player_starting_point = np.random.choice(self.grid_size)
    
            self.state = self.player_abs_to_state(self.player_starting_point)
    
    def player_abs_to_state(self, player_abs):
        # We represent state as linear combination of (player, goal and pit) were coordinates are (z,y,x) accordingly
        # So state = z*a^2 + y*a + x where z is player coordinate, y - goal and x - pit 
        return int(player_abs * self.grid_size_square + self.goal * self.grid_size + self.pit)

    @classmethod
    def player_abs_from_state(cls, state):
        # We need to find z coordinate from state = z*a^2 + y*a + x
        return int(state / cls.grid_size_square)
        
    @classmethod
    def goal_abs_from_state(cls, state):
        # We need to find y coordinate from state = z*a^2 + y*a + x
        return int((state % cls.grid_size_square) / cls.grid_size)
    
    @classmethod
    def pit_abs_from_state(cls, state):
        # We need to find x coordinate from state = z*a^2 + y*a + x
        return int(float(state % cls.grid_size_square) % cls.grid_size)
    
    @classmethod
    def wall_abs_from_state(cls, state):
        # In this environment wall is fixed
        return 10


class FullyRandomEnvironment(EnvironmentBase):    
    def __init__(self, player=None, goal=None, pit=None, wall=None, state=None):
        if state != None:
            super(FullyRandomEnvironment, self).__init__(player, goal, pit, wall, state)
        else:
            self.num_states = self.grid_size ** 4
            # Initialize wall random location
            self.wall = np.random.choice(self.grid_size)
            self.wall_cartesian = (int(self.wall / self.size), int(self.wall % self.size))
            
            # Initialize goal random location
            self.pit = np.random.choice(self.grid_size)
            while self.pit in [self.wall]:
                self.pit = np.random.choice(self.grid_size)
            self.pit_cartesian = (int(self.pit / self.size), int(self.pit % self.size))
    
            # Initialize goal random location
            self.goal = np.random.choice(self.grid_size)
            while self.goal in [self.wall, self.pit]:
                self.goal = np.random.choice(self.grid_size)
            self.goal_cartesian = (int(self.goal / self.size), int(self.goal % self.size))
    
            # Initialize player random location
            self.player_starting_point = np.random.choice(self.grid_size)
            while self.player_starting_point in [self.wall, self.pit, self.goal]:
                self.player_starting_point = np.random.choice(self.grid_size)
    
            self.state = self.player_abs_to_state(self.player_starting_point)
    
    def player_abs_to_state(self, player_abs):
        # We represent state as linear combination of (player, goal, pit and wall) were coordinates are (z,y,x,w) accordingly
        # So state = z*a^3 + y*a^2 + x*a + w where z is player coordinate, y - goal and x - pit and w - wall 
        return int(player_abs * self.grid_size_cube + self.goal * self.grid_size_square + self.pit * self.grid_size + self.wall)

    @classmethod
    def player_abs_from_state(cls, state):
        # We need to find z coordinate from state = z*a^3 + y*a^2 + x*a + w
        return int(state / cls.grid_size_cube)
        
    @classmethod
    def goal_abs_from_state(cls, state):
        # We need to find y coordinate from state = z*a^3 + y*a^2 + x*a + w
        return int((state % cls.grid_size_cube) / cls.grid_size_square)
    
    @classmethod
    def pit_abs_from_state(cls, state):
        # We need to find x coordinate from state = z*a^3 + y*a^2 + x*a + w
        return int(((state % cls.grid_size_cube) % cls.grid_size_square) / cls.grid_size)
    
    @classmethod
    def wall_abs_from_state(cls, state):
        # We need to find w coordinate from state = z*a^3 + y*a^2 + x*a + w
        return int(((state % cls.grid_size_cube) % cls.grid_size_square) % cls.grid_size) 

