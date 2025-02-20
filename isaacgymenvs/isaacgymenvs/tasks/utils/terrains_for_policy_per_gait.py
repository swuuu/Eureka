import numpy as np
from isaacgym import terrain_utils

class TerrainsForPolicyPerGait():
    def __init__(self, cfg, num_robots) -> None:
        self.type = cfg["terrainType"]
        if self.type in ["none", 'plane']:
            return
        self.horizontal_scale = 0.1
        self.vertical_scale = 0.005
        self.border_size = 20
        self.env_length = cfg["mapLength"]
        self.env_width = cfg["mapWidth"]
        self.proportions = [np.sum(cfg["terrainProportions"][:i+1]) for i in range(len(cfg["terrainProportions"]))]

        self.env_rows = cfg["numLevels"]
        self.env_cols = cfg["numTerrains"]
        self.num_maps = self.env_rows * self.env_cols
        self.env_origins = np.zeros((self.env_rows, self.env_cols, 3))

        self.width_per_env_pixels = int(self.env_width / self.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / self.horizontal_scale)

        self.border = int(self.border_size/self.horizontal_scale)
        self.tot_cols = int(self.env_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(self.env_rows * self.length_per_env_pixels) + 2 * self.border

        print(f'{cfg["terrainProportions"]}')
        self.terrain_types = cfg["terrainColTypes"]

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        if cfg["curriculum"]:
            self.curiculum(num_robots, num_terrains=self.env_cols, num_levels=self.env_rows)
        else:
            self.randomized_terrain()   
        self.heightsamples = self.height_field_raw
        self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(self.height_field_raw, self.horizontal_scale, self.vertical_scale, cfg["slopeTreshold"])

    def curiculum(self, num_robots, num_terrains, num_levels):
        for j in range(num_terrains):
            for i in range(num_levels):
                difficulty = i / num_levels
                choice = self.terrain_types[j]
                terrain = self.make_terrain_cell(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)
                
    def make_terrain_cell(self, choice, difficulty):
        # terrain parameters
        platform_size = 2 # [m]
        slope_rng = [0.0, 0.15] # [rad]
        max_height_rough = [0.0, 0.1] # [m]
        stair_width = [1.0, 0.4]
        # stair_height = [0.05, 0.20, 0.12, 0.25] # min - max single step, min - max multi step
        stair_height = [0.05, 0.20, 0.05, 0.25]
        max_stair_height_scale = 0.2
        discrete_obstacles_height = [0.0, 0.1]
        space_before_stairs = 3.0
        amplitude = [0.05, 0.1]

        # make the terrain
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        if choice == "smooth_pyramid_slope":
            slope = slope_rng[0] + (slope_rng[1] - slope_rng[0]) * difficulty
            slope = -slope if np.random.rand() < 0.5 else slope 
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=platform_size)
        elif choice == "rough_pyramid_slope":
            slope = slope_rng[0] + (slope_rng[1] - slope_rng[0]) * difficulty
            slope = -slope if np.random.rand() < 0.5 else slope 
            max_height = max_height_rough[0] + (max_height_rough[1] - max_height_rough[0]) * difficulty
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=platform_size)
            terrain_utils.random_uniform_terrain(terrain, min_height=-max_height, max_height=max_height, step=0.005, downsampled_scale=0.2)
        elif choice == "pyramid_stairs_down":
            step_width = stair_width[0] + (stair_width[1] - stair_width[0]) * difficulty
            step_height = stair_height[2] + (stair_height[3] - stair_height[2]) * difficulty
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=step_width, step_height=step_height, platform_size=platform_size)
        elif choice == "pyramid_stairs_up":
            step_width = stair_width[0] + (stair_width[1] - stair_width[0]) * difficulty
            step_height = stair_height[2] + (self.cfg.stair_height[3] - self.cfg.stair_height[2]) * difficulty
            step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=step_width, step_height=step_height, platform_size=platform_size)
        elif choice == "discrete":
            height = discrete_obstacles_height[0] + (discrete_obstacles_height[1] - discrete_obstacles_height[0]) * difficulty
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=platform_size)
        elif choice == "wave":
            num_waves = 5
            amplitude = amplitude[0] + (amplitude[1] - amplitude[0]) * difficulty
            terrain_utils.wave_terrain(terrain, num_waves=num_waves, amplitude=amplitude)
        else:
            raise ValueError("Unknown terrain type: {}".format(choice))
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        # save the env bounds (used for modifying the env terrain)
        self.terrain_bounds[i, j] = np.array([start_x, end_x, start_y, end_y])

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
