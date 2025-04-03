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

        self.env_rows = cfg["numLevels"]
        self.env_cols = cfg["numTerrains"]
        self.num_maps = self.env_rows * self.env_cols
        self.env_origins = np.zeros((self.env_rows, self.env_cols, 3))

        self.width_per_env_pixels = int(self.env_width / self.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / self.horizontal_scale)

        self.border = int(self.border_size/self.horizontal_scale)
        self.tot_cols = int(self.env_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(self.env_rows * self.length_per_env_pixels) + 2 * self.border

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
        stair_height = [0.05, 0.25]
        max_stair_height_scale = 0.2
        discrete_obstacles_height = [0.0, 0.1]
        space_before_stairs = 3.0
        amplitude = [0.05, 0.1]
        constant_step_width = 0.31
        use_fixed_step_width = True 

        # make the terrain
        terrain = terrain_utils.SubTerrain("terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.vertical_scale,
                                horizontal_scale=self.horizontal_scale)
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
            if not use_fixed_step_width:
                step_width = stair_width[0] + (stair_width[1] - stair_width[0]) * difficulty
            else:
                step_width = constant_step_width
            step_height = stair_height[0] + (stair_height[1] - stair_height[0]) * difficulty
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=step_width, step_height=step_height, platform_size=platform_size)
        elif choice == "pyramid_stairs_up":
            if not use_fixed_step_width:
                step_width = stair_width[0] + (stair_width[1] - stair_width[0]) * difficulty
            else:
                step_width = constant_step_width
            step_height = stair_height[0] + (stair_height[1] - stair_height[0]) * difficulty
            step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=step_width, step_height=step_height, platform_size=platform_size)
            print(f'step_width: {step_width}, step_height: {step_height}')
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
        elif choice == "flat":
            pass
        elif choice == "pyramid_stairs_up_terrain_inner_square":
            pyramid_area_size = 7 # terrain.width * 0.25
            if not use_fixed_step_width:
                step_width = stair_width[0] + (stair_width[1] - stair_width[0]) * difficulty
            else:
                step_width = constant_step_width
            step_height = stair_height[0] + (stair_height[1] - stair_height[0]) * difficulty
            step_height *= -1
            pyramid_stairs_terrain_inner_square(terrain, step_width=step_width, step_height=step_height, platform_size=platform_size, pyramid_area_size=pyramid_area_size)
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

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

def pyramid_stairs_terrain_inner_square(terrain, step_width, step_height, platform_size=1., pyramid_area_size=2.0):
    """
    Generate a pyramid stairs structure only in the center of the terrain.
    
    Parameters:
        terrain (SubTerrain): the terrain object
        step_width (float): the width of each stair step [meters]
        step_height (float): the height of each stair step [meters]
        platform_size (float): size of the flat platform at the center of the stairs [meters]
        pyramid_area_size (float): size of the square area in which to apply the pyramid [meters]
    Returns:
        terrain (SubTerrain): updated terrain
    """
    # Convert real-world dimensions to grid units
    step_width = int(step_width / terrain.horizontal_scale)
    step_height = int(step_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    pyramid_size = int(pyramid_area_size / terrain.horizontal_scale)

    # Determine the bounds of the central square to apply the pyramid
    center_x = terrain.width // 2
    center_y = terrain.length // 2
    half_size = pyramid_size // 2
    start_x = center_x - half_size
    stop_x = center_x + half_size
    start_y = center_y - half_size
    stop_y = center_y + half_size

    # Clip to ensure the sub-region is within terrain bounds
    start_x = max(start_x, 0)
    stop_x = min(stop_x, terrain.width)
    start_y = max(start_y, 0)
    stop_y = min(stop_y, terrain.length)

    # Create a copy of the bounds to shrink as we add stairs inward
    cur_start_x = start_x
    cur_stop_x = stop_x
    cur_start_y = start_y
    cur_stop_y = stop_y

    height = 0
    while (cur_stop_x - cur_start_x) > platform_size and (cur_stop_y - cur_start_y) > platform_size:
        cur_start_x += step_width
        cur_stop_x -= step_width
        cur_start_y += step_width
        cur_stop_y -= step_width
        height += step_height
        terrain.height_field_raw[cur_start_x:cur_stop_x, cur_start_y:cur_stop_y] = height

    return terrain
