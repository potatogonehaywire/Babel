# Vannah Tam
# Feb 23, 2026
# Tileset


import random
import arcade
import arcade.gui
import pyglet
import math
import json
from pyglet.math import Vec2
from collections import Counter

SPRITE_SCALING_PLAYER = 3
SPRITE_SCALING_HOTBAR = 1.5

SCREEN_WIDTH = 1680
SCREEN_HEIGHT = 1050

MOVE_SPEED = 6
CAMERA_SPEED = 1

TILE_SCALING = 4
TIME_LIMIT = 30.0

MAP_HEIGHT = 50

BGM = arcade.load_sound("data/sounds/library.ogg")
SLAP = arcade.load_sound("data/sounds/slap.ogg")
BUCKET = arcade.load_sound("data/sounds/water_bucket.ogg")
MOP = arcade.load_sound("data/sounds/mop.ogg")
MATCH = arcade.load_sound("data/sounds/match.ogg")
SHOVEL = arcade.load_sound("data/sounds/shovel.ogg")
FIRE = arcade.load_sound("data/sounds/fire.ogg")

LEVEL = 1

class Character(arcade.Sprite):
    """ Class for all Animated Characters, updates animation each frame """
    def __init__(self, sprite_frames, scale, frame_duration, x, y):
        super().__init__()
        self.frames = sprite_frames
        self.frame_duration = frame_duration
        self.current_time = 0.0
        self.current_frame_index = 0
        self.scale = scale
        self.frames_per_direction = 4
        self.direction = [-1,-1]
        self.center_x = x
        self.center_y = y        
        self.texture = self.frames[0]


    def update_animation(self, delta_time: float = 1 / 60):
        """Update which frame is displayed based on elapsed time."""
        self.current_time += delta_time
        if self.current_time >= self.frame_duration:
            self.current_time = 0
            match self.direction:
                case [-1, -1]:
                    self.current_frame_index = (self.current_frame_index + 1) % self.frames_per_direction
                    self.texture = self.frames[self.current_frame_index]
                case [1,-1]:
                    self.current_frame_index = (self.current_frame_index + 1) % self.frames_per_direction
                    self.texture = self.frames[self.current_frame_index + 4]
                case [1, 1]:
                    self.current_frame_index = (self.current_frame_index + 1) % self.frames_per_direction
                    self.texture = self.frames[self.current_frame_index + 8]
                case [-1, 1]:
                    self.current_frame_index = (self.current_frame_index + 1) % self.frames_per_direction
                    self.texture = self.frames[self.current_frame_index + 12]
                    
                    
class Guest(Character):
    """ Class for all NPCs, inherited from the Character class, they move in random directions for random intervals
    and move towards the exit when hit"""
    def __init__(self, sprite_frames, scale, frame_duration, x, y, player, wall_list):
        super().__init__(sprite_frames, scale, frame_duration, x, y)
        #self.path = []
        #self.current_point = 0
        self.player = player
        self.speed = 3
        self.to_move_x = 0
        self.to_move_y = 0
        self.target_x = 0
        self.target_y = 0
        self.wait_time = 0
        self.hurtbox_radius = 300
        self.change_direction()
        self.wall_list = wall_list
        self.walking_to_exit = False
    
        
    def update(self, delta_time: float = 1 / 60):
        
        if not self.walking_to_exit:

            self.wait_time -= delta_time
    
            if self.wait_time <= 0:
                hitting_wall = arcade.check_for_collision_with_list(self, self.wall_list)
                if hitting_wall:
                    for wall in hitting_wall:
                        diff_x = (wall.center_x - self.center_x) or 1e-6
                        diff_y = (wall.center_y - self.center_y) or 1e-6
                        wall_direction_x = diff_x / abs(diff_x)
                        wall_direction_y = diff_y / abs(diff_y)
                        self.walk_away_from_wall(wall_direction_x, wall_direction_y)
                self.wander()

    
    def walk_away_from_wall(self, x_direction, y_direction):
        """walk in the opposite direction of the wall"""
        # set direction to the opposite of the wall's direction
        self.direction[0] = -x_direction
        self.direction[1] = -y_direction
        # set random distance
        self.to_move_x = random.randint(200, 500)
        self.to_move_y = random.randint(200, 500)
        # make new target position based on current_location, direction and distance
        self.target_x = self.to_move_x * self.direction[0] + self.center_x
        self.target_y = self.to_move_y * self.direction[1] + self.center_y
    
    
    def change_direction(self):
        """change wandering direction randomly"""
        # set random direction
        self.direction[0] = random.choice([-1,1])
        self.direction[1] = random.choice([-1,1])
        # set random distance
        self.to_move_x = random.randint(200, 500)
        self.to_move_y = random.randint(200, 500)
        # make new target position based on current location, random direction and distance
        self.target_x = self.to_move_x * self.direction[0] + self.center_x
        self.target_y = self.to_move_y * self.direction[1] + self.center_y        
    

    def change_wait_time(self):
        """ set how long the Guest stays idle for"""
        # random time between 1 and 4 seconds
        self.wait_time = random.uniform(1,4)
        # change guest direction to default so it looks better
        self.direction = [-1,-1]
    

    def wander(self):
        """ wander towards target position"""
        if abs(self.center_x - self.target_x) < 10:
            if abs(self.center_y - self.target_y) < 10:
                self.change_wait_time()
                self.change_direction()
            else:
                self.center_y += self.speed * self.direction[1]
            
        elif abs(self.center_y - self.target_y) < 10:
            if abs(self.center_x - self.target_x) < 10:
                self.change_wait_time()
                self.change_direction()
            else:
                self.center_x += self.speed * self.direction[0]
        else:
            self.center_x += self.speed * self.direction[0]
            self.center_y += self.speed * self.direction[1]
    
    
    def is_hit(self):
        self.walking_to_exit = True
        
        
    def hurtbox_in_range(self):
        return self.check_distance()[0] < self.hurtbox_radius
        
        
    def check_distance(self):
        dx = self.player.center_x - self.center_x
        dy = self.player.center_y - self.center_y
        distance = math.sqrt(dx**2 + dy**2)
        
        return [distance, dx, dy]
    
    #def set_path(self, path):
        #self.path = path
        #self.current_point = 0
        
        
    #def update(self):
        #if self.current_point < len(self.path):
            #target_x, target_y = self.path[self.current_point]
            
            #dx = target_x - self.center_x
            #dy = target_y - self.center_y
            #distance = math.sqrt(dx**2 + dy**2)
            
            #if distance < self.speed:
                #self.current_point += 1
            #else:
                #self.center_x += (dx / distance) * self.speed
                #self.center_y += (dy / distance) * self.speed


class Librarian(Guest):
    """ Class for hostile NPCs, inherited from the Guest class, they will chase the player when the player is nearby"""
    
    def __init__(self, sprite_frames, scale, frame_duration, x, y, player, wall_list):
        super().__init__(sprite_frames, scale, frame_duration, x, y, player, wall_list)
        self.speed = 3
        self.detection_radius = 500
        self.hit_timer = 0.2
        self.hit_player = False

    def update(self, delta_time : float = 1/60):
        if not self.walking_to_exit:
            distances = self.check_distance()
            hitting_wall = arcade.check_for_collision_with_list(self, self.wall_list)
            if distances[0] < self.detection_radius:
                if not hitting_wall:
                    self.center_x += (distances[1] / distances[0]) * self.speed * 2
                    self.center_y += (distances[2] / distances[0]) * self.speed * 2
    
            else:
                self.wait_time -= delta_time
                if self.wait_time <= 0:
                    if hitting_wall:
                        for wall in hitting_wall:
                            diff_x = (wall.center_x - self.center_x) or 1e-6
                            diff_y = (wall.center_y - self.center_y) or 1e-6
                            direction_x = -diff_x / abs(diff_x)
                            direction_y = -diff_y / abs(diff_y)
                            self.walk_away_from_wall(direction_x, direction_y)
                    self.wander()
                
                
    def player_hit(self, delta_time : float = 1/60):
        """check if librarian is hitting player, 
        only returns True if they have been colliding for a while to give player time to dodge"""
        hitting_player = arcade.check_for_collision(self, self.player)
        if hitting_player:
            self.hit_timer -= delta_time
            if self.hit_timer <= 0:
                self.hit_timer = 0.4
                return True
        else:
            self.hit_timer = 0.4
        
        return False


class MyGame(arcade.View):
    """ Our custom Window Class"""
    
    def __init__(self):
        """Initializer"""
        # Call the parent class initializer
        super().__init__()
        
        self.window.set_mouse_visible(False)
        
        # Variables that will have sprite lists
        self.player_list = None
        self.guest_list = None
        self.librarian_list = None
        self.wall_list = None
        self.floor_list = None
        self.hotbar_sprite_list = None
        self.used_fire = False
        self.fire_list = None
        self.small_pud_list = None
        self.cursor_sprite = None
        self.restart_sprite = None
        self.restart_bgs = None

        #setup hotbar
        self.hotbar = ["hand", "mop", "bucket", "torch" , "shovel"]
        self.current_item = 0
        
        # Set up the player info
        self.player = None
        LEVEL = 1
        self.setup()
        
        self.bucket = None
        
        self.physics_engine = None
        
        self.camera_sprites = arcade.Camera(SCREEN_WIDTH, SCREEN_HEIGHT)
        self.camera_gui = arcade.Camera(SCREEN_WIDTH, SCREEN_HEIGHT)
        
        self.mouse_x = 0
        self.mouse_y = 0

        self.burning_sound = None
        self.library_sound = None


    def on_show(self):
        self.setup()


    def setup(self):
        """Set up the game and initialize the variables. """
        
        self.library_sound = arcade.play_sound(BGM)

        # find map
        match LEVEL:
            case 1:
                print("level1")
                map_name = "data/maps/level6.tmj"
                with open('data/maps/graph6.json') as json_file:
                    data = json.load(json_file)
                    print(data)
            case 2:
                print("level2")
                map_name = "data/maps/level6.tmj"
                with open('data/maps/graph6.json') as json_file:
                    data = json.load(json_file)
                    print(data)

        start_node = data["nodes"][0]
        end_node = data["nodes"][len(data["nodes"]) - 1]


        self.lost_game = False
        self.used_fire = False
        self.time_left = TIME_LIMIT
        self.fire_spread_time = 0
        self.hovered_guests = []
        self.hovered_puddles = []
        self.correct_tool = False
        self.guests_hit = 0
        # Sprite lists
        self.player_list = arcade.SpriteList()
        self.guest_list = arcade.SpriteList()
        self.librarian_list = arcade.SpriteList()
        
        # Load textures for animation
        self.player_textures = arcade.load_spritesheet("data\sprites\player.png", sprite_width = 32, sprite_height = 64, columns = 4, count = 16)
        player_pos = self.tile_to_pixel(start_node["x"], start_node["y"], 32 * TILE_SCALING, 32 * TILE_SCALING, data["dimensions"][1])
        print(player_pos)
        self.player = Character(self.player_textures, SPRITE_SCALING_PLAYER, 0.2, player_pos[0], player_pos[1])
        
        # Set up the player
        self.player_list.append(self.player)
        
        # set up guests
        self.guest_textures = arcade.load_spritesheet("data\sprites\guest.png", sprite_width = 32, sprite_height = 64, columns = 4, count = 16)
        self.guest = Guest(self.guest_textures, SPRITE_SCALING_PLAYER, 0.2, 896, 4964, self.player, self.wall_list)
        self.guest_list.append(self.guest)
        
        # set up librarians
        self.librarian_textures = arcade.load_spritesheet("data\sprites\librarian.png", sprite_width = 32, sprite_height = 64, columns = 4, count = 16)
        self.librarian = Librarian(self.librarian_textures, SPRITE_SCALING_PLAYER, 0.2, 1200, 4964, self.player, self.wall_list)
        self.librarian_list.append(self.librarian)
        
        self.water_level = 0        
        

        # Read in the tiled map
        self.tile_map = arcade.load_tilemap(map_name, scaling=TILE_SCALING)
        
        # Set wall SpriteList and any others that you have.
        self.wall_list = self.tile_map.sprite_lists["Room Walls"]
        self.floor_list = self.tile_map.sprite_lists["Room Floors"]
        self.cor_wall_list = self.tile_map.sprite_lists["Corridor Walls"]
        self.cor_floor_list = self.tile_map.sprite_lists["Corridor Floors"]
        self.puddle_list = self.tile_map.sprite_lists["Puddles"]
        
        self.cursor_sprite = arcade.Sprite()
        
        self.cursor_textures = arcade.load_spritesheet("data/sprites/cursor.png", sprite_width = 32, sprite_height = 32, columns = 5, count = 5)
        
        self.cursor_sprite.texture = self.cursor_textures[0]
        
        self.burnable_coords = []
        self.burnable_tiles = []
        self.wall_coords = []
        self.wall_tiles = []
        self.floor_coords = []
        self.floor_tiles = []

        for wall in self.wall_list:
            self.wall_coords.append(wall.position)
            self.wall_tiles.append(self.pixel_to_tile(wall.position[0], wall.position[1], 32 * TILE_SCALING , 32 * TILE_SCALING , MAP_HEIGHT))
        
        for cor_wall in self.cor_wall_list:
            self.wall_coords.append(cor_wall.position)
            self.wall_tiles.append(self.pixel_to_tile(cor_wall.position[0], cor_wall.position[1], 32 * TILE_SCALING , 32 * TILE_SCALING , MAP_HEIGHT))
        
        for floor in self.floor_list:
            self.floor_coords.append(floor.position)
            self.floor_tiles.append(self.pixel_to_tile(floor.position[0], floor.position[1], 32 * TILE_SCALING , 32 * TILE_SCALING , MAP_HEIGHT))
        
        for cor_floor in self.floor_list:
            self.floor_coords.append(cor_floor.position)
            self.floor_tiles.append(self.pixel_to_tile(cor_floor.position[0], cor_floor.position[1], 32 * TILE_SCALING , 32 * TILE_SCALING , MAP_HEIGHT))
        
        self.burnable_coords = self.wall_coords + self.floor_coords
        self.burnable_tiles = self.wall_tiles + self.floor_tiles

        self.puddle_coords = {}
        
        for puddle in self.puddle_list:
            self.puddle_coords[puddle] = [puddle.position, self.pixel_to_tile(puddle.position[0], puddle.position[1], 32 * TILE_SCALING , 32 * TILE_SCALING , MAP_HEIGHT)]

        self.tiles_on_fire = []
        
        self.initial_puddles = len(self.puddle_list)
        
        # create hotbar sprites
        self.hotbar_sprite_list = arcade.SpriteList()
        hotbar_sprites = arcade.load_spritesheet("data/sprites/hotbar_items.png", sprite_width = 64, sprite_height = 64, columns = 5, count = 25)

        mop = arcade.Sprite(texture=hotbar_sprites[1], scale = SPRITE_SCALING_HOTBAR, center_x = 735, center_y = 150)
        self.hotbar_sprite_list.append(mop)
        
        self.buckets_textures = [hotbar_sprites[2], hotbar_sprites[7], hotbar_sprites[12], hotbar_sprites[17], hotbar_sprites[22]]
        self.bucket = arcade.Sprite(texture=self.buckets_textures[0], scale = SPRITE_SCALING_HOTBAR, center_x = 840, center_y = 150)
        self.hotbar_sprite_list.append(self.bucket)
        
        self.torch_textures = [hotbar_sprites[3], hotbar_sprites[8]]
        self.torch = arcade.Sprite(texture=self.torch_textures[0], scale = SPRITE_SCALING_HOTBAR, center_x = 945, center_y = 150)
        self.hotbar_sprite_list.append(self.torch)
        
        shovel = arcade.Sprite(texture=hotbar_sprites[4], scale = SPRITE_SCALING_HOTBAR, center_x = 1050, center_y = 150)
        self.hotbar_sprite_list.append(shovel)
        
        # create restart button sprites
        self.restart_sprite = arcade.Sprite(
            texture=arcade.load_spritesheet("data/sprites/restart_button.png", sprite_width = 64, sprite_height = 64, columns = 1, count = 1)[0],
            scale = 0.7, center_x = 1500, center_y = 935)
        
        # restart button
        self.restart_bgs = arcade.SpriteList()
        self.restart_texture = arcade.load_spritesheet("data/sprites/restart_bg.png", sprite_width = 190, sprite_height = 60, columns = 1, count = 2)
        self.restart_normal = arcade.Sprite(texture = self.restart_texture[0], center_x = 1560, center_y = 935)
        self.restart_bgs.append(self.restart_normal)
        
        # initialize fire & small water puddle sprites
        self.fire_list = arcade.SpriteList()
        self.fire_sprites = arcade.load_spritesheet("data/sprites/fire.png", sprite_width = 32, sprite_height = 32, columns = 4, count = 16)
        
        self.small_pud_list = arcade.SpriteList()
        self.small_pud_sprites = arcade.load_spritesheet("data/sprites/puddles.png", sprite_width = 32, sprite_height = 32, columns = 10, count = 50)
        
        self.slowed_fire_tiles = {}
        self.puddles_touching_fire = []
        self.slowed_fire_timer = {}
        
        self.percent_level_burnt = 0
        
        # Set the background color to what is specified in the map
        if self.tile_map.background_color:
            arcade.set_background_color(self.tile_map.background_color)
            
        # Keep player from running through the wall_list layer
        self.physics_engine = arcade.PhysicsEnginePlatformer(
            self.player, self.wall_list, gravity_constant=0
        )
        
        
    def on_draw(self):
        
        # select camera to use to draw all sprites
        self.camera_sprites.use()
        
        arcade.start_render()

        self.wall_list.draw(pixelated = True)
        self.floor_list.draw(pixelated = True)
        self.cor_floor_list.draw(pixelated = True)
        self.cor_wall_list.draw(pixelated = True)
        self.puddle_list.draw(pixelated = True)
        self.small_pud_list.draw(pixelated = True)
        self.guest_list.draw(pixelated = True)
        self.librarian_list.draw(pixelated = True)
        self.player_list.draw(pixelated = True)
        self.fire_list.draw(pixelated = True)
        
        # select separate camera to create GUI
        self.camera_gui.use()
        
        # draw hotbar 
        arcade.draw_rectangle_filled(SCREEN_WIDTH/2, 150, 530, 110, (200, 200, 200))
        
        arcade.draw_rectangle_filled(630 + self.current_item * 105, 150, 110, 110, (230,190,10))
        
        for i in range(5):
            
            location = SCREEN_WIDTH/2 - 210
        
            arcade.draw_rectangle_filled(location + i*105, 150, 100, 100, (20, 20, 20))
        
        # display text
        arcade.draw_text(f"Percentage of level burnt: {self.percent_level_burnt:.2f}%", 30, 990, arcade.color.WHITE, 25)
        arcade.draw_text(f"Puddles Wiped: {int((self.initial_puddles - len(self.puddle_list)))} / {int(self.initial_puddles)}", 30, 930, arcade.color.WHITE, 25)
        arcade.draw_text(f"Time Left: {self.time_left:.2f} s", 1380, 980, arcade.color.WHITE, 25)
        
        # draw restart button
        self.restart_bgs.draw(pixelated = True)
        arcade.draw_text("Restart", 1530, 920, arcade.color.WHITE, 25)
        self.restart_sprite.draw(pixelated = True)
        self.hotbar_sprite_list.draw(pixelated = True)
        
        # draw cursor sprite
        self.cursor_sprite.draw(pixelated = True)
        
        
    def update(self, delta_time):
        """ Movement and game logic"""
        
        # Call update on all sprites
        self.player_list.update()
        self.player_list.update_animation()
        self.guest_list.update()
        self.guest_list.update_animation()
        self.librarian_list.update()
        self.librarian_list.update_animation()
        self.fire_list.update()
        self.small_pud_list.update()
        
        # check if librarian hit player
        for librarian in self.librarian_list:
            if librarian.player_hit():
                self.setup()
        
        # set bucket texture based on current water level
        self.bucket.texture = self.buckets_textures[self.water_level]
        
        # change cursor sprite depending on the item used
        self.cursor_sprite.texture = self.cursor_textures[self.current_item]
        
        # change torch sprite and start counting time when fire is used
        if self.used_fire:
            self.torch.texture = self.torch_textures[1]
            self.fire_spread_time -= delta_time
            self.time_left -= delta_time
        else:
            self.torch.texture = self.torch_textures[0]
        
            
        self.hotbar_sprite_list.update()
        # update physics
        self.physics_engine.update()
        
        # move screen
        self.scroll_to_player()
        
        # spread fire at random intervals
        if self.fire_spread_time <= 0:
            self.spread_fire()
                    
        # if the fire is touching a small puddle
        for puddle in self.puddles_touching_fire:
            # count the time where the fire slows down 
            self.slowed_fire_timer[puddle] -= delta_time
            # if time runs out for the puddle
            if self.slowed_fire_timer[puddle] <= 0:
                keys = [key for key, val in self.slowed_fire_tiles.items() if val == puddle]
                # remove puddle and stop slowing down the fire
                for key in keys:
                    key.remove_from_sprite_lists()
                    del self.slowed_fire_tiles[key]
        
        # update how much of the level got burnt
        self.percent_level_burnt = len(self.tiles_on_fire)/len(self.burnable_tiles) * 100
        
        # upate cursor 
        self.change_cursor()
        
        # player loses game if the fire takes too long to spread, doesn't restart game fully
        if self.time_left <= 0:
            self.lost_game = True
        
        if self.lost_game:
            self.game_over()
        
        
    def scroll_to_player(self):
        
        """Scroll the window to the player. If CAMERA_SPEED is 1, the camera will immediately move to the desire
        Anything between 0 and 1 will have the camera move to the location with pan."""
        
        position = Vec2(self.player.center_x - SCREEN_WIDTH / 2,
                        self.player.center_y - SCREEN_HEIGHT / 2)
        self.camera_sprites.move_to(position, CAMERA_SPEED)  
    
    
    def on_resize(self, width, height):
        """
        Resize window
        Handle the user grabbing the edge and resizing the window.
        """
        self.camera_sprites.resize(int(width), int(height))
        self.camera_gui.resize(int(width), int(height))    
    
    
    def on_key_press(self, key, modifiers):
        """Check to see which key is being pressed and move the player in the appropriate direction"""
        
        match key:
            case arcade.key.KEY_1:
                self.current_item = 0
            
            case arcade.key.KEY_2:
                self.current_item = 1
                
            case arcade.key.KEY_3:
                self.current_item = 2
            
            case arcade.key.KEY_4:
                self.current_item = 3
                
            case arcade.key.KEY_5:
                self.current_item = 4
            
            case arcade.key.ESCAPE:
                if self.burning_sound:
                    arcade.stop_sound(self.burning_sound)
                if self.library_sound:
                    arcade.stop_sound(self.library_sound)
                menu_view = MenuView(self)
                self.window.show_view(menu_view)
                
            case arcade.key.A:
                self.player.change_x = -MOVE_SPEED
                self.player.direction[0] = -1
                
            case arcade.key.D:
                self.player.change_x = MOVE_SPEED
                self.player.direction[0] = 1
                
            case arcade.key.W:
                self.player.change_y = MOVE_SPEED
                self.player.direction[1] = 1
                
            case arcade.key.S:
                self.player.change_y = -MOVE_SPEED
                self.player.direction[1] = -1
            
            case arcade.key.L:
                if LEVEL == 1 and self.guests_hit >= 2 and self.percent_level_burnt > 95:
                    LEVEL = 2
                else:
                    LEVEL = 1
                self.setup()
    
    
    def on_key_release(self, key, modifiers):
        """Called whenever a user releases a key"""
        pass
        match key:
            case arcade.key.A:
                self.player.change_x = 0
                    
            case arcade.key.D:
                self.player.change_x = 0          
                    
            case arcade.key.W:
                self.player.change_y = 0

            case arcade.key.S:
                self.player.change_y = 0


    def on_mouse_press(self, x, y, button, modifiers):
        """Handle mouse click events."""

        if button == arcade.MOUSE_BUTTON_LEFT:
            # restart game if cursor is hovering over restart button
            if self.restart_normal.texture == self.restart_texture[1]:
                self.setup()
            else:
                # if using hand
                if self.current_item == 0:
                    # if cursor is hovering over a guest
                    if self.hovered_guests:
                        for guest in self.hovered_guests:
                            # slap guest if the guest is in range
                            if guest.hurtbox_in_range():
                                self.hit_sound = arcade.play_sound(SLAP, looping = False)
                                guest.is_hit()
                                self.guests_hit += 1
                # if using mop
                elif self.current_item == 1:
                    
                    hovered_big_puddles = arcade.get_sprites_at_point((self.world_x,self.world_y), self.puddle_list)
                    
                    # if cursor is hovering over a large puddle
                    if hovered_big_puddles:

                        self.mop_sound = arcade.play_sound(MOP, looping = False)
                        
                        # fill up water bucket
                        if self.water_level <= 3:
                            self.water_level += 1
                        
                        # remove water puddle
                        for sprite in hovered_big_puddles:
                            sprite.remove_from_sprite_lists()
                            del self.puddle_coords[sprite]
                            
                    else:
                        hovered_small_pud = arcade.get_sprites_at_point((self.world_x,self.world_y), self.small_pud_list)
                        # if hovering over a small puddle
                        if hovered_small_pud:
                            if self.water_level <= 3:
                                self.water_level += 1                            
                            for sprite in hovered_small_pud:
                                sprite.remove_from_sprite_lists()
                                del self.slowed_fire_tiles[sprite]                            
        
        
                elif self.current_item == 2 and self.water_level >= 1 and self.clicked_tile not in self.slowed_fire_tiles.values() and self.clicked_tile in self.floor_tiles:
                    self.bucket_sound = arcade.play_sound(BUCKET, looping = False)
                    self.spawn_water(32 * TILE_SCALING * (self.clicked_tile[0] + 0.5) , 32 * TILE_SCALING * (MAP_HEIGHT - 0.5 - self.clicked_tile[1]))
                    self.water_level -= 1
                    
                elif self.current_item == 3 and self.used_fire == False and self.clicked_tile in self.burnable_tiles:
                    self.used_fire = True
                    arcade.stop_sound(self.library_sound)
                    self.match_sound = arcade.play_sound(MATCH, looping = False)
                    self.burning_sound = arcade.play_sound(FIRE)
                    if self.clicked_tile in self.burnable_tiles:
                        print(f"FIRE: {self.clicked_tile}", flush = True)
                        self.spawn_fire(32 * TILE_SCALING * (self.clicked_tile[0] + 0.5) , 32 * TILE_SCALING * (MAP_HEIGHT - 0.5 - self.clicked_tile[1]))
                        self.tiles_on_fire.append(self.clicked_tile)
                
                elif self.current_item == 4 and self.clicked_tile in self.wall_tiles:
                    self.shovel_sound = arcade.play_sound(SHOVEL, looping = False)
                    hovered_sprites = arcade.get_sprites_at_point((self.world_x,self.world_y), self.wall_list)
                    for sprite in hovered_sprites:
                        sprite.remove_from_sprite_lists()
                    self.burnable_tiles.remove(self.clicked_tile)


    def on_mouse_motion(self, x, y, dx, dy):
        """ updates where the cursor is on the world map instead of the screen"""
        self.mouse_x = x
        self.mouse_y = y
        self.cursor_sprite.center_x = x
        self.cursor_sprite.center_y = y
        # check if mouse if hovering over the restart button
        hovered_buttons = arcade.get_sprites_at_point((x,y), self.restart_bgs)
        if hovered_buttons:
            self.restart_normal.texture = self.restart_texture[1]
        else:
            self.restart_normal.texture = self.restart_texture[0]


    def pixel_to_tile(self, pixel_x, pixel_y, tile_width, tile_height, map_height):
        """
        Convert Arcade pixel coordinates to Tiled tile coordinates.
        """
        tile_x = int(pixel_x // tile_width)
        tile_y = int(map_height - (pixel_y // tile_height) - 1)
        return tile_x, tile_y


    def tile_to_pixel(self, tile_x, tile_y, tile_width, tile_height, map_height):
        """
        Convert Tiled tile coordinates to Arcade pixel coordinates
        """
        pixel_x = tile_x * tile_width + tile_width / 2
        pixel_y = (map_height - tile_y - 1) * tile_height + tile_height / 2
        return pixel_x, pixel_y

    def spawn_fire(self, x, y):
        """ spawns fire using inputted location"""
        fire = arcade.Sprite()
        fire.texture = self.fire_sprites[random.randrange(0, len(self.fire_sprites))]
        fire.scale = random.randrange(0, 7)
        fire.center_x = x + random.randrange(-15 * TILE_SCALING, 15 * TILE_SCALING)
        fire.center_y = y + random.randrange(-15 * TILE_SCALING, 15 * TILE_SCALING)
        self.fire_list.append(fire)
        # lose game if spawned fire in a large puddle
        colliding_bigpuddle = arcade.check_for_collision_with_list(fire, self.puddle_list)
        if colliding_bigpuddle:
            self.lost_game = True


    def spread_fire(self):
        """fire spreads of adjacent tiles"""
        self.fire_spread_time = random.random()
        if len(self.tiles_on_fire) > 0:
            tiles_to_add = []
            for tile in self.tiles_on_fire:
                up = (tile[0], tile[1] - 1)
                down = (tile[0], tile[1] + 1)
                left = (tile[0] - 1, tile[1])
                right = (tile[0] + 1, tile[1])
                tiles_to_add += up, down, left, right
                for item in tiles_to_add:
                    if item in self.slowed_fire_tiles.values():
                        tiles_to_add.remove(item)
                        if item not in self.puddles_touching_fire:
                            self.puddles_touching_fire.append(item)
                            
            for tile in tiles_to_add:
                if tile not in self.tiles_on_fire and tile in self.burnable_tiles:
                    self.tiles_on_fire.append(tile)
                    self.spawn_fire(32 * TILE_SCALING * (tile[0] + 0.5) , 32 * TILE_SCALING * (MAP_HEIGHT - 0.5 - tile[1]))

            
    def spawn_water(self, x, y):
        """spawn small water puddles"""
        water = arcade.Sprite()
        water.texture = self.small_pud_sprites[random.randrange(0, 4) * 10 + 2]
        water.scale = TILE_SCALING
        water.center_x = x
        water.center_y = y
        self.small_pud_list.append(water)
        self.slowed_fire_tiles[water] = self.clicked_tile
        self.slowed_fire_timer[self.clicked_tile] = 3.0
            
            
    def game_over(self):
        """ player failed to burn the whole level, stop all fire to let player retry"""
        self.lost_game = False
        self.fire_list.clear(deep=True)
        self.tiles_on_fire.clear()
        self.time_left = TIME_LIMIT
        self.used_fire = False
        arcade.stop_sound(self.burning_sound)
        self.library_sound = arcade.play_sound(BGM)
    
    
    def change_cursor(self):
        """ change cursor based on the item the player is holding and whether or not that item can be used"""
        self.world_x = self.mouse_x - SCREEN_WIDTH/2 + self.player.position[0]
        self.world_y = self.mouse_y - SCREEN_HEIGHT/2 + self.player.position[1]

        if self.current_item > 0:
            # only update clicked_tile if not using hand
            self.clicked_tile = self.pixel_to_tile(self.world_x, self.world_y, 32 * TILE_SCALING, 32 * TILE_SCALING, MAP_HEIGHT)

            if self.current_item == 1:
                self.hovered_puddles = arcade.get_sprites_at_point((self.world_x,self.world_y), self.puddle_list)
                self.hovered_puddles+= arcade.get_sprites_at_point((self.world_x,self.world_y), self.small_pud_list)
                if self.hovered_puddles:
                    self.correct_tool = True
                else:
                    self.correct_tool = False

            elif self.current_item == 2 and self.clicked_tile in self.floor_tiles and self.water_level >= 1:
                self.correct_tool = True

            elif self.current_item == 3 and self.clicked_tile in self.burnable_tiles:
                self.correct_tool = True

            elif self.current_item == 4 and self.clicked_tile in self.wall_tiles:
                self.correct_tool = True       
            else:
                self.correct_tool = False
        else:
            self.hovered_guests = arcade.get_sprites_at_point((self.world_x,self.world_y), self.guest_list)
            self.hovered_guests += arcade.get_sprites_at_point((self.world_x,self.world_y), self.librarian_list)
            if self.hovered_guests:
                for guest in self.hovered_guests:
                    if guest.hurtbox_in_range():
                        self.correct_tool = True
                    else:
                        self.correct_tool = False
            else:
                self.correct_tool = False

        if self.correct_tool:
            self.cursor_sprite.alpha = 255
            self.cursor_sprite.scale = 2
        else:
            self.cursor_sprite.alpha = 100
            self.cursor_sprite.scale = 1    


class MenuView(arcade.View):
    """Main menu view class."""

    def __init__(self, main_view):
        super().__init__()

        self.window.set_mouse_visible(True)

        # Changing background color of screen
        arcade.set_background_color((82, 55, 16))

        # Creating a UI MANAGER to handle the UI
        self.uimanager = arcade.gui.UIManager()

        self.game_view = main_view

    def on_hide_view(self):
        self.uimanager.disable()


    def on_show(self):
        self.uimanager.enable()
        self.create_menu()


    def create_menu(self):
        # Creating Button using UIFlatButton
        lvl1_button = arcade.gui.UIFlatButton(text="Level 1",
                                               width=200)
        lvl2_button = arcade.gui.UIFlatButton(text="Level 2",
                                                    width=200)
        lvl3_button = arcade.gui.UIFlatButton(text="Level 3",
                                                    width=200)
        lvl4_button = arcade.gui.UIFlatButton(text="Level 4",
                                               width=200)
        lvl5_button = arcade.gui.UIFlatButton(text="Level 5",
                                               width=200)
        lvl6_button = arcade.gui.UIFlatButton(text="Level 6",
                                               width=200)
        exit_button = arcade.gui.UIFlatButton(text="Quit",
                                               width=200)
        
        # Assigning our on_buttonclick() function
        #lvl1_button.on_click = self.on_buttonclick

        v_box = arcade.gui.UIBoxLayout(space_between=10)
        v_box.add(lvl1_button)
        v_box.add(lvl2_button)
        v_box.add(lvl3_button)
        v_box.add(lvl4_button)
        v_box.add(lvl5_button)
        v_box.add(lvl6_button)
        v_box.add(exit_button)

        anchor = arcade.gui.UIAnchorWidget(child=v_box, anchor_x="center", anchor_y="center")

        self.uimanager.add(anchor)


        @lvl1_button.event("on_click")
        def on_click_1(event):
            global LEVEL
            LEVEL = 1

            self.window.show_view(self.game_view)

        @lvl2_button.event("on_click")
        def on_click_2(event):
            global LEVEL
            LEVEL = 2
            self.window.show_view(self.game_view)

        @lvl3_button.event("on_click")
        def on_click_3(event):
            self.window.show_view(self.game_view)

        @lvl4_button.event("on_click")
        def on_click_4(event):
            self.window.show_view(self.game_view)

        @lvl5_button.event("on_click")
        def on_click_5(event):
            self.window.show_view(self.game_view)

        @lvl6_button.event("on_click")
        def on_click_6(event):
            self.window.show_view(self.game_view)
        
        @exit_button.event("on_click")
        def on_click_exit(event):
            arcade.close_window()

    def on_draw(self):
        arcade.start_render()
        
        # Drawing our ui manager
        self.uimanager.draw()

        arcade.draw_text("Library of Babel", SCREEN_WIDTH/3, SCREEN_HEIGHT*3/4, arcade.color.WHITE, 100, font_name = "times", bold = True, italic = True)


def main():
    """Main method """
    window = arcade.Window(SCREEN_WIDTH, SCREEN_HEIGHT, "Library of Babel")
    # Set location of window on screen
    window.set_location(0,0)
    game_view = MyGame()
    menu_view = MenuView(game_view)
    window.show_view(menu_view)
    arcade.run()
    
    
if __name__ == "__main__":
    main()
