#This is the MAIN script of DUGA. This is where the main loop is located and this is where all resources are loaded.
#All the classes will be located at the bottom of this script.

import pygame
import math
import os
import pickle
import logging
import sys
#--Imports added by Mikel Musquiz for player prediction--
from os import listdir
from os.path import isfile, join
import json
import copy
from keras.models import model_from_json
import numpy as np

#-- Engine imports--
import SETTINGS
import PLAYER
import TEXTURES
import MAP
import RAYCAST
import SPRITES
import NPC
import LEVELS
import GUNS
import PATHFINDING
import TEXT
#-- Game imports --
import EFFECTS
import HUD
import ITEMS
import INVENTORY
import ENTITIES
import SEGMENTS
import GENERATION
import MENU
import MUSIC
import TUTORIAL
from GEOM import sort_distance
from EVENTS import add_event_single
from EVENTS import TIMER_PLAYTIME
from EVENTS import EVENT_NPC_UPDATE
from EVENTS import EVENT_PLAYER_INPUT
from EVENTS import EVENT_PLAYER_LOCATION_CHANGED
from EVENTS import EVENT_PLAYER_VIEW_CHANGED
from EVENTS import EVENT_RAY_CASTING_CALCULATED

SECONDS_IN_MINUTE = 60
MILLISECONDS_IN_SECOND = 1000.0

FOV_MIN = 10
FOV_MAX = 100

pygame.init()
pygame.font.init()
pygame.display.set_mode((1,1))

#Load resources
class Load:

    def load_resources(self):
        ID = 0
        current_texture = 0
        self.timer = 0
        for texture in TEXTURES.all_textures:
            if SETTINGS.texture_type[ID] == 'sprite':
                SETTINGS.texture_list.append(pygame.image.load(texture))
            else:
                SETTINGS.texture_list.append(Texture(texture, ID))
            ID += 1
        #Update the dictionary in SETTINGS
        for texture in SETTINGS.texture_list:
            SETTINGS.tile_texture.update({current_texture : texture})
            current_texture += 1

        #Mixer goes under here as well
        pygame.mixer.init()

        #Load custom settings
        with open(os.path.join('data', 'settings.dat'), 'rb') as settings_file:
            settings = pickle.load(settings_file)
            
        SETTINGS.fov = settings['fov']
        SETTINGS.sensitivity = settings['sensitivity']
        SETTINGS.volume = settings['volume']
        SETTINGS.music_volume = settings['music volume']
        SETTINGS.resolution = settings['graphics'][0]
        SETTINGS.render = settings['graphics'][1]
        SETTINGS.fullscreen = settings['fullscreen']

        #Load statistics
        with open(os.path.join('data', 'statistics.dat'), 'rb') as stats_file:
            stats = pickle.load(stats_file)

        SETTINGS.statistics = stats

    def get_canvas_size(self):
        SETTINGS.canvas_map_width = len(SETTINGS.levels_list[SETTINGS.current_level].array[0])*SETTINGS.tile_size
        SETTINGS.canvas_map_height = len(SETTINGS.levels_list[SETTINGS.current_level].array)*SETTINGS.tile_size
        SETTINGS.canvas_actual_width = SETTINGS.canvas_target_width
        SETTINGS.canvas_actual_height = SETTINGS.canvas_target_height
        SETTINGS.canvas_aspect_ratio = SETTINGS.canvas_actual_width / SETTINGS.canvas_actual_height
        SETTINGS.player_map_pos = SETTINGS.levels_list[SETTINGS.current_level].player_pos
        SETTINGS.player_pos[0] = int((SETTINGS.levels_list[SETTINGS.current_level].player_pos[0] * SETTINGS.tile_size) + SETTINGS.tile_size/2)
        SETTINGS.player_pos[1] = int((SETTINGS.levels_list[SETTINGS.current_level].player_pos[1] * SETTINGS.tile_size) + SETTINGS.tile_size/2)
        if len(SETTINGS.gun_list) != 0:
            for gun in SETTINGS.gun_list:
                gun.re_init()

    def load_entities(self):
        ENTITIES.load_guns()
        ENTITIES.load_npc_types()
        ENTITIES.load_item_types()

    def load_custom_levels(self):
        if not os.stat(os.path.join('data', 'customLevels.dat')).st_size == 0:
            with open(os.path.join('data', 'customLevels.dat'), 'rb') as file:
                custom_levels = pickle.load(file)
                
            for level in custom_levels:
                SETTINGS.clevels_list.append(LEVELS.Level(level))

        with open(os.path.join('data', 'tutorialLevels.dat'), 'rb') as file:
            tutorial_levels = pickle.load(file)

        for level in tutorial_levels:
            SETTINGS.tlevels_list.append(LEVELS.Level(level))

    def load_new_level(self):    
        #Remove old level info
        SETTINGS.npc_list = []
        SETTINGS.all_items = []
        SETTINGS.walkable_area = []
        SETTINGS.all_tiles = []
        SETTINGS.all_doors = []
        SETTINGS.all_solid_tiles = []
        SETTINGS.all_sprites = []
        
        #Retrieve new level info
        self.get_canvas_size()
        gameMap.__init__(SETTINGS.levels_list[SETTINGS.current_level].array)
        SETTINGS.player_rect.center = (SETTINGS.levels_list[SETTINGS.current_level].player_pos[0]*SETTINGS.tile_size, SETTINGS.levels_list[SETTINGS.current_level].player_pos[1]*SETTINGS.tile_size)
        SETTINGS.player_rect.centerx += SETTINGS.tile_size/2
        SETTINGS.player_rect.centery += SETTINGS.tile_size/2
        gamePlayer.real_x = SETTINGS.player_rect.centerx
        gamePlayer.real_y = SETTINGS.player_rect.centery

        if SETTINGS.shade and SETTINGS.levels_list[SETTINGS.current_level].shade:
            SETTINGS.shade_rgba = SETTINGS.levels_list[SETTINGS.current_level].shade_rgba
            SETTINGS.shade_visibility = SETTINGS.levels_list[SETTINGS.current_level].shade_visibility

        if SETTINGS.current_level > 0:
            SETTINGS.changing_level = False
            SETTINGS.player_states['fade'] = True
        else:
            SETTINGS.player_states['fade'] = True
            SETTINGS.player_states['black'] = True

        SETTINGS.player_states['title'] = True
                
        SETTINGS.walkable_area = list(PATHFINDING.pathfind(SETTINGS.player_map_pos, SETTINGS.all_tiles[-1].map_pos))
        gameMap.move_inaccessible_entities()
        ENTITIES.spawn_npcs()
        ENTITIES.spawn_items()
        # to force the rendering to refersh
        rotate_screen()
        player_moved()


#Texturing
class Texture:
    
    def __init__(self, file_path, ID):
        self.slices = []
        self.texture = pygame.image.load(file_path).convert()
        self.rect = self.texture.get_rect()
        self.ID = ID

        self.create_slices()

    def create_slices(self): # Fills list - Nothing else
        row = 0
        for row in range(self.rect.width):
            self.slices.append(row)
            row += 1


#Canvas
class Canvas:
    '''== Create game canvas ==\nwidth -> px\nheight -> px'''
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.res_width = 0
        if SETTINGS.mode == 1:
            if SETTINGS.original_aspect:
                self.width = int(SETTINGS.canvas_target_width / SETTINGS.resolution) * SETTINGS.resolution
            else:
                self.width = SETTINGS.canvas_target_width
            self.height = SETTINGS.canvas_target_height
            self.res_width = SETTINGS.canvas_actual_width

        if SETTINGS.fullscreen:
            if SETTINGS.original_aspect:
                self.window = pygame.display.set_mode((self.width, int(self.height + (self.height * 0.15))), pygame.FULLSCREEN)
            else:
                self.window = pygame.display.set_mode((self.width, self.height), pygame.FULLSCREEN)
        else:
            if SETTINGS.original_aspect:
                self.window = pygame.display.set_mode((self.width, int(self.height + (self.height * 0.15))))
            else:
                self.window = pygame.display.set_mode((self.width, self.height))
        self.canvas = pygame.Surface((self.width, self.height))
        
        pygame.display.set_caption("DUGA")

        self.shade = [pygame.Surface((self.width, self.height)).convert_alpha(),
                      pygame.Surface((self.width, self.height/1.2)).convert_alpha(),
                      pygame.Surface((self.width, self.height/2)).convert_alpha(),
                      pygame.Surface((self.width, self.height/4)).convert_alpha(),
                      pygame.Surface((self.width, self.height/8)).convert_alpha(),
                      pygame.Surface((self.width, self.height/18)).convert_alpha()]
        self.rgba = [SETTINGS.shade_rgba[0], SETTINGS.shade_rgba[1], SETTINGS.shade_rgba[2], int(min(255, SETTINGS.shade_rgba[3]*(50/SETTINGS.shade_visibility)))]

    def change_mode(self):
        if SETTINGS.mode == 1: #1 - 3D / 0 - 2D
            SETTINGS.mode = 0
            self.__init__(SETTINGS.canvas_actual_width, SETTINGS.canvas_target_height)
        else:
            SETTINGS.mode = 1
            self.__init__(self.res_width, SETTINGS.canvas_target_height)
        SETTINGS.switch_mode = False

    def draw(self):
        if SETTINGS.mode == 1:
            self.canvas.fill(SETTINGS.levels_list[SETTINGS.current_level].sky_color)
            self.window.fill(SETTINGS.BLACK)
            pygame.draw.rect(self.canvas, SETTINGS.levels_list[SETTINGS.current_level].ground_color, (0, self.height/2, self.width, self.height/2))

            if SETTINGS.shade:
                for i in range(len(self.shade)):
                    if i != 5:
                        self.shade[i].fill((self.rgba[0], self.rgba[1], self.rgba[2], self.rgba[3]))
                    else:
                        self.shade[i].fill((self.rgba[0], self.rgba[1], self.rgba[2], SETTINGS.shade_rgba[3]))
                    self.canvas.blit(self.shade[i], (0, self.height/2 - self.shade[i].get_height()/2))

        else:
            self.window.fill(SETTINGS.WHITE)


def render_screen(canvas):
    '''render_screen(canvas) -> Renders everything but NPC\'s'''
    #SETTINGS.rendered_tiles = []

    #Get sprite positions
    for sprite in SETTINGS.all_sprites:
        sprite.get_pos(canvas)

    #Sort zbuffer
    SETTINGS.zbuffer = sorted(SETTINGS.zbuffer, key=sort_distance, reverse=True)

    # prepare zbuffer transforming only what we need to:
    zbuffer_len = len(SETTINGS.zbuffer)
    for item_index, item in enumerate(SETTINGS.last_zbuffer):
        if item is None:
            continue
        if item.type == 'slice':
            if item_index < zbuffer_len:
                if SETTINGS.last_zbuffer[item_index] == SETTINGS.zbuffer[item_index]:
                    SETTINGS.zbuffer[item_index] = item

    #Render all items in zbuffer
    for item_index, item in enumerate(SETTINGS.zbuffer):
        render_zbuffer_item(canvas, item_index, item)

    #Draw weapon if it is there
    if SETTINGS.current_gun:
        SETTINGS.current_gun.draw(gameCanvas.canvas)
    elif SETTINGS.next_gun:
        SETTINGS.next_gun.draw(gameCanvas.canvas)

    #Draw Inventory and effects
    if SETTINGS.player_states['invopen']:
        gameInv.draw(gameCanvas.canvas)
    EFFECTS.render(gameCanvas.canvas)

    SETTINGS.last_zbuffer = SETTINGS.zbuffer
    SETTINGS.zbuffer = []

    #Draw HUD and canvas
    gameCanvas.window.blit(canvas, (SETTINGS.axes))
    gameHUD.render(gameCanvas.window)

    #Draw tutorial strings
    if SETTINGS.levels_list == SETTINGS.tlevels_list:
            tutorialController.control(gameCanvas.window)


def render_zbuffer_item(canvas, item_index, item):
    if item == None:
        pass
    elif item.type == 'slice':
        if item.slice is None:
            SETTINGS.zbuffer[item_index].prepare_slice()

        blit_dest = item.blit_dest
        canvas.blit(item.slice, blit_dest)

        if item.vh == 'v':
            # Make vertical walls slightly darker
            canvas.blit(item.dark_slice, blit_dest)
        if SETTINGS.shade:
            canvas.blit(item.shade_slice, blit_dest)

    else:
        if item.new_rect.right > 0 and item.new_rect.x < SETTINGS.canvas_actual_width and item.distance < (
            SETTINGS.render * SETTINGS.tile_size):
            item.draw(canvas)


def update_game_visual():
    # Update logic
    gamePlayer.control(gameCanvas.canvas)

    if SETTINGS.fov >= FOV_MAX:
        SETTINGS.fov = FOV_MAX
    elif SETTINGS.fov <= FOV_MIN:
        SETTINGS.fov = FOV_MIN

    if SETTINGS.switch_mode:
        gameCanvas.change_mode()

    # Render - Draw
    gameRaycast.calculate()


def draw_game_visual():
    gameCanvas.draw()

    if SETTINGS.mode == 1:
        render_screen(gameCanvas.canvas)

        # BETA
    #  beta.draw(gameCanvas.window)

    elif SETTINGS.mode == 0:
        gameMap.draw(gameCanvas.window)
        gamePlayer.draw(gameCanvas.window)

        for x in SETTINGS.raylines:
            pygame.draw.line(gameCanvas.window, SETTINGS.RED, (x[0][0] / 4, x[0][1] / 4), (x[1][0] / 4, x[1][1] / 4))
        SETTINGS.raylines = []

        for i in SETTINGS.npc_list:
            if i.rect and i.dist <= SETTINGS.render * SETTINGS.tile_size * 1.2:
                pygame.draw.rect(gameCanvas.window, SETTINGS.RED,
                                 (i.rect[0] / 4, i.rect[1] / 4, i.rect[2] / 4, i.rect[3] / 4))
            elif i.rect:
                pygame.draw.rect(gameCanvas.window, SETTINGS.DARKGREEN,
                                 (i.rect[0] / 4, i.rect[1] / 4, i.rect[2] / 4, i.rect[3] / 4))


def update_game_state():
    if SETTINGS.npc_list:
        for npc in SETTINGS.npc_list:
            if not npc.dead:
                npc.think()

    SETTINGS.ground_weapon = None
    for item in SETTINGS.all_items:
        item.update()

    for tile in SETTINGS.all_solid_tiles:
        tile.update()
    if menuController.current_type == 'main':
        print("exit???")

    if (SETTINGS.changing_level and SETTINGS.player_states['black']) or SETTINGS.player_states['dead']:
        if SETTINGS.current_level < len(SETTINGS.levels_list)-1 and SETTINGS.changing_level:
            SETTINGS.current_level += 1
            SETTINGS.statistics['last levels']
            gameLoad.load_new_level()
        
        elif (SETTINGS.current_level == len(SETTINGS.levels_list)-1 or SETTINGS.player_states['dead']) and gameLoad.timer < 4 and not SETTINGS.player_states['fade']:
            if not SETTINGS.player_states['dead'] and SETTINGS.current_level == len(SETTINGS.levels_list)-1 and text.string != 'YOU  WON':
                text.update_string('YOU  WON')
            elif SETTINGS.player_states['dead'] and text.string != 'GAME  OVER':
                text.update_string('GAME  OVER')
            text.draw(gameCanvas.window)
            if not SETTINGS.game_won:
                gameLoad.timer = 0
            SETTINGS.game_won = True
            gameLoad.timer += SETTINGS.dt
            
        #Reset for future playthroughs
        elif SETTINGS.game_won and gameLoad.timer >= 4:
            gameLoad.timer = 0
            SETTINGS.game_won = False
            menuController.current_type = 'main'
            menuController.current_menu = 'score'
            calculate_statistics()
            SETTINGS.menu_showing = True
            SETTINGS.current_level = 0

def calculate_statistics():
    #Update 'all' stats
    SETTINGS.statistics['all enemies'] += SETTINGS.statistics['last enemies']
    SETTINGS.statistics['all ddealt'] += SETTINGS.statistics['last ddealt']
    SETTINGS.statistics['all dtaken'] += SETTINGS.statistics['last dtaken']
    SETTINGS.statistics['all shots'] += SETTINGS.statistics['last shots']
    SETTINGS.statistics['all levels'] += SETTINGS.statistics['last levels']

    #Update 'best' stats
    if SETTINGS.statistics['best enemies'] < SETTINGS.statistics['last enemies']:
        SETTINGS.statistics['best enemies'] = SETTINGS.statistics['last enemies']
    if SETTINGS.statistics['best ddealt'] < SETTINGS.statistics['last ddealt']:
        SETTINGS.statistics['best ddealt'] = SETTINGS.statistics['last ddealt']
    if SETTINGS.statistics['best dtaken'] < SETTINGS.statistics['last dtaken']:
        SETTINGS.statistics['best dtaken'] = SETTINGS.statistics['last dtaken']
    if SETTINGS.statistics['best shots'] < SETTINGS.statistics['last shots']:
        SETTINGS.statistics['best shots'] = SETTINGS.statistics['last shots']
    if SETTINGS.statistics['best levels'] < SETTINGS.statistics['last levels']:
        SETTINGS.statistics['best levels'] = SETTINGS.statistics['last levels']
    #'last' statistics will be cleared when starting new game in menu.
    with open(os.path.join('data', 'statistics.dat'), 'wb') as saved_stats:
        pickle.dump(SETTINGS.statistics, saved_stats)


from GEOM import sort_atan


def rotate_screen():
    for tile in SETTINGS.all_solid_tiles:
        tile.atan = sort_atan(tile)

def player_moved():
    SETTINGS.rendered_tiles = [
        tile for tile in SETTINGS.all_solid_tiles if
        tile.calculate_render_visible() and
        SETTINGS.tile_visible[tile.ID] and
        (
            (abs(tile.atan) <= SETTINGS.fov)
            or
            (tile.distance <= SETTINGS.tile_size * 1.5)
        )
    ]
    # SETTINGS.all_solid_tiles = sorted(SETTINGS.all_solid_tiles, key=lambda x: (x.type, x.atan, x.distance))

def get_state():
    if SETTINGS.npc_list[0].dist_from_player != None:
        # Initialize the closest_npcs with fake npcs with infinite distance from the player
        npc_copy = copy.copy(SETTINGS.npc_list[0])
        npc_copy.dist_from_player = float('inf')
        closest_npcs = [npc_copy, npc_copy, npc_copy]
        
        # Get the three closest NPCs sorted by distance
        for npc in SETTINGS.npc_list:
            dist = npc.dist_from_player
            if closest_npcs[2].dist_from_player > dist and not npc.dead:
                if closest_npcs[0].dist_from_player > dist and not npc.dead:
                    closest_npcs[2] = closest_npcs[1]
                    closest_npcs[1] = closest_npcs[0]
                    closest_npcs[0] = npc
                elif closest_npcs[1].dist_from_player > dist and not npc.dead:
                    closest_npcs[2] = closest_npcs[1]
                    closest_npcs[1] = npc
                else:
                    closest_npcs[2] = npc
        state = {   'pl_speed': SETTINGS.player_states["cspeed"],
                    'pl_pos_x': SETTINGS.player_map_pos[0],
                    'pl_pos_y': SETTINGS.player_map_pos[1],
                    'pl_angle': SETTINGS.player_angle,
                    'pl_armor': SETTINGS.player_armor,
                    'pl_health': SETTINGS.player_health,
                    'gun_name': SETTINGS.current_gun.name,
                    'gun_reload': int(SETTINGS.current_gun.reload_busy),
                    'gun_mag': SETTINGS.current_gun.current_mag,
                    'gun_bullets': SETTINGS.held_ammo['bullet'],
                    'npc1_ID': closest_npcs[0].ID,
                    'npc1_name': closest_npcs[0].name,
                    'npc1_mind': closest_npcs[0].mind,
                    'npc1_state': closest_npcs[0].state,
                    'npc1_dist': closest_npcs[0].dist_from_player,
                    #'npc1_seen': closest_npcs[0].last_seen_player_position,
                    'npc2_ID': closest_npcs[1].ID,
                    'npc2_name': closest_npcs[1].name,
                    'npc2_mind': closest_npcs[1].mind,
                    'npc2_state': closest_npcs[1].state,
                    'npc2_dist': closest_npcs[1].dist_from_player,
                    'npc3_ID': closest_npcs[2].ID,
                    'npc3_name': closest_npcs[2].name,
                    'npc3_mind': closest_npcs[2].mind,
                    'npc3_state': closest_npcs[2].state,
                    'npc3_dist': closest_npcs[2].dist_from_player
                    }
    return state
            
def update_data(data):
        if SETTINGS.current_gun != None:
            #data.append(get_state())    
            update_last_player_states(process_state())
            

def process_state():
    state = get_state()
    parameters = SETTINGS.parameters
    nparam = 114
    state_processed = np.zeros([nparam])
    i = 0
    for param in parameters:
        if type(param).__name__ !='dict':
            state_processed[i] = state[param]
            i = i + 1
        else:
            param_name = list(param.keys())[0]            
            values = state[param_name]
            idx = param[param_name].index(values)
            state_processed[i+idx-1] = 1
            i = i + len(param[param_name])
    
    return state_processed
    
def update_last_player_states(actual_state):
    SETTINGS.last_player_states[1:] = SETTINGS.last_player_states[:-1]
    SETTINGS.last_player_states[0] = actual_state
        
def predict_player_state(model):
    SETTINGS.prediction = model.predict(SETTINGS.last_player_states)
    print("predicted")

    
#Main loop
def main_loop():
    game_exit = False
    clock = pygame.time.Clock()
    logging.basicConfig(filename = os.path.join('data', 'CrashReport.log'), level=logging.WARNING)

    pygame.time.set_timer(TIMER_PLAYTIME, int(SECONDS_IN_MINUTE * MILLISECONDS_IN_SECOND))
    
    # Getting the name of the newest data file to later save the new ones
    mypath = "../DUGA-master"
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for file_name in files:
        if file_name[:8] == "data_log":
            max_file = int(file_name[9])
            
        
    # loading the trained model to predict player's position
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    loaded_model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['accuracy'])
    
    SETTINGS.last_player_states = np.zeros([1,100,114])

    
    #Dictionary to log player state
    data = []
    
    while not game_exit:
        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or SETTINGS.quit_game:
                    print("exit game")
                    game_exit = True
                    menuController.save_settings()
                    calculate_statistics()
                    pygame.quit()
                    sys.exit(0)
                elif event.type == EVENT_PLAYER_VIEW_CHANGED:
                    if event.value != 0:
                        # todo figure how much work to do based on the event.value
                        rotate_screen()
                        player_moved()
                        update_data(data)
                        if SETTINGS.last_player_states[0,0,59] < 200 and SETTINGS.last_player_states[0,0,59] > 0:
                            predict_player_state(loaded_model)
                elif event.type == EVENT_PLAYER_LOCATION_CHANGED:
                    if SETTINGS.last_player_states[0,0,59] < 200 and SETTINGS.last_player_states[0,0,59] > 0:
                        predict_player_state(loaded_model)
                    update_data(data)
                    player_moved()
                elif event.type == TIMER_PLAYTIME:
                    if SETTINGS.play_seconds >= SECONDS_IN_MINUTE:
                        SETTINGS.statistics['playtime'] += 1
                        SETTINGS.play_seconds = 0
                    else:
                        SETTINGS.play_seconds += 1

                elif event.type == EVENT_NPC_UPDATE:
                    print(event)
                elif event.type == EVENT_RAY_CASTING_CALCULATED:
                    draw_game_visual()

            #Music
            musicController.control_music()
            
            if SETTINGS.menu_showing and menuController.current_type == 'main':
                gameCanvas.window.fill(SETTINGS.WHITE)
                menuController.control()

                #Load custom maps
                if SETTINGS.playing_customs:
                    SETTINGS.levels_list = SETTINGS.clevels_list
                    gameLoad.get_canvas_size()
                    gameLoad.load_new_level()

                #Load generated maps
                elif SETTINGS.playing_new:
                    mapGenerator.__init__()
                    mapGenerator.generate_levels(SETTINGS.glevels_amount, SETTINGS.glevels_size)
                    SETTINGS.levels_list = SETTINGS.glevels_list
                    gameLoad.get_canvas_size()
                    gameLoad.load_new_level()

                #Or.. If they are playing the tutorial
                elif SETTINGS.playing_tutorial:
                    SETTINGS.levels_list = SETTINGS.tlevels_list
                    gameLoad.get_canvas_size()
                    gameLoad.load_new_level()

            elif SETTINGS.menu_showing and menuController.current_type == 'game':
                menuController.control()

            else:
                update_game_state()
                update_game_visual()
#                # If there is a change of level or the game is won, write in the file
#                if data!=[] and ((SETTINGS.changing_level and SETTINGS.player_states['black']) or SETTINGS.player_states['dead'] or SETTINGS.game_won and gameLoad.timer >= 4):
#                    print('printing data in data_log_'+str(max_file+1+SETTINGS.current_level)+'.txt')
#                    with open('data_log_'+str(max_file+1+SETTINGS.current_level)+'.txt', 'w') as outfile:
#                        json.dump(data, outfile)
#                    data = []
#                    max_file = max_file + 1


        except Exception as e:
            print(e)
            menuController.save_settings()
            calculate_statistics()
            logging.warning("DUGA has crashed. Please send this report to MaxwellSalmon, so he can fix it.")
            logging.exception("Error message: ")
            print('data_log_'+str(max_file+1)+'.txt')
            print('printing data')
            with open('data_log_'+str(max_file+1)+'.txt', 'w') as outfile:
                json.dump(data, outfile)
            pygame.quit()
            sys.exit(0)

        # Update Game
        pygame.display.update()
        delta_time = clock.tick(SETTINGS.fps)
        SETTINGS.dt = delta_time / MILLISECONDS_IN_SECOND
        SETTINGS.cfps = int(clock.get_fps())
        #pygame.display.set_caption(str(SETTINGS.cfps))
        # allfps.append(clock.get_fps())

#Probably temporary object init
#SETTINGS.current_level = 5 #temporary
if __name__ == '__main__':
    gameLoad = Load()
    gameLoad.load_resources()
    gameLoad.load_entities()
    gameLoad.load_custom_levels()

    mapGenerator = GENERATION.Generator()
    mapGenerator.generate_levels(1,2)
    SETTINGS.levels_list = SETTINGS.glevels_list

    gameLoad.get_canvas_size()

    #Setup and classes

    text = TEXT.Text(0,0,"YOU  WON", SETTINGS.WHITE, "DUGAFONT.ttf", 48)
    beta = TEXT.Text(5,5,"DUGA  BETA  BUILD  V. 1.3", SETTINGS.WHITE, "DUGAFONT.ttf", 20)
    text.update_pos(SETTINGS.canvas_actual_width/2 - text.layout.get_width()/2, SETTINGS.canvas_target_height/2 - text.layout.get_height()/2)

    #Classes for later use
    gameMap = MAP.Map(SETTINGS.levels_list[SETTINGS.current_level].array)
    gameCanvas = Canvas(SETTINGS.canvas_map_width, SETTINGS.canvas_map_height)
    gamePlayer = PLAYER.Player(SETTINGS.player_pos)
    gameRaycast = RAYCAST.Raycast(gameCanvas.canvas, gameCanvas.window)
    gameInv = INVENTORY.inventory({'bullet': 150, 'shell':25, 'ferromag' : 50})
    gameHUD = HUD.hud()

    #More loading - Level specific
    gameLoad.load_new_level()

    #Controller classes
    menuController = MENU.Controller(gameCanvas.window)
    musicController = MUSIC.Music()
    tutorialController = TUTORIAL.Controller()

    #Run at last
    main_loop()

