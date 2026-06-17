import json
import random


ROOM_W      = 8  # default room width  (tiles)
ROOM_H      = 9   # default room height (tiles)
MAP_PADDING = 3    # empty tile border around the whole map
TILE_SIZE   = 32  # pixels per tile

# Tile IDs
TILE_EMPTY         = 0
TILE_ROOM_FLOOR    = 3
TILE_ROOM_WALL     = 4
TILE_CORRIDOR_FLOOR = 3
TILE_CORRIDOR_WALL  = 4


PUDDLE_TILESET_FIRSTGID = 17
 
PUDDLE_TILE_GIDS = [
    PUDDLE_TILESET_FIRSTGID,   # puddle variant 1
    PUDDLE_TILESET_FIRSTGID + 5,   # puddle variant 2
    PUDDLE_TILESET_FIRSTGID + 10,   # puddle variant 3
    PUDDLE_TILESET_FIRSTGID + 15,   # puddle variant 3
    PUDDLE_TILESET_FIRSTGID + 20,   # puddle variant 3
]
 
# Probability (0.0–1.0) that any given interior floor tile gets a puddle.
PUDDLE_DENSITY = 0.5

RANDOM_SEED = None

# Output paths
GRAPH_JSON_PATH = "graph.json"
MAP_PATH = "dungeon_map.tmj"


# graph data
nodes = [
    {"id": "A", "x": 2,  "y": 2},
    {"id": "B", "x": 14, "y": 2},
    {"id": "C", "x": 2,  "y": 12},
    {"id": "D", "x": 14, "y": 12},
    {"id": "E", "x": 28, "y": 7, "w": 9, "h": 7},
]

edges = [
    ["A", "B"],
    ["A", "C"],
    ["B", "D"],
    ["C", "D"],
    ["D", "E"],
]


for node in nodes:
    node.setdefault("w", ROOM_W)
    node.setdefault("h", ROOM_H)

node_by_id = {str(n["id"]): n for n in nodes}

with open(GRAPH_JSON_PATH, "w") as f:
    json.dump({"nodes": nodes, "edges": edges}, f, indent=2)
print(f"Exported graph '{GRAPH_JSON_PATH}'")


# Compute bounding box
min_x = min(n["x"] for n in nodes)
min_y = min(n["y"] for n in nodes)
max_x = max(n["x"] + n["w"] for n in nodes)
max_y = max(n["y"] + n["h"] for n in nodes)

# Offset so all content fits inside the padded map
origin_x = -min_x + MAP_PADDING
origin_y = -min_y + MAP_PADDING
map_w    = (max_x - min_x) + MAP_PADDING * 2
map_h    = (max_y - min_y) + MAP_PADDING * 2

corridor_floors = [TILE_EMPTY] * (map_w * map_h)
corridor_walls  = [TILE_EMPTY] * (map_w * map_h)
room_floors = [TILE_EMPTY] * (map_w * map_h)
room_walls = [TILE_EMPTY] * (map_w * map_h)
puddles = [TILE_EMPTY] * (map_w * map_h)

# A set of all tile coords occupied by any room (interior + wall border).
# Used to clip corridors so they don't overlap room tiles.
room_tiles = set()
room_wall_tiles = set()   # only the wall border tiles
room_interior_tiles = set()   # only the interior floor tiles


def idx(x, y):
    return y * map_w + x


def fill_rect(grid, x, y, w, h, tile_id):
    for row in range(y, y + h):
        for col in range(x, x + w):
            if 0 <= col < map_w and 0 <= row < map_h:
                grid[idx(col, row)] = tile_id


for n in nodes:
    rx = n["x"] + origin_x
    ry = n["y"] + origin_y
    rw = n["w"]
    rh = n["h"]
 
    # Interior floor (shrink by 1 on each side to leave room for walls)
    fill_rect(room_floors, rx + 1, ry + 1, rw - 2, rh - 2, TILE_ROOM_FLOOR)
 
    # Wall border (top, bottom, left, right edges)
    fill_rect(room_walls, rx,ry,rw, 1,  TILE_ROOM_WALL)  # top
    fill_rect(room_walls, rx,ry + rh - 1, rw, 1, TILE_ROOM_WALL)  # bottom
    fill_rect(room_walls, rx,ry,1,rh, TILE_ROOM_WALL)  # left
    fill_rect(room_walls, rx + rw - 1, ry, 1, rh, TILE_ROOM_WALL)  # right
 
    # Record every tile this room occupies (interior + walls)
    for row in range(ry, ry + rh):
        for col in range(rx, rx + rw):
            room_tiles.add((col, row))
 
 
def set_corridor_floor(x, y):
    """Place a corridor floor tile only if this cell is not part of any room."""
    if 0 <= x < map_w and 0 <= y < map_h and (x, y) not in room_tiles:
        corridor_floors[idx(x, y)] = TILE_CORRIDOR_FLOOR
 
 
def add_corridor_walls(corridor_cells):
    """
    Given a set of (x,y) corridor floor cells, paint wall tiles on any
    orthogonally adjacent empty cell that is not already a room tile or
    another corridor floor cell.
    """
    for (cx, cy) in corridor_cells:
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (1, -1), (-1, 1), (1, 1)]:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < map_w and 0 <= ny < map_h):
                continue
            if (nx, ny) in room_tiles:
                continue
            if (nx, ny) in corridor_cells:
                continue
            corridor_walls[idx(nx, ny)] = TILE_CORRIDOR_WALL
 
 
def room_exit_point(node, toward_x, toward_y):
    """
    Return the tile just outside the room wall that faces (toward_x, toward_y).
    The corridor starts here rather than at the room centre.
 
    Picks the wall face (top/bottom/left/right) that is closest to the
    target point, then steps one tile outside that wall.
    """
    rx = node["x"] + origin_x
    ry = node["y"] + origin_y
    rw = node["w"]
    rh = node["h"]
 
    cx = rx + rw // 2   # room centre in map-space
    cy = ry + rh // 2
 
    dx = toward_x - cx
    dy = toward_y - cy
 
    # Choose the dominant axis; break ties by whichever keeps corridors tidy
    if abs(dx) >= abs(dy):
        # Exit through left or right wall; use centre row of room
        if dx >= 0:
            return rx + rw, cy        # right wall, step one tile outside
        else:
            return rx - 1, cy         # left wall
    else:
        # Exit through top or bottom wall; use centre column of room
        if dy >= 0:
            return cx, ry + rh        # bottom wall
        else:
            return cx, ry - 1         # top wall
 
 
def draw_l_corridor(ax, ay, bx, by):
    """
    Draw an L-shaped corridor: horizontal from (ax,ay) to (bx,ay),
    then vertical from (bx,ay) to (bx,by).
    Skips any cell occupied by a room.
    Returns the set of corridor floor cells painted.
    """
    cells = set()
    for col in range(min(ax, bx), max(ax, bx) + 1):
        set_corridor_floor(col, ay)
        if (col, ay) not in room_tiles:
            cells.add((col, ay))
    for row in range(min(ay, by), max(ay, by) + 1):
        set_corridor_floor(bx, row)
        if (bx, row) not in room_tiles:
            cells.add((bx, row))
    return cells
 
 
# corridors
all_corridor_cells = set()

for id_a, id_b in edges:
    a = node_by_id.get(str(id_a))
    b = node_by_id.get(str(id_b))
    if not a or not b:
        print(f"  Warning: edge [{id_a}, {id_b}] references unknown node skipped.")
        continue
 
    # Centre of each room in map-space (used to pick exit face)
    cax = a["x"] + a["w"] // 2 + origin_x
    cay = a["y"] + a["h"] // 2 + origin_y
    cbx = b["x"] + b["w"] // 2 + origin_x
    cby = b["y"] + b["h"] // 2 + origin_y
 
    # Exit points: one tile outside the room wall facing the other room
    ax_exit, ay_exit = room_exit_point(a, cbx, cby)
    bx_exit, by_exit = room_exit_point(b, cax, cay)
 
    cells = draw_l_corridor(ax_exit, ay_exit, bx_exit, by_exit)
    all_corridor_cells |= cells

add_corridor_walls(all_corridor_cells)

ORTHO = [(-1, 0), (1, 0), (0, -1), (0, 1)]
 
for (wx, wy) in list(room_wall_tiles):
    for dx, dy in ORTHO:
        nx, ny = wx + dx, wy + dy
        if (nx, ny) in all_corridor_cells:
            # Replace wall tile with floor tile on the room_walls layer
            room_walls[idx(wx, wy)] = TILE_EMPTY
            room_floors[idx(wx, wy)] = TILE_ROOM_FLOOR
            break   # only need one adjacent corridor cell to open the wall


# paint puddles
for (px, py) in room_interior_tiles:
    if px % 2 != 0 or py % 2 != 0:
        continue   # skip odd coords to prevent puddles overlapping each other
    # Check all 4 cells the puddle would cover are interior floor
    footprint = [(px, py), (px+1, py), (px, py+1), (px+1, py+1)]
    if not all(cell in room_interior_tiles for cell in footprint):
        continue   # too close to a wall
    if random.random() < PUDDLE_DENSITY:
        puddles[idx(px, py)] = random.choice(PUDDLE_TILE_GIDS)



# build the tilemap
def make_layer(layer_id, name, data):
    return {
        "id": layer_id,
        "name": name,
        "type": "tilelayer",
        "x": 0, "y": 0,
        "width": map_w,
        "height": map_h,
        "visible": True,
        "opacity": 1,
        "data": data,
    }

tiled_map = {
    "version": "1.10",
    "tiledversion": "1.10.2",
    "type": "map",
    "orientation": "orthogonal",
    "renderorder": "right-down",
    "width": map_w,
    "height": map_h,
    "tilewidth": TILE_SIZE,
    "tileheight": TILE_SIZE,
    "infinite": False,
    "nextlayerid": 3,
    "nextobjectid": 1,
    # list of tilesets
    "tilesets": [
        {"firstgid": 1, "source": "data/sprites/room_tileset.tsx"},
        {"firstgid": PUDDLE_TILESET_FIRSTGID, "source": "data/sprites/puddles_tileset.tsx"}
    ],
    "layers": [
        make_layer(1, "Corridor Floors", corridor_floors),
        make_layer(2, "Corridor Walls", corridor_walls),
        make_layer(3, "Room Floors", room_floors),
        make_layer(4, "Room Walls", room_walls),
        make_layer(5, "Puddles", puddles),
    ],
}


# export map
with open(MAP_PATH, "w") as f:
    json.dump(tiled_map, f, indent=2)

print(f" Generated map: '{MAP_PATH}'")
print(f"Rooms: {len(nodes)} | Corridors: {len(edges)}")
