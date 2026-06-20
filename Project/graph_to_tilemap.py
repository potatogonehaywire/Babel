"""
export_graph.py

Exports your graph to:
  1. graph.json       — the raw graph data
  2. dungeon_map.tmj  — a ready-to-open Tiled map file

Layers produced (bottom to top):
  - Corridor Floors  : floor tiles for hallways
  - Corridor Walls   : wall tiles bordering hallways
  - Room Floors      : floor tiles inside rooms (including opened doorway cells)
  - Room Walls       : wall tiles on room borders
  - Puddles          : randomly placed puddle tiles (2x room tile size) inside rooms
"""

import json
import random
import generation_test

# ── Configuration ─────────────────────────────────────────────────────────────

ROOM_W      = 13
ROOM_H      = 12
MAP_PADDING = 3    # empty tile border around the whole map
TILE_SIZE   = 32   # pixels per tile (main tileset)

# ── Tile IDs — main tileset (1-based GID) ─────────────────────────────────────
TILE_EMPTY          = 0
TILE_ROOM_FLOOR     = 3
TILE_ROOM_WALL      = 4
TILE_CORRIDOR_FLOOR = 3
TILE_CORRIDOR_WALL  = 4

# ── Puddle tileset ────────────────────────────────────────────────────────────
# firstgid must be > last tile ID in main tileset (main has 4 tiles → use 5).
PUDDLE_TILESET_FIRSTGID = 5

# How many puddle tile variants are in the puddle tileset image.
PUDDLE_TILE_COUNT = 3

# GIDs of each puddle tile variant (auto-built from the above two values).
PUDDLE_TILE_GIDS = [PUDDLE_TILESET_FIRSTGID + i * 5 for i in range(PUDDLE_TILE_COUNT)]


# Probability (0–1) that an eligible 2×2 interior cell gets a puddle.
PUDDLE_DENSITY = 0.15


# Output paths
GRAPH_JSON_PATH = "data/maps/graph6.json"
MAP_PATH        = "data/maps/level6.tmj"


nodes , edges = generation_test.generate_graphs()
print(nodes, edges)

# ── Corridor helpers ──────────────────────────────────────────────────────────

def set_corridor_floor(x, y):
    if 0 <= x < map_w and 0 <= y < map_h and (x, y) not in room_tiles:
        corridor_floors[idx(x, y)] = TILE_CORRIDOR_FLOOR

def add_corridor_walls(corridor_cells):
    for (cx, cy) in corridor_cells:
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)]:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < map_w and 0 <= ny < map_h):
                continue
            if (nx, ny) in room_tiles:
                continue
            if (nx, ny) in corridor_cells:
                continue
            corridor_walls[idx(nx, ny)] = TILE_CORRIDOR_WALL


def room_exit_point(node, toward_x, toward_y):
    rx = node["x"] + origin_x
    ry = node["y"] + origin_y
    rw = node["w"]
    rh = node["h"]
    cx = rx + rw // 2
    cy = ry + rh // 2
    dx = toward_x - cx
    dy = toward_y - cy
    if abs(dx) >= abs(dy):
        # Horizontal exit — 2 tile tall doorway at centre row
        row = cy - 1   # shifted up so both rows clear the room interior
        if dx >= 0:
            return [(rx + rw - 1, row), (rx + rw - 1, row + 1)]   # right wall
        else:
            return [(rx, row), (rx, row + 1)]                      # left wall
    else:
        # Vertical exit — 1 tile wide doorway
        if dy >= 0:
            return [(cx, ry + rh - 1)]    # bottom wall
        else:
            return [(cx, ry)]             # top wall
        

def draw_l_corridor(ax, ay, bx, by):
    cells = set()
    # Horizontal segment — 2 tiles tall (ay and ay+1)
    for col in range(min(ax, bx), max(ax, bx) + 1):
        for row in [ay, ay + 1]:
            set_corridor_floor(col, row)
            if (col, row) not in room_tiles:
                cells.add((col, row))
    # Vertical segment — 1 tile wide (bx only)
    for row in range(min(ay, by), max(ay, by) + 1):
        set_corridor_floor(bx, row)
        if (bx, row) not in room_tiles:
            cells.add((bx, row))
    return cells


def room_centre(node):
    rx = node["x"] + origin_x
    ry = node["y"] + origin_y
    rw = node["w"]
    rh = node["h"]
    return (rx + rw // 2, ry + rh // 2)


# ── Normalise nodes ───────────────────────────────────────────────────────────

for node in nodes:
    node.setdefault("w", ROOM_W)
    node.setdefault("h", ROOM_H)

node_by_id = {str(n["id"]): n for n in nodes}


# ── Step 2: Build the tile grid ───────────────────────────────────────────────

min_x = min(n["x"] for n in nodes)
min_y = min(n["y"] for n in nodes)
max_x = max(n["x"] + n["w"] for n in nodes)
max_y = max(n["y"] + n["h"] for n in nodes)

origin_x = -min_x + MAP_PADDING
origin_y = -min_y + MAP_PADDING
map_w    = (max_x - min_x) + MAP_PADDING * 2
map_h    = (max_y - min_y) + MAP_PADDING * 2

# exported_nodes = [
#     {**n, "x": n["x"] + origin_x, "y": n["y"] + origin_y}
#     for n in nodes
# ]

exported_nodes = []
for n in nodes:
    cx, cy = room_centre(n)
    exported_nodes.append({
        **n,
        "x": n["x"] + origin_x,
        "y": n["y"] + origin_y,
        "centre_x": cx,
        "centre_y": cy,
    })

with open(GRAPH_JSON_PATH, "w") as f:
    json.dump({"room_center": exported_nodes, "edges": edges, "dimensions" : (map_w, map_h)}, f, indent=2)


corridor_floors = [TILE_EMPTY] * (map_w * map_h)
corridor_walls  = [TILE_EMPTY] * (map_w * map_h)
room_floors     = [TILE_EMPTY] * (map_w * map_h)
room_walls      = [TILE_EMPTY] * (map_w * map_h)
puddles         = [TILE_EMPTY] * (map_w * map_h)

room_tiles          = set()
room_wall_tiles     = set()
room_interior_tiles = set()

def idx(x, y):
    return y * map_w + x

def fill_rect(grid, x, y, w, h, tile_id):
    for row in range(y, y + h):
        for col in range(x, x + w):
            if 0 <= col < map_w and 0 <= row < map_h:
                grid[idx(col, row)] = tile_id

# ── Paint rooms ───────────────────────────────────────────────────────────────

for n in nodes:
    rx = n["x"] + origin_x
    ry = n["y"] + origin_y
    rw = n["w"]
    rh = n["h"]

    fill_rect(room_floors, rx + 1, ry + 1, rw - 2, rh - 2, TILE_ROOM_FLOOR)
    fill_rect(room_walls,  rx,          ry,          rw, 1,  TILE_ROOM_WALL)
    fill_rect(room_walls,  rx,          ry + rh - 1, rw, 1,  TILE_ROOM_WALL)
    fill_rect(room_walls,  rx,          ry,          1,  rh, TILE_ROOM_WALL)
    fill_rect(room_walls,  rx + rw - 1, ry,          1,  rh, TILE_ROOM_WALL)

    for row in range(ry, ry + rh):
        for col in range(rx, rx + rw):
            room_tiles.add((col, row))
            is_wall = (
                col == rx or col == rx + rw - 1 or
                row == ry or row == ry + rh - 1
            )
            if is_wall:
                room_wall_tiles.add((col, row))
            else:
                room_interior_tiles.add((col, row))

# ── Paint corridors ───────────────────────────────────────────────────────────

all_corridor_cells = set()
doorway_wall_tiles = set()   # wall tiles that need to be opened


for id_a, id_b in edges:
    a = node_by_id.get(str(id_a))
    b = node_by_id.get(str(id_b))
    if not a or not b:
        print(f"  Warning: edge [{id_a}, {id_b}] references unknown node – skipped.")
        continue

    cax = a["x"] + a["w"] // 2 + origin_x
    cay = a["y"] + a["h"] // 2 + origin_y
    cbx = b["x"] + b["w"] // 2 + origin_x
    cby = b["y"] + b["h"] // 2 + origin_y

    # Wall tiles to open — these are the starting points of the corridor
    # aw = room_exit_point(a, cbx, cby)
    # bw = room_exit_point(b, cax, cay)
    # doorway_wall_tiles.add(aw)
    # doorway_wall_tiles.add(bw)

    aw_tiles = room_exit_point(a, cbx, cby)
    bw_tiles = room_exit_point(b, cax, cay)
    doorway_wall_tiles.update(aw_tiles)
    doorway_wall_tiles.update(bw_tiles)
    

    # Step one tile outside the wall to find the first corridor cell.
    # Use the first tile in the list to determine direction.
    def step_outside(wall_tiles, node):
        wx, wy = wall_tiles[0]
        rx = node["x"] + origin_x
        ry = node["y"] + origin_y
        rw = node["w"]
        rh = node["h"]
        if wx == rx + rw - 1: return (wx + 1, wy)   # right wall → step right
        if wx == rx:           return (wx - 1, wy)   # left wall  → step left
        if wy == ry + rh - 1: return (wx, wy + 1)   # bottom     → step down
        return (wx, wy - 1)                          # top        → step up

    ax_start, ay_start = step_outside(aw_tiles, a)
    bx_end,   by_end   = step_outside(bw_tiles, b)

    cells = draw_l_corridor(ax_start, ay_start, bx_end, by_end)
    all_corridor_cells |= cells

add_corridor_walls(all_corridor_cells)

# ── Open doorway wall tiles ───────────────────────────────────────────────────
# Replace each wall tile at a corridor entrance with a floor tile.

for (wx, wy) in doorway_wall_tiles:
    if (wx, wy) in room_wall_tiles:
        room_walls[idx(wx, wy)]  = TILE_EMPTY
        room_floors[idx(wx, wy)] = TILE_ROOM_FLOOR

# ── Paint puddles ─────────────────────────────────────────────────────────────
# Puddle tiles are 2×2 room tiles. Place on even-coord interior cells only,
# ensuring the full 2×2 footprint stays within room interior tiles.

# for (px, py) in sorted(room_interior_tiles):
#     if px % 2 != 0 or py % 2 != 0:
#         continue
#     footprint = [(px, py), (px+1, py), (px, py+1), (px+1, py+1)]
#     if not all(cell in room_interior_tiles for cell in footprint):
#         continue
#     if random.random() < PUDDLE_DENSITY:
#         puddles[idx(px, py)] = random.choice(PUDDLE_TILE_GIDS)

# ── Step 3: Build the Tiled JSON map ─────────────────────────────────────────

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
    "nextlayerid": 6,
    "nextobjectid": 1,
    "tilesets": [
        {"firstgid": 1, "source": "../sprites/room_tileset.tsx"},
        {"firstgid" : PUDDLE_TILESET_FIRSTGID, "source": "../sprites/puddles_tileset.tsx"}
    ],
    "layers": [
        make_layer(1, "Corridor Floors", corridor_floors),
        make_layer(2, "Corridor Walls",  corridor_walls),
        make_layer(3, "Room Floors",     room_floors),
        make_layer(4, "Room Walls",      room_walls),
        make_layer(5, "Puddles",         puddles),
    ],
}

with open(MAP_PATH, "w") as f:
    json.dump(tiled_map, f, indent=2)

print(f"[2/2] Generated map   → '{MAP_PATH}'")
print(f"      Map size: {map_w} × {map_h} tiles")
print(f"      Rooms: {len(nodes)}  |  Corridors: {len(edges)}")
print(f"\nOpen '{MAP_PATH}' directly in Tiled.")