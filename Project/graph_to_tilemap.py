"""
export_graph.py

Exports your graph to:
  1. graph.json       — the raw graph data
  2. dungeon_map.tmj  — a ready-to-open Tiled map file

Just open dungeon_map.tmj in Tiled. No Tiled scripting required.

Node format:
    Each node must have:
      - "id"  : unique string or int identifier
      - "x"   : tile-space X of the room's top-left corner
      - "y"   : tile-space Y of the room's top-left corner
      - "w"   : room width  in tiles  (optional, defaults to ROOM_W)
      - "h"   : room height in tiles  (optional, defaults to ROOM_H)

Edge format:
    Each edge is a pair [source_id, target_id].
"""

import json

# ── Configuration ─────────────────────────────────────────────────────────────

ROOM_W      = 6  # default room width  (tiles)
ROOM_H      = 7   # default room height (tiles)
MAP_PADDING = 3    # empty tile border around the whole map
TILE_SIZE   = 32  # pixels per tile

# Tile IDs (1-based; 0 = empty).
# Adjust to match your own tileset.
TILE_EMPTY    = 0
TILE_FLOOR    = 18
TILE_WALL     = 19
TILE_CORRIDOR = 18   # corridors use the same tile as floor

# Output paths
GRAPH_JSON_PATH = "graph.json"
MAP_PATH        = "dungeon_map.tmj"

# ── Your graph data ───────────────────────────────────────────────────────────
# Replace these with your actual nodes and edges.

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

# ── Normalise nodes ───────────────────────────────────────────────────────────

for node in nodes:
    node.setdefault("w", ROOM_W)
    node.setdefault("h", ROOM_H)

node_by_id = {str(n["id"]): n for n in nodes}

# ── Step 1: Export graph.json ─────────────────────────────────────────────────

with open(GRAPH_JSON_PATH, "w") as f:
    json.dump({"nodes": nodes, "edges": edges}, f, indent=2)
print(f"[1/2] Exported graph  → '{GRAPH_JSON_PATH}'")

# ── Step 2: Build the tile grid ───────────────────────────────────────────────

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

# Initialise two flat tile arrays (row-major, length = map_w * map_h)
corridors = [TILE_EMPTY] * (map_w * map_h)
rooms     = [TILE_EMPTY] * (map_w * map_h)

def idx(x, y):
    return y * map_w + x

def fill_rect(grid, x, y, w, h, tile_id):
    for row in range(y, y + h):
        for col in range(x, x + w):
            if 0 <= col < map_w and 0 <= row < map_h:
                grid[idx(col, row)] = tile_id

def hollow_rect(grid, x, y, w, h, floor_id, wall_id):
    fill_rect(grid, x, y, w, h, floor_id)             # interior floor
    fill_rect(grid, x,         y,         w, 1, wall_id)  # top
    fill_rect(grid, x,         y + h - 1, w, 1, wall_id)  # bottom
    fill_rect(grid, x,         y,         1, h, wall_id)  # left
    fill_rect(grid, x + w - 1, y,         1, h, wall_id)  # right

def draw_l_corridor(grid, ax, ay, bx, by, tile_id):
    """Horizontal from (ax,ay) to (bx,ay), then vertical to (bx,by)."""
    for col in range(min(ax, bx), max(ax, bx) + 1):   # horizontal segment
        if 0 <= col < map_w and 0 <= ay < map_h:
            grid[idx(col, ay)] = tile_id
    for row in range(min(ay, by), max(ay, by) + 1):   # vertical segment
        if 0 <= bx < map_w and 0 <= row < map_h:
            grid[idx(bx, row)] = tile_id

# Paint corridors
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
    draw_l_corridor(corridors, cax, cay, cbx, cby, TILE_CORRIDOR)

# Paint rooms (hollow rectangles)
for n in nodes:
    hollow_rect(
        rooms,
        n["x"] + origin_x,
        n["y"] + origin_y,
        n["w"], n["h"],
        TILE_FLOOR, TILE_WALL
    )

# ── Step 3: Build the Tiled JSON map structure ────────────────────────────────

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
    # list o f tilesets
    "tilesets": [
        {
            "firstgid": 1,
            "name": "RoomTileset",
            "tilewidth": TILE_SIZE,
            "tileheight": TILE_SIZE,
            "spacing": 0,
            "margin": 0,
            "tilecount": 20,
            "columns": 4,
            "imagewidth": TILE_SIZE * 4,
            "imageheight": TILE_SIZE * 5,
            # tileset image
            "image": "data/sprites/tile_textures.png",
        },
        {
            "firstgid": 2,
            "name": "RoomTileset",
            "tilewidth": TILE_SIZE * 2,
            "tileheight": TILE_SIZE * 2,
            "spacing": 0,
            "margin": 0,
            "tilecount": 20,
            "columns": 4,
            "imagewidth": TILE_SIZE * 2,
            "imageheight": TILE_SIZE * 10,
            "image": "data/sprites/puddles.png",
        }
    ],
    "layers": [
        {
            "id": 1,
            "name": "Floor",
            "type": "tilelayer",
            "x": 0, "y": 0,
            "width": map_w,
            "height": map_h,
            "visible": True,
            "opacity": 1,
            "data": corridors,
        },
        {
            "id": 2,
            "name": "Wall",
            "type": "tilelayer",
            "x": 0, "y": 0,
            "width": map_w,
            "height": map_h,
            "visible": True,
            "opacity": 1,
            "data": rooms,
        },
        {
            "id": 3,
            "name": "Puddle",
            "type": "tilelayer",
            "x": 0, "y": 0,
            "width": map_w,
            "height": map_h,
            "visible": True,
            "opacity": 1,
            "data": rooms,
        },
    ],
}

with open(MAP_PATH, "w") as f:
    json.dump(tiled_map, f, indent=2)

print(f"[2/2] Generated map   → '{MAP_PATH}'")
print(f"      Map size: {map_w} × {map_h} tiles")
print(f"      Rooms: {len(nodes)}  |  Corridors: {len(edges)}")
print(f"\nOpen '{MAP_PATH}' directly in Tiled.")