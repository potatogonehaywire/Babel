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
"""

import json
import random
import generation_test


def set_corridor_floor(x, y, map_w, map_h, room_tiles, corridor_floors):
    if 0 <= x < map_w and 0 <= y < map_h and (x, y) not in room_tiles:
        corridor_floors[y * map_w + x] = TILE_CORRIDOR_FLOOR

def add_corridor_walls(corridor_cells, map_w, map_h, room_tiles, corridor_walls):
    for (cx, cy) in corridor_cells:
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)]:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < map_w and 0 <= ny < map_h):
                continue
            if (nx, ny) in room_tiles:
                continue
            if (nx, ny) in corridor_cells:
                continue
            corridor_walls[ny * map_w + nx] = TILE_CORRIDOR_WALL


def room_exit_point(node, toward_x, toward_y, origin_x, origin_y):
    rx = node["x"] + origin_x
    ry = node["y"] + origin_y
    rw = node["w"]
    rh = node["h"]
    cx = rx + rw // 2
    cy = ry + rh // 2
    dx = toward_x - cx
    dy = toward_y - cy
    if abs(dx) >= abs(dy):
        # Horizontal exit — 3 tile tall doorway, centred on cy
        if dx >= 0:
            return [(rx + rw - 1, cy - 1),
                    (rx + rw - 1, cy),
                    (rx + rw - 1, cy + 1)]   # right wall
        else:
            return [(rx, cy - 1),
                    (rx, cy),
                    (rx, cy + 1)]             # left wall
    else:
        # Vertical exit — 2 tile wide doorway, centred on cx
        if dy >= 0:
            return [(cx, ry + rh - 1),
                    (cx + 1, ry + rh - 1)]   # bottom wall
        else:
            return [(cx, ry),
                    (cx + 1, ry)]             # top wall
        

def draw_l_corridor(ax, ay, bx, by, map_w, map_h, room_tiles, corridor_floors):
    cells = set()
    # Horizontal segment — 3 tiles tall (ay and ay+1)
    for col in range(min(ax, bx), max(ax, bx) + 1):
        for row in [ay - 1, ay, ay + 1]:
            set_corridor_floor(col, row, map_w, map_h, room_tiles, corridor_floors)
            if (col, row) not in room_tiles:
                cells.add((col, row))
    # Vertical segment — 2 tile wide (bx only)
    for row in range(min(ay, by), max(ay, by) + 1):
        for col in [bx, bx + 1]:
            set_corridor_floor(col, row, map_w, map_h, room_tiles, corridor_floors)
            if (col, row) not in room_tiles:
                cells.add((col, row))
    return cells


def room_centre(node, origin_x, origin_y):
    rx = node["x"] + origin_x
    ry = node["y"] + origin_y
    rw = node["w"]
    rh = node["h"]
    return (rx + rw // 2, ry + rh // 2)


def fill_rect(grid, x, y, w, h, tile_id, map_h, map_w):
    for row in range(y, y + h):
        for col in range(x, x + w):
            if 0 <= col < map_w and 0 <= row < map_h:
                grid[row * map_w + col] = tile_id


# Step one tile outside the wall to find the first corridor cell.
# Use the first tile in the list to determine direction.
def step_outside(wall_tiles, node, origin_x, origin_y):
        # Use the middle tile of the doorway to determine which wall face it's on
        wx, wy = wall_tiles[len(wall_tiles) // 2]
        rx = node["x"] + origin_x
        ry = node["y"] + origin_y
        rw = node["w"]
        rh = node["h"]
        if wx == rx + rw - 1: return (wx + 1, wy)
        if wx == rx:           return (wx - 1, wy)
        if wy == ry + rh - 1: return (wx, wy + 1)
        return (wx, wy - 1)                    # top        → step up


def make_layer(layer_id, name, data, map_w, map_h):
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


def make_map(level):
    nodes , edges = generation_test.generate_graphs(level)


    for node in nodes:
        node.setdefault("w", ROOM_W)
        node.setdefault("h", ROOM_H)

    node_by_id = {str(n["id"]): n for n in nodes}

    min_x = min(n["x"] for n in nodes)
    min_y = min(n["y"] for n in nodes)
    max_x = max(n["x"] + n["w"] for n in nodes)
    max_y = max(n["y"] + n["h"] for n in nodes)

    origin_x = -min_x + MAP_PADDING
    origin_y = -min_y + MAP_PADDING
    map_w    = (max_x - min_x) + MAP_PADDING * 2
    map_h    = (max_y - min_y) + MAP_PADDING * 2

    exported_nodes = []
    for n in nodes:
        cx, cy = room_centre(n, origin_x, origin_y)
        exported_nodes.append({
            **n,
            "x": n["x"] + origin_x,
            "y": n["y"] + origin_y,
            "centre_x": cx,
            "centre_y": cy,
        })

    with open("data/maps/graph" + str(level) + ".json", "w") as f:
        json.dump({"room_center": exported_nodes, "edges": edges, "dimensions" : (map_w, map_h)}, f, indent=2)


    corridor_floors = [TILE_EMPTY] * (map_w * map_h)
    corridor_walls  = [TILE_EMPTY] * (map_w * map_h)
    room_floors     = [TILE_EMPTY] * (map_w * map_h)
    room_walls      = [TILE_EMPTY] * (map_w * map_h)

    room_tiles          = set()
    room_wall_tiles     = set()
    room_interior_tiles = set()


    for n in nodes:
        rx = n["x"] + origin_x
        ry = n["y"] + origin_y
        rw = n["w"]
        rh = n["h"]

        fill_rect(room_floors, rx + 1, ry + 1, rw - 2, rh - 2, TILE_ROOM_FLOOR, map_h, map_w)
        fill_rect(room_walls,  rx,          ry,          rw, 1,  TILE_ROOM_WALL, map_h, map_w)
        fill_rect(room_walls,  rx,          ry + rh - 1, rw, 1,  TILE_ROOM_WALL, map_h, map_w)
        fill_rect(room_walls,  rx,          ry,          1,  rh, TILE_ROOM_WALL, map_h, map_w)
        fill_rect(room_walls,  rx + rw - 1, ry,          1,  rh, TILE_ROOM_WALL, map_h, map_w)

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


    all_corridor_cells = set()
    doorway_wall_tiles = set()


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

        aw_tiles = room_exit_point(a, cbx, cby, origin_x, origin_y)
        bw_tiles = room_exit_point(b, cax, cay, origin_x, origin_y)
        doorway_wall_tiles.update(aw_tiles)
        doorway_wall_tiles.update(bw_tiles)

        ax_start, ay_start = step_outside(aw_tiles, a, origin_x, origin_y)
        bx_end,   by_end   = step_outside(bw_tiles, b, origin_x, origin_y)

        cells = draw_l_corridor(ax_start, ay_start, bx_end, by_end, map_w, map_h, room_tiles, corridor_floors)
        all_corridor_cells |= cells

    add_corridor_walls(all_corridor_cells, map_w, map_h, room_tiles, corridor_walls)


    for (wx, wy) in doorway_wall_tiles:
        if (wx, wy) in room_wall_tiles:
            room_walls[wy * map_w + wx]  = TILE_EMPTY
            room_floors[wy * map_w + wx] = TILE_ROOM_FLOOR


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
            {"firstgid": 1, "source": "../sprites/room_tileset.tsx"}
        ],
        "layers": [
            make_layer(1, "Corridor Floors", corridor_floors, map_w, map_h),
            make_layer(2, "Corridor Walls",  corridor_walls, map_w, map_h),
            make_layer(3, "Room Floors",     room_floors, map_w, map_h),
            make_layer(4, "Room Walls",      room_walls, map_w, map_h),
        ],
    }

    with open("data/maps/level" + str(level) + ".tmj", "w") as f:
        json.dump(tiled_map, f, indent=2)


# room confiduration
ROOM_W      = 13
ROOM_H      = 12
MAP_PADDING = 1
TILE_SIZE   = 32

# tile ids
TILE_EMPTY          = 0
TILE_ROOM_FLOOR     = 3
TILE_ROOM_WALL      = 4
TILE_CORRIDOR_FLOOR = 3
TILE_CORRIDOR_WALL  = 4



# def main():
#     make_map(4)

# main()