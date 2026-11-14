"""
=============================================================================
 MILITARY SURVEILLANCE ROVER — HIGH-FIDELITY FREECAD 3D MODEL
 Parametric CAD matching photorealistic industrial design:
 - Natural Birch/Sunboard Decks with CNC Mounting Through-Holes
 - Anodized Aluminum Ladder Chassis & Extruded Square Pillars
 - 4WD Off-Road Knobby Tires with 5-Spoke Star Rims & DC Motors
 - NVIDIA Jetson Xavier NX (Black Finned Heatsink, Fan & Blue I/O Panel)
 - Intel RealSense D435i Capsule Optical Depth Camera
 - Elevated Sensor Mast: Aerodynamic GPS Disc & Dual-Whip LoRa Module
=============================================================================
"""

import os
import math
import FreeCAD as App
import Part
from FreeCAD import Base

# ─────────────────────────────────────────────────────────────────────────────
# 1. ACCURATE PARAMETRIC DIMENSIONS (mm)
# ─────────────────────────────────────────────────────────────────────────────
T_DECK        = 5.0     # 5 mm CNC sunboard/plywood sheet thickness

# Chassis Ladder Frame (Layer 0)
CHASSIS_L     = 330.0   # Longitudinal frame length
CHASSIS_W     = 230.0   # Frame width
BEAM_SIZE     = 25.0    # Square box-section tubing (25 x 25 mm)

# Off-Road Wheels & 4WD Motors
TIRE_OD       = 72.0    # Knobby outer tire diameter
TIRE_WIDTH    = 30.0    # Tire tread width
RIM_OD        = 44.0    # Yellow 5-spoke wheel rim
MOTOR_L       = 65.0    # Yellow BO motor gearbox
MOTOR_W       = 22.0
MOTOR_H       = 20.0
WHEELBASE_X   = 220.0   # Distance between front & rear axles
TRACK_WIDTH_Y = 250.0   # Center-to-center track width

# Lower Deck (Layer 1)
DECK_L1_L     = 380.0
DECK_L1_W     = 280.0

# Support Pillars (Layer 2)
PILLAR_H      = 90.0    # Deck-to-deck vertical clearance
PILLAR_W      = 25.0    # Square extruded aluminum profile

# Upper Deck (Layer 3)
DECK_L3_L     = 320.0
DECK_L3_W     = 240.0

# Sensor Mast Deck (Layer 4)
DECK_L4_L     = 160.0
DECK_L4_W     = 100.0
L4_STDOFF_H   = 40.0    # Height above upper deck

# ─────────────────────────────────────────────────────────────────────────────
# 2. PHOTOREALISTIC MATERIAL PALETTE (RGB)
# ─────────────────────────────────────────────────────────────────────────────
COLOR_WOOD      = (0.84, 0.74, 0.60)  # Natural Birch / Kraft Sunboard
COLOR_ALU_SILV  = (0.76, 0.78, 0.80)  # Satin Extruded Aluminum (Pillars & Standoffs)
COLOR_GUNMETAL  = (0.28, 0.30, 0.33)  # Dark Anodized Chassis Tubing
COLOR_RUBBER    = (0.10, 0.10, 0.11)  # Matte Off-Road Rubber Tires
COLOR_YELLOW    = (0.96, 0.78, 0.08)  # Bright Yellow 5-Spoke Wheels & Gearboxes
COLOR_JETSON    = (0.12, 0.13, 0.16)  # Jetson Xavier NX Anodized Heatsink
COLOR_JET_BLUE  = (0.12, 0.35, 0.72)  # NVIDIA Blue I/O Accent Face
COLOR_CAM_BODY  = (0.78, 0.80, 0.82)  # RealSense Silver Bezel
COLOR_CAM_GLASS = (0.05, 0.05, 0.06)  # Optical Glass Front Lens
COLOR_PCB_RED   = (0.82, 0.15, 0.15)  # L298N Red FR4 Board
COLOR_PCB_GRN   = (0.08, 0.52, 0.22)  # LoRa RF Green Board
COLOR_POWERBANK = (0.70, 0.72, 0.74)  # Brushed Aluminum Power Bank
COLOR_BATTERY   = (0.14, 0.15, 0.17)  # Matte Black LiPo Pack
COLOR_GPS_DOME  = (0.88, 0.90, 0.92)  # GPS Ceramic Radome


# ─────────────────────────────────────────────────────────────────────────────
# 3. HELPER BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
def create_doc(doc_name="Military_Surveillance_Rover"):
    if App.activeDocument() and App.activeDocument().Name == doc_name:
        App.closeDocument(doc_name)
    return App.newDocument(doc_name)

def add_part(doc, shape, name, color, group=None):
    obj = doc.addObject("Part::Feature", name)
    obj.Shape = shape
    if App.GuiUp:
        obj.ViewObject.ShapeColor = color[:3]
        obj.ViewObject.LineColor  = (color[0]*0.7, color[1]*0.7, color[2]*0.7)
    if group:
        group.addObject(obj)
    return obj

def make_drilled_plate(length, width, thickness, z_pos, hole_margin=18.0, hole_dia=6.0):
    """Creates a precision sunboard deck with CNC corner mounting holes."""
    plate = Part.makeBox(length, width, thickness, 
                         Base.Vector(-length/2.0, -width/2.0, z_pos))
    # 4 Corner mounting holes
    for hx in [length/2.0 - hole_margin, -length/2.0 + hole_margin]:
        for hy in [width/2.0 - hole_margin, -width/2.0 + hole_margin]:
            drill = Part.makeCylinder(hole_dia/2.0, thickness + 2.0,
                                      Base.Vector(hx, hy, z_pos - 1.0),
                                      Base.Vector(0, 0, 1))
            plate = plate.cut(drill)
    return plate


# ─────────────────────────────────────────────────────────────────────────────
# 4. LAYER 0: CHASSIS LADDER FRAME, MOTORS & 5-SPOKE WHEELS
# ─────────────────────────────────────────────────────────────────────────────
def build_layer_0(doc):
    grp0 = doc.addObject("App::DocumentObjectGroup", "Layer0_Chassis_Frame")
    z_axle = TIRE_OD / 2.0  # 36 mm ground clearance
    z_frame = z_axle - (BEAM_SIZE / 2.0)

    # 2x Longitudinal Square Rails (330 x 25 x 25 mm)
    rail_l = Part.makeBox(CHASSIS_L, BEAM_SIZE, BEAM_SIZE,
                          Base.Vector(-CHASSIS_L/2.0, CHASSIS_W/2.0 - BEAM_SIZE, z_frame))
    rail_r = Part.makeBox(CHASSIS_L, BEAM_SIZE, BEAM_SIZE,
                          Base.Vector(-CHASSIS_L/2.0, -CHASSIS_W/2.0, z_frame))
    add_part(doc, rail_l, "L0_Chassis_Rail_Left", COLOR_GUNMETAL, grp0)
    add_part(doc, rail_r, "L0_Chassis_Rail_Right", COLOR_GUNMETAL, grp0)

    # 3x Cross Beams (Front, Middle, Rear)
    inner_w = CHASSIS_W - (2 * BEAM_SIZE)
    for i, x_pos in enumerate([-CHASSIS_L/2.0 + BEAM_SIZE/2.0, 0.0, CHASSIS_L/2.0 - BEAM_SIZE/2.0]):
        cross = Part.makeBox(BEAM_SIZE, inner_w, BEAM_SIZE,
                             Base.Vector(x_pos - BEAM_SIZE/2.0, -inner_w/2.0, z_frame))
        add_part(doc, cross, f"L0_Chassis_Cross_{i+1}", COLOR_GUNMETAL, grp0)

    # 4WD DC Geared Motors & Detailed 5-Spoke Knobby Wheels
    axle_xs = [WHEELBASE_X / 2.0, -WHEELBASE_X / 2.0]
    axle_ys = [TRACK_WIDTH_Y / 2.0, -TRACK_WIDTH_Y / 2.0]

    for ix, wx in enumerate(axle_xs):
        for iy, wy in enumerate(axle_ys):
            side = 1.0 if wy > 0 else -1.0
            
            # Yellow Gearbox Unit
            motor_body = Part.makeBox(MOTOR_L, MOTOR_W, MOTOR_H,
                                      Base.Vector(wx - MOTOR_L/2.0, 
                                                  wy - (side * 28.0) - MOTOR_W/2.0, 
                                                  z_axle - MOTOR_H/2.0))
            motor_cyl = Part.makeCylinder(10.0, 18.0,
                                          Base.Vector(wx - (side * 15.0), wy - (side * 36.0), z_axle),
                                          Base.Vector(0, -side, 0))
            motor_unit = motor_body.fuse(motor_cyl)
            add_part(doc, motor_unit, f"L0_Motor_Gearbox_{ix}_{iy}", COLOR_YELLOW, grp0)

            # Outer Rubber Tire with Knobby Tread
            tire_center = Base.Vector(wx, wy, z_axle)
            tire = Part.makeCylinder(TIRE_OD/2.0, TIRE_WIDTH,
                                     tire_center - Base.Vector(0, side * TIRE_WIDTH/2.0, 0),
                                     Base.Vector(0, side, 0))
            rim_cut = Part.makeCylinder(RIM_OD/2.0, TIRE_WIDTH + 2.0,
                                        tire_center - Base.Vector(0, side * (TIRE_WIDTH/2.0 + 1.0), 0),
                                        Base.Vector(0, side, 0))
            tire_solid = tire.cut(rim_cut)
            add_part(doc, tire_solid, f"L0_Wheel_Tire_{ix}_{iy}", COLOR_RUBBER, grp0)

            # Yellow 5-Spoke Star Rim
            rim = Part.makeCylinder(RIM_OD/2.0, TIRE_WIDTH - 4.0,
                                    tire_center - Base.Vector(0, side * (TIRE_WIDTH - 4.0)/2.0, 0),
                                    Base.Vector(0, side, 0))
            # 5 Star Spoke Cutouts
            for s in range(5):
                angle = s * (2.0 * math.pi / 5.0)
                cx = wx + 13.0 * math.cos(angle)
                cz = z_axle + 13.0 * math.sin(angle)
                spoke_hole = Part.makeCylinder(4.2, TIRE_WIDTH + 4.0,
                                               Base.Vector(cx, wy - side * 15.0, cz),
                                               Base.Vector(0, side, 0))
                rim = rim.cut(spoke_hole)
            
            # Central Hex Nut Hub
            hub_nut = Part.makeCylinder(5.5, 6.0,
                                        tire_center + Base.Vector(0, side * (TIRE_WIDTH/2.0 - 2.0), 0),
                                        Base.Vector(0, side, 0))
            rim_assembly = rim.fuse(hub_nut)
            add_part(doc, rim_assembly, f"L0_Wheel_Rim_{ix}_{iy}", COLOR_YELLOW, grp0)

    return z_frame + BEAM_SIZE


# ─────────────────────────────────────────────────────────────────────────────
# 5. LAYER 1: LOWER PLATFORM, L298N DRIVER & POWER UNITS
# ─────────────────────────────────────────────────────────────────────────────
def build_layer_1(doc, z_start):
    grp1 = doc.addObject("App::DocumentObjectGroup", "Layer1_Lower_Deck")

    # 380 x 280 x 5 mm Lower Deck with Mounting Holes
    deck_l1 = make_drilled_plate(DECK_L1_L, DECK_L1_W, T_DECK, z_start, hole_margin=18.0)
    add_part(doc, deck_l1, "L1_Lower_Sunboard_Platform", COLOR_WOOD, grp1)

    # L298N Dual H-Bridge Motor Driver (Red PCB + Finned Black Heatsink)
    pcb_x, pcb_y = -40.0, 45.0
    l298_pcb = Part.makeBox(45.0, 45.0, 2.5, Base.Vector(pcb_x - 22.5, pcb_y - 22.5, z_start + T_DECK))
    l298_sink = Part.makeBox(24.0, 16.0, 26.0, Base.Vector(pcb_x - 12.0, pcb_y - 8.0, z_start + T_DECK + 2.5))
    add_part(doc, l298_pcb, "L1_L298N_Driver_Board", COLOR_PCB_RED, grp1)
    add_part(doc, l298_sink, "L1_L298N_Finned_Heatsink", COLOR_JETSON, grp1)

    # Blue Screw Terminals & Electrolytic Capacitors
    term1 = Part.makeBox(38.0, 8.0, 10.0, Base.Vector(pcb_x - 19.0, pcb_y + 13.0, z_start + T_DECK + 2.5))
    term2 = Part.makeBox(38.0, 8.0, 10.0, Base.Vector(pcb_x - 19.0, pcb_y - 21.0, z_start + T_DECK + 2.5))
    add_part(doc, term1, "L1_L298N_Terminal_1", COLOR_JET_BLUE, grp1)
    add_part(doc, term2, "L1_L298N_Terminal_2", COLOR_JET_BLUE, grp1)

    # Dual Power System: Brushed Aluminum Power Bank + Black LiPo Pack
    pbank = Part.makeBox(135.0, 68.0, 20.0, Base.Vector(-60.0, -85.0, z_start + T_DECK))
    battery = Part.makeBox(120.0, 60.0, 22.0, Base.Vector(-50.0, -18.0, z_start + T_DECK))
    add_part(doc, pbank, "L1_Aluminum_Power_Bank", COLOR_POWERBANK, grp1)
    add_part(doc, battery, "L1_Tactical_Battery_Pack", COLOR_BATTERY, grp1)

    return z_start + T_DECK


# ─────────────────────────────────────────────────────────────────────────────
# 6. LAYER 2: SIX SQUARE EXTRUDED ALUMINUM PILLARS (90 mm CLEARANCE)
# ─────────────────────────────────────────────────────────────────────────────
def build_layer_2(doc, z_start):
    grp2 = doc.addObject("App::DocumentObjectGroup", "Layer2_Support_Pillars")

    # 4 Corner Pillars + 2 Mid-span Pillars
    px_coords = [DECK_L3_L/2.0 - PILLAR_W/2.0 - 8.0,
                 0.0,
                 -(DECK_L3_L/2.0 - PILLAR_W/2.0 - 8.0)]
    py_coords = [DECK_L3_W/2.0 - PILLAR_W/2.0 - 8.0,
                 -(DECK_L3_W/2.0 - PILLAR_W/2.0 - 8.0)]

    idx = 1
    for px in px_coords:
        for py in py_coords:
            pillar = Part.makeBox(PILLAR_W, PILLAR_W, PILLAR_H,
                                  Base.Vector(px - PILLAR_W/2.0, py - PILLAR_W/2.0, z_start))
            add_part(doc, pillar, f"L2_Aluminum_Pillar_{idx}", COLOR_ALU_SILV, grp2)
            idx += 1

    return z_start + PILLAR_H


# ─────────────────────────────────────────────────────────────────────────────
# 7. LAYER 3: UPPER DECK (JETSON XAVIER NX + REALSENSE D435i CAMERA)
# ─────────────────────────────────────────────────────────────────────────────
def build_layer_3(doc, z_start):
    grp3 = doc.addObject("App::DocumentObjectGroup", "Layer3_Upper_Deck")

    # 320 x 240 x 5 mm Upper Sunboard Deck with Corner Holes
    deck_l3 = make_drilled_plate(DECK_L3_L, DECK_L3_W, T_DECK, z_start, hole_margin=16.0)
    add_part(doc, deck_l3, "L3_Upper_Sunboard_Platform", COLOR_WOOD, grp3)

    # NVIDIA Jetson Xavier NX: Black Finned Enclosure + Fan Grille + Blue I/O
    jx, jy = 5.0, -10.0
    jetson_case = Part.makeBox(105.0, 95.0, 32.0, 
                               Base.Vector(jx - 52.5, jy - 47.5, z_start + T_DECK))
    # Circular Fan Grille on top
    fan_grille = Part.makeCylinder(32.0, 2.0,
                                   Base.Vector(jx, jy, z_start + T_DECK + 32.0),
                                   Base.Vector(0, 0, 1))
    add_part(doc, jetson_case, "L3_Jetson_Xavier_Enclosure", COLOR_JETSON, grp3)
    add_part(doc, fan_grille, "L3_Jetson_Fan_Grille", COLOR_GUNMETAL, grp3)

    # Blue I/O Port Face on Left Side
    blue_io = Part.makeBox(85.0, 3.0, 20.0,
                           Base.Vector(jx - 42.5, jy + 47.5, z_start + T_DECK + 5.0))
    # Silver RJ45 / USB Ports
    eth_port = Part.makeBox(16.0, 6.0, 14.0, Base.Vector(jx + 12.0, jy + 46.0, z_start + T_DECK + 8.0))
    usb_port = Part.makeBox(15.0, 6.0, 15.0, Base.Vector(jx - 12.0, jy + 46.0, z_start + T_DECK + 8.0))
    add_part(doc, blue_io, "L3_Jetson_Blue_Accent", COLOR_JET_BLUE, grp3)
    add_part(doc, eth_port, "L3_Jetson_Ethernet_Port", COLOR_ALU_SILV, grp3)
    add_part(doc, usb_port, "L3_Jetson_USB_Ports", COLOR_ALU_SILV, grp3)

    # Intel RealSense D435i Depth Camera (Front Center Capsule)
    cx = DECK_L3_L / 2.0 + 6.0
    cy = 0.0
    cz = z_start + T_DECK + 6.0
    
    # Sleek Rounded Camera Capsule Housing (90 x 25 x 25 mm)
    cam_silver = Part.makeBox(24.0, 90.0, 24.0, Base.Vector(cx - 12.0, -45.0, cz))
    cam_face   = Part.makeBox(3.0, 84.0, 20.0,  Base.Vector(cx + 12.0, -42.0, cz + 2.0))
    add_part(doc, cam_silver, "L3_RealSense_Housing", COLOR_CAM_BODY, grp3)
    add_part(doc, cam_face,   "L3_RealSense_Glass_Face", COLOR_CAM_GLASS, grp3)

    # Stereo IR Lenses + Center RGB Optical Sensor
    lens_l = Part.makeCylinder(4.8, 4.0, Base.Vector(cx + 14.0, -30.0, cz + 12.0), Base.Vector(1, 0, 0))
    lens_r = Part.makeCylinder(4.8, 4.0, Base.Vector(cx + 14.0,  30.0, cz + 12.0), Base.Vector(1, 0, 0))
    lens_c = Part.makeCylinder(3.6, 4.0, Base.Vector(cx + 14.0,  -5.0, cz + 12.0), Base.Vector(1, 0, 0))
    add_part(doc, lens_l, "L3_RealSense_IR_Lens_1", COLOR_CAM_GLASS, grp3)
    add_part(doc, lens_r, "L3_RealSense_IR_Lens_2", COLOR_CAM_GLASS, grp3)
    add_part(doc, lens_c, "L3_RealSense_RGB_Sensor", (0.05, 0.15, 0.45), grp3)

    return z_start + T_DECK


# ─────────────────────────────────────────────────────────────────────────────
# 8. LAYER 4: SENSOR MOUNT TOWER (GPS MAST & DUAL-ANTENNA LoRa)
# ─────────────────────────────────────────────────────────────────────────────
def build_layer_4(doc, z_start):
    grp4 = doc.addObject("App::DocumentObjectGroup", "Layer4_Sensor_Tower")

    z_l4 = z_start + L4_STDOFF_H

    # 4x Extruded Aluminum Standoff Pillars supporting Layer 4
    for sx in [-DECK_L4_L/2.0 + 12.0, DECK_L4_L/2.0 - 12.0]:
        for sy in [-DECK_L4_W/2.0 + 12.0, DECK_L4_W/2.0 - 12.0]:
            standoff = Part.makeBox(14.0, 14.0, L4_STDOFF_H,
                                   Base.Vector(sx - 7.0, sy - 7.0, z_start))
            add_part(doc, standoff, f"L4_Tower_Standoff_{int(sx)}_{int(sy)}", COLOR_ALU_SILV, grp4)

    # 160 x 100 x 5 mm Sensor Mount Plate with Corner Holes
    plate_l4 = make_drilled_plate(DECK_L4_L, DECK_L4_W, T_DECK, z_l4, hole_margin=10.0, hole_dia=5.0)
    add_part(doc, plate_l4, "L4_Sensor_Mount_Platform", COLOR_WOOD, grp4)

    # Elevated GPS NEO-6M Aerodynamic Mushroom Radome
    gps_x, gps_y = 35.0, 0.0
    gps_mast = Part.makeCylinder(5.5, 30.0, 
                                 Base.Vector(gps_x, gps_y, z_l4 + T_DECK), 
                                 Base.Vector(0, 0, 1))
    # Beveled Mushroom Disc Dome
    gps_base = Part.makeCylinder(26.0, 6.0,
                                 Base.Vector(gps_x, gps_y, z_l4 + T_DECK + 30.0),
                                 Base.Vector(0, 0, 1))
    gps_cap  = Part.makeCylinder(22.0, 6.0,
                                 Base.Vector(gps_x, gps_y, z_l4 + T_DECK + 36.0),
                                 Base.Vector(0, 0, 1))
    gps_dome_unit = gps_base.fuse(gps_cap)
    add_part(doc, gps_mast, "L4_GPS_Mount_Mast", COLOR_GUNMETAL, grp4)
    add_part(doc, gps_dome_unit, "L4_GPS_Mushroom_Radome", COLOR_GPS_DOME, grp4)

    # LoRa SX1278 Module with Dual High-Gain Rubber Duck Whip Antennas
    lora_x, lora_y = -35.0, 0.0
    lora_pcb = Part.makeBox(36.0, 30.0, 2.5, Base.Vector(lora_x - 18.0, lora_y - 15.0, z_l4 + T_DECK))
    add_part(doc, lora_pcb, "L4_LoRa_Transceiver_PCB", COLOR_PCB_GRN, grp4)

    # Dual Antennas with Gold SMA Knurled Bases
    for a_idx, ay in enumerate([-8.0, 8.0]):
        sma_base = Part.makeCylinder(3.8, 8.0,
                                     Base.Vector(lora_x - 5.0, lora_y + ay, z_l4 + T_DECK + 2.5),
                                     Base.Vector(0, 0, 1))
        whip = Part.makeCylinder(3.0, 68.0,
                                 Base.Vector(lora_x - 5.0, lora_y + ay, z_l4 + T_DECK + 10.5),
                                 Base.Vector(0, 0, 1))
        whip_tip = Part.makeCylinder(2.2, 5.0,
                                     Base.Vector(lora_x - 5.0, lora_y + ay, z_l4 + T_DECK + 78.5),
                                     Base.Vector(0, 0, 1))
        ant_unit = whip.fuse(whip_tip)
        add_part(doc, sma_base, f"L4_LoRa_SMA_Connector_{a_idx+1}", COLOR_YELLOW, grp4)
        add_part(doc, ant_unit, f"L4_LoRa_Whip_Antenna_{a_idx+1}", COLOR_RUBBER, grp4)


# ─────────────────────────────────────────────────────────────────────────────
# 9. MAIN PIPELINE EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
def main():
    doc = create_doc("Military_Surveillance_Rover")
    print("\n" + "=" * 65)
    print(" GENERATING PHOTOREALISTIC MILITARY SURVEILLANCE ROVER CAD")
    print("=" * 65)

    z_l0 = build_layer_0(doc)
    print(" [1/5] Layer 0: Chassis Ladder Frame, 4WD Motors & 5-Spoke Wheels built.")

    z_l1 = build_layer_1(doc, z_l0)
    print(" [2/5] Layer 1: Natural Wood Lower Deck (380x280) + L298N + Power Units built.")

    z_l2 = build_layer_2(doc, z_l1)
    print(" [3/5] Layer 2: 6x Satin Aluminum Extruded Square Pillars built.")

    z_l3 = build_layer_3(doc, z_l2)
    print(" [4/5] Layer 3: Upper Deck (320x240) + Jetson Xavier NX + RealSense D435i built.")

    build_layer_4(doc, z_l3)
    print(" [5/5] Layer 4: Sensor Tower (160x100) + Aerodynamic GPS + Dual LoRa built.")

    doc.recompute()

    if App.GuiUp:
        import FreeCADGui as Gui
        Gui.activeDocument().activeView().viewAxometric()
        Gui.SendMsgToActiveView("ViewSelection")
        Gui.activeDocument().activeView().fitAll()

    print("=" * 65)
    print(" >>> ROVER 3D CAD MODEL COMPLETED & FIT TO SCREEN! <<<")
    print("=" * 65 + "\n")

main()
