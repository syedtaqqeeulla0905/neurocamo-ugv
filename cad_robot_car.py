import FreeCAD as App
import FreeCADGui as Gui
import Part
from FreeCAD import Base, Vector

# 1. Create a New Document
doc_name = "NeuroCamo_Robot_UGV"
if App.ActiveDocument:
    doc = App.ActiveDocument
else:
    doc = App.newDocument(doc_name)

App.setActiveDocument(doc.Name)

def make_box(name, length, width, height, pos=(0,0,0), rot=(0,0,0), color=(0.7,0.7,0.7)):
    obj = doc.addObject("Part::Box", name)
    obj.Length = length
    obj.Width = width
    obj.Height = height
    obj.Placement.Base = Vector(pos[0], pos[1], pos[2])
    doc.recompute()
    if Gui.ActiveDocument:
        gui_obj = Gui.ActiveDocument.getObject(name)
        if gui_obj:
            gui_obj.ShapeColor = color
    return obj

def make_cylinder(name, radius, height, pos=(0,0,0), direction=(0,1,0), color=(0.2,0.2,0.2)):
    obj = doc.addObject("Part::Cylinder", name)
    obj.Radius = radius
    obj.Height = height
    obj.Placement.Base = Vector(pos[0], pos[1], pos[2])
    obj.Placement.Rotation = App.Rotation(Vector(0,0,1), Vector(direction[0], direction[1], direction[2]))
    doc.recompute()
    if Gui.ActiveDocument:
        gui_obj = Gui.ActiveDocument.getObject(name)
        if gui_obj:
            gui_obj.ShapeColor = color
    return obj

print("Building NeuroCamo Ground Vehicle CAD Model...")

# -------------------------------------------------------------
# Base Chassis Dimensions (in mm)
# Robot dimensions: ~240mm (length, X), 160mm (width, Y)
# Ground clearance: 30mm
# -------------------------------------------------------------
chassis_len = 240
chassis_wid = 150
plate_thk = 3.5

# 1. Lower Chassis Deck (Tactical Olive/Military Tan)
lower_deck = make_box("Lower_Chassis_Deck", chassis_len, chassis_wid, plate_thk, 
                      pos=(-chassis_len/2, -chassis_wid/2, 30), 
                      color=(0.72, 0.65, 0.48))

# 2. Upper Chassis Deck
upper_deck = make_box("Upper_Chassis_Deck", chassis_len - 30, chassis_wid, plate_thk, 
                      pos=(-chassis_len/2 + 20, -chassis_wid/2, 65), 
                      color=(0.65, 0.58, 0.42))

# 3. Structural Brass Standoffs (4 corners)
standoff_radius = 3.0
standoff_height = 35.0
standoffs_pos = [
    (-80, -60, 30),
    (80, -60, 30),
    (-80, 60, 30),
    (80, 60, 30)
]
for idx, (sx, sy, sz) in enumerate(standoffs_pos):
    make_cylinder(f"Standoff_{idx+1}", standoff_radius, standoff_height, pos=(sx, sy, sz), direction=(0,0,1), color=(0.85, 0.75, 0.2))

# 4. 4WD All-Terrain Wheels & Hubs
wheel_radius = 33.0
wheel_width = 28.0
hub_radius = 16.0
wheel_x_positions = [-75, 75]
wheel_y_offset = (chassis_wid / 2) + 4

for idx, wx in enumerate(wheel_x_positions):
    # Left Wheels (+Y side)
    make_cylinder(f"Wheel_Left_{idx+1}", wheel_radius, wheel_width, 
                  pos=(wx, wheel_y_offset, 33), direction=(0,1,0), color=(0.12, 0.12, 0.12))
    make_cylinder(f"Rim_Left_{idx+1}", hub_radius, wheel_width + 1, 
                  pos=(wx, wheel_y_offset - 0.5, 33), direction=(0,1,0), color=(0.8, 0.8, 0.85))
    
    # Right Wheels (-Y side)
    make_cylinder(f"Wheel_Right_{idx+1}", wheel_radius, wheel_width, 
                  pos=(wx, -wheel_y_offset - wheel_width, 33), direction=(0,1,0), color=(0.12, 0.12, 0.12))
    make_cylinder(f"Rim_Right_{idx+1}", hub_radius, wheel_width + 1, 
                  pos=(wx, -wheel_y_offset - wheel_width - 0.5, 33), direction=(0,1,0), color=(0.8, 0.8, 0.85))

# -------------------------------------------------------------
# Mounted Hardware Modules
# -------------------------------------------------------------

# 5. NVIDIA Jetson Nano (Center of Upper Deck)
# Carrier board: 100mm x 80mm
jetson_pcb = make_box("Jetson_Nano_PCB", 100, 80, 2.5, 
                      pos=(-50, -40, 68.5), color=(0.05, 0.45, 0.18))

# Jetson Extruded Aluminum Heatsink
jetson_heatsink = make_box("Jetson_Heatsink", 60, 50, 16, 
                           pos=(-30, -25, 71.0), color=(0.1, 0.1, 0.1))

# Fins groove aesthetics
for f in range(4):
    make_box(f"Heatsink_Fin_{f+1}", 55, 6, 4, 
             pos=(-27.5, -22 + f*12, 87.0), color=(0.2, 0.2, 0.2))

# 6. Front CSI PiCam v2 (Camera Mast & Housing)
# Front is +X axis
cam_mast = make_box("Camera_Bracket", 6, 25, 45, 
                    pos=(95, -12.5, 68.5), color=(0.15, 0.15, 0.15))

cam_sensor = make_box("Camera_Sensor_PCB", 25, 24, 2, 
                      pos=(98, -12, 95), color=(0.1, 0.55, 0.25))

cam_lens = make_cylinder("Camera_Lens", 5.5, 8, 
                         pos=(100, 0, 107), direction=(1,0,0), color=(0.02, 0.02, 0.02))

# 7. L298N Dual H-Bridge Motor Driver (Mounted Lower Deck Front/Side)
l298n_pcb = make_box("L298N_Driver_PCB", 45, 45, 2, 
                     pos=(10, -chassis_wid/2 + 10, 33.5), color=(0.8, 0.1, 0.1)) # Red Board

l298n_heatsink = make_box("L298N_Heatsink", 16, 32, 22, 
                          pos=(24, -chassis_wid/2 + 16, 35.5), color=(0.25, 0.25, 0.28))

# 8. 11.1V 3S Li-Po Battery Pack (Rear Lower Deck for balance)
lipo_battery = make_box("LiPo_3S_Battery", 105, 35, 25, 
                        pos=(-100, -17.5, 33.5), color=(0.1, 0.35, 0.85)) # Blue Li-Po Pack

# 9. LM2596 DC-DC Buck Converter (Step-Down Regulator)
buck_converter = make_box("Buck_Converter_5V", 43, 21, 14, 
                          pos=(-35, -chassis_wid/2 + 15, 33.5), color=(0.1, 0.2, 0.6))

# 10. Rear LoRa/GPS Antenna Mast (Rear +X=-95, Y=50)
antenna_bracket = make_box("Antenna_Bracket", 15, 15, 10, 
                           pos=(-100, 45, 68.5), color=(0.15, 0.15, 0.15))
antenna_mast = make_cylinder("Antenna_Mast", 2.2, 75, 
                             pos=(-92.5, 52.5, 78.5), direction=(0,0,1), color=(0.1, 0.1, 0.1))

doc.recompute()
if Gui.ActiveDocument:
    Gui.SendMsgToActiveView("ViewAxo")
    Gui.SendMsgToActiveView("ViewFit")

print("CAD Model of NeuroCamo Car Generated Successfully!")
