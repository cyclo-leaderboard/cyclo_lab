"""Author the measured two-rail, fourteen-peg wooden plate holder."""
from pathlib import Path
import json
import math
TASK=Path(__file__).resolve().parent
OUT=TASK/'runtime/data/objects/plate_holder_v1/wood_plate_holder.usda'
OUT.parent.mkdir(parents=True,exist_ok=True)
L=.295;W=.12;H=.09;RAIL=.017;PEG_D=.009;PEG_SPACING=.045
PEGS=7
rail_y=(W-RAIL)/2
xs=[(i-(PEGS-1)/2)*PEG_SPACING for i in range(PEGS)]
assert abs(xs[0]+.135)<1e-9 and abs((L/2-xs[-1])-.0125)<1e-9
lines=['''#usda 1.0
(
 defaultPrim = "WoodPlateHolder"
 metersPerUnit = 1
 upAxis = "Z"
)
def Xform "WoodPlateHolder"
{
 def Scope "Looks"
 {
  def Material "LightWood"
  {
   token outputs:surface.connect = </WoodPlateHolder/Looks/LightWood/Shader.outputs:surface>
   def Shader "Shader"
   {
    uniform token info:id = "UsdPreviewSurface"
    color3f inputs:diffuseColor.connect = </WoodPlateHolder/Looks/LightWood/Texture.outputs:rgb>
    float inputs:roughness = 0.62
    token outputs:surface
   }
   def Shader "Texture"
   {
    uniform token info:id = "UsdUVTexture"
    asset inputs:file = @./wood_grain_v2.png@
    float4 inputs:scale = (0.88, 1.14, 0.86, 1)
    token inputs:sourceColorSpace = "sRGB"
    float2 inputs:st.connect = </WoodPlateHolder/Looks/LightWood/PrimvarReader.outputs:result>
    token inputs:wrapS = "repeat"
    token inputs:wrapT = "repeat"
    float3 outputs:rgb
   }
   def Shader "PrimvarReader"
   {
    uniform token info:id = "UsdPrimvarReader_float2"
    token inputs:varname = "st"
    float2 outputs:result
   }
  }
 }
''']
# The cylindrical side grain is darker and less yellow than the flat rails
# and bright end grain in the saved head image.
material_start = lines[0].index('  def Material "LightWood"')
material_end = lines[0].rfind('\n }')
peg_material = lines[0][material_start:material_end].replace('LightWood', 'PegSideWood')
peg_material = peg_material.replace('(0.88, 1.14, 0.86, 1)', '(0.57, 0.82, 0.96, 1)')
lines[0] = lines[0][:material_end] + '\n' + peg_material + lines[0][material_end:]
def cube(name,size,pos):
 lines.append(f''' def Cube "{name}" (prepend apiSchemas = ["MaterialBindingAPI", "PhysicsCollisionAPI"])
 {{
  rel material:binding = </WoodPlateHolder/Looks/LightWood>
  bool physics:collisionEnabled = true
  token visibility = "invisible"
  double size = 1
  float3 xformOp:scale = ({size[0]:.7f}, {size[1]:.7f}, {size[2]:.7f})
  double3 xformOp:translate = ({pos[0]:.7f}, {pos[1]:.7f}, {pos[2]:.7f})
  uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
 }}''')
 box_visual(name,size,pos)
def cylinder(name,radius,height,pos,axis=None):
 rotation='  float3 xformOp:rotateXYZ = (90, 0, 0)\n' if axis=='y' else ''
 order='["xformOp:translate", "xformOp:rotateXYZ"]' if axis=='y' else '["xformOp:translate"]'
 lines.append(f''' def Cylinder "{name}" (prepend apiSchemas = ["MaterialBindingAPI", "PhysicsCollisionAPI"])
 {{
  rel material:binding = </WoodPlateHolder/Looks/LightWood>
  bool physics:collisionEnabled = true
  token visibility = "invisible"
  uniform token axis = "Z"
  double radius = {radius:.7f}
  double height = {height:.7f}
{rotation}  double3 xformOp:translate = ({pos[0]:.7f}, {pos[1]:.7f}, {pos[2]:.7f})
  uniform token[] xformOpOrder = {order}
 }}''')
 cylinder_visual(name,radius,height,pos,axis)

def _array(values):
 return '['+', '.join(str(v) if isinstance(v,int) else '('+', '.join(f'{x:.7f}' for x in v)+')' for v in values)+']'

def _visual_mesh(name,points,faces,uvs,pos,axis=None):
 rotation='  float3 xformOp:rotateXYZ = (90, 0, 0)\n' if axis=='y' else ''
 order='["xformOp:translate", "xformOp:rotateXYZ"]' if axis=='y' else '["xformOp:translate"]'
 side_binding = ''
 if 'Peg' in name:
  side_binding = '''  def GeomSubset "SideGrain" (prepend apiSchemas = ["MaterialBindingAPI"])
  {
   uniform token elementType = "face"
   uniform token familyName = "materialBind"
   int[] indices = ''' + _array(list(range(24))) + '''
   rel material:binding = </WoodPlateHolder/Looks/PegSideWood>
  }
'''
 lines.append(f''' def Mesh "{name}Visual" (prepend apiSchemas = ["MaterialBindingAPI"])
 {{
  rel material:binding = </WoodPlateHolder/Looks/LightWood>
  point3f[] points = {_array(points)}
  int[] faceVertexCounts = {_array([len(face) for face in faces])}
  int[] faceVertexIndices = {_array([i for face in faces for i in face])}
  texCoord2f[] primvars:st = {_array(uvs)} (
   interpolation = "faceVarying"
  )
  uniform token subdivisionScheme = "none"
{rotation}  double3 xformOp:translate = ({pos[0]:.7f}, {pos[1]:.7f}, {pos[2]:.7f})
  uniform token[] xformOpOrder = {order}
{side_binding}
 }}''')

def box_visual(name,size,pos):
 hx,hy,hz=(v/2 for v in size)
 points=[(-hx,-hy,-hz),(hx,-hy,-hz),(hx,hy,-hz),(-hx,hy,-hz),
         (-hx,-hy,hz),(hx,-hy,hz),(hx,hy,hz),(-hx,hy,hz)]
 faces=[[3,2,1,0],[4,5,6,7],[0,1,5,4],[2,3,7,6],[3,0,4,7],[1,2,6,5]]
 uvs=[]
 for face_index,face in enumerate(faces):
  for i in face:
   x,y,z=points[i]
   if face_index<2:uv=((x+hx)/size[0],(y+hy)/size[1])
   elif face_index<4:uv=((x+hx)/size[0],(z+hz)/size[2])
   else:uv=((y+hy)/size[1],(z+hz)/size[2])
   uvs.append(uv)
 _visual_mesh(name,points,faces,uvs,pos)

def cylinder_visual(name,radius,height,pos,axis):
 sides=24
 points=[(radius*math.cos(2*math.pi*i/sides),radius*math.sin(2*math.pi*i/sides),z)
         for z in (-height/2,height/2) for i in range(sides)]
 faces=[];uvs=[]
 for i in range(sides):
  j=(i+1)%sides
  faces.append([i,j,sides+j,sides+i])
  uvs.extend([(0,i/sides),(0,(i+1)/sides),(1,(i+1)/sides),(1,i/sides)])
 for face in (list(reversed(range(sides))),list(range(sides,2*sides))):
  faces.append(face)
  uvs.extend([((points[i][0]/radius+1)/2,(points[i][1]/radius+1)/2) for i in face])
 _visual_mesh(name,points,faces,uvs,pos,axis)

for row,y in [('Near',-rail_y),('Far',rail_y)]:
 cube(f'{row}Rail',(L,RAIL,RAIL),(0,y,RAIL/2))
 for i,x in enumerate(xs):
  bottom=RAIL-.0025; h=H-bottom
  cylinder(f'{row}Peg{i}',PEG_D/2,h,(x,y,bottom+h/2))
# Crossbars seen near both ends in the right wrist view. Their radius/spacing
# are visual estimates, since only overall holder and peg sizes were measured.
for i,x in enumerate((-.1125,.1125)):
 cylinder(f'CrossBrace{i}',.005,2*rail_y,(x,0,RAIL/2),'y')
lines.append('}')
OUT.write_text('\n'.join(lines)+'\n')
report={'asset':str(OUT),'measured':{'length_m':L,'width_m':W,'height_m':H,'base_rail_square_m':RAIL,'peg_diameter_m':PEG_D,'peg_count_each_row':PEGS,'peg_center_spacing_m':PEG_SPACING},'derived':{'peg_centers_x_m':xs,'peg_center_end_margin_m':L/2-xs[-1],'rail_centers_y_m':[-rail_y,rail_y],'rail_outer_width_m':2*rail_y+RAIL,'peg_top_z_m':H,'crossbrace_radius_m_visual_estimate':.005,'crossbrace_x_m_visual_estimate':[-.1125,.1125]},'source_photos':['head.jpg','right.jpg']}
OUT.with_name('holder_geometry.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
