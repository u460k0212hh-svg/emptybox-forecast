import sys,glob,json,os
sys.path.insert(0,os.path.dirname(__file__)); import ocr_bpa as o
from PIL import Image
A=sys.argv[1]
key=lambda f:os.path.basename(f).split('-',1)[1]
allimg=glob.glob(os.path.join(A,'*image.png'))
g1=sorted([f for f in allimg if Image.open(f).size==(2560,1600)],key=key)
g2=sorted([f for f in allimg if 2140<=Image.open(f).size[0]<=2190 and Image.open(f).size[1]>1100],key=key)
spec=[]
eo={1:0,2:12,3:2,4:3,5:4,6:5,7:6,8:7,9:8,10:9,11:10,12:11}
for m,i in eo.items(): spec.append(('empty_out',m,g1[i]))
for m in range(1,13): spec.append(('laden_in',m,g1[12+m]))
for m in range(1,13): spec.append(('laden_out',m,g1[24+m]))
ei={1:2,2:3,3:10,4:5,5:6,6:7,7:8,8:9,9:11,10:12,11:13,12:14}
for m,i in ei.items(): spec.append(('empty_in',m,g2[i]))
out=[]
for s,m,f in spec:
    r=o.read_image(f)
    out.append({'series':s,'month':m,'file':os.path.basename(f),'vals':r})
    print(s,m,len(r),[v['white'] if v['white']==v['pink'] or v['pink'] is None else (v['white'],v['pink']) for v in r])
json.dump(out,open(sys.argv[2],'w'),ensure_ascii=False,indent=1)
