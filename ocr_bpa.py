"""BPA 체인포털 '컨테이너 별' 화면 캡처에서 막대 안 흰 숫자를 읽는다."""
import sys, glob, subprocess, re, json
import numpy as np
from PIL import Image

def bars(im):
    a=np.asarray(im.convert('RGB')).astype(int)
    H,W,_=a.shape
    blue=(a[:,:,2]>120)&(a[:,:,0]<40)&(a[:,:,1]>50)&(a[:,:,1]<110)   # #004EA2 계열
    col=blue.sum(0)
    xs=np.where(col>H*0.03)[0]
    runs=[];s=None
    for i,x in enumerate(xs):
        if s is None: s=p=x; continue
        if x==p+1: p=x; continue
        runs.append((s,p)); s=p=x
    if s is not None: runs.append((s,p))
    runs=[r for r in runs if r[1]-r[0]>W*0.025]
    out=[]
    for x0,x1 in runs:
        ys=np.where(blue[:,(x0+x1)//2])[0]
        out.append((x0,x1,ys.min(),ys.max()))
    return out

def read_num(im,box):
    x0,x1,y0,y1=box
    c=im.crop((x0+2,y0+2,x1-2,y1-2)).convert('RGB')
    a=np.asarray(c).astype(int)
    white=(a.min(2)>200)
    m=Image.fromarray(np.where(white,0,255).astype('uint8')).resize((c.width*3,c.height*3))
    m.save('/tmp/claude-0/-home-claude/645dc096-2900-5a44-a23b-ac486c98fc23/scratchpad/ocr_tmp.png')
    t=subprocess.run(['tesseract','/tmp/claude-0/-home-claude/645dc096-2900-5a44-a23b-ac486c98fc23/scratchpad/ocr_tmp.png','-','--psm','6'],capture_output=True,text=True).stdout
    nums=re.findall(r'(?<![\d.])(\d{1,3}(?:,\d{3})?)(?![\d.%])',t)
    return nums[0].replace(',','') if nums else None, t.replace('\n',' | ')

if __name__=='__main__':
    for f in sys.argv[1:]:
        im=Image.open(f)
        bs=bars(im)
        vals=[read_num(im,b) for b in bs]
        print(f.split('/')[-1][:8], len(bs), [v[0] for v in vals])

def read_pink(im,box):
    x0,x1,y0,y1=box
    c=im.crop((x0-15,max(0,y0-75),x1+15,y0-2)).convert('RGB')
    a=np.asarray(c).astype(int)
    pink=(a[:,:,0]>180)&(a[:,:,1]<140)&(a[:,:,2]>90)&(a[:,:,2]<180)
    m=Image.fromarray(np.where(pink,0,255).astype('uint8')).resize((c.width*3,c.height*3))
    p='/tmp/claude-0/-home-claude/645dc096-2900-5a44-a23b-ac486c98fc23/scratchpad/ocr_tmp2.png'; m.save(p)
    t=subprocess.run(['tesseract',p,'-','--psm','6'],capture_output=True,text=True).stdout
    nums=re.findall(r'(?<![\d.])(\d{1,3}(?:,\d{3})?)(?![\d.%])',t)
    return nums[0].replace(',','') if nums else None

def read_image(f):
    im=Image.open(f)
    bs=bars(im)[-10:]
    res=[]
    for b in bs:
        w,_=read_num(im,b); p=read_pink(im,b)
        res.append({'white':w,'pink':p})
    return res

def read_pink2(im,box):
    x0,x1,y0,y1=box
    cx0,cy0=x0-25,max(0,y0-95)
    c=im.crop((cx0,cy0,x1+25,y0-8)).convert('RGB')
    a=np.asarray(c).astype(int)
    pink=(a[:,:,0]>170)&(a[:,:,1]<150)&(a[:,:,2]>80)&(a[:,:,2]<190)&(a[:,:,0]-a[:,:,1]>60)
    rows=np.where(pink.sum(1)>=2)[0]
    if len(rows)==0: return None
    # group rows into text lines
    lines=[];s=p=rows[0]
    for r in rows[1:]:
        if r<=p+2: p=r; continue
        lines.append((s,p)); s=p=r
    lines.append((s,p))
    lines=[l for l in lines if l[1]-l[0]>=6]
    if not lines: return None
    l0,l1=lines[0]
    sub=pink[max(0,l0-2):l1+3]
    cols=np.where(sub.any(0))[0]
    sub=sub[:,max(0,cols.min()-4):cols.max()+5]
    m=Image.fromarray(np.where(sub,0,255).astype('uint8'))
    m=m.resize((m.width*4,m.height*4),Image.LANCZOS)
    p='/tmp/claude-0/-home-claude/645dc096-2900-5a44-a23b-ac486c98fc23/scratchpad/ocr_tmp3.png'
    bg=Image.new('L',(m.width+40,m.height+40),255); bg.paste(m,(20,20)); bg.save(p)
    t=subprocess.run(['tesseract',p,'-','--psm','7','-c','tessedit_char_whitelist=0123456789,'],capture_output=True,text=True).stdout.strip().replace(',','')
    return t or None
