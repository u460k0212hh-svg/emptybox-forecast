"""BPA 체인포털 캡처 OCR 결과 + 수동 판독 보정으로 부산항 월별 적·공컨테이너(환적 제외) 시계열을 만든다.
출력: data/processed/busan_monthly_bpa.csv (단위: 천TEU, 2016.09~2026.08)"""
import json, pandas as pd
R=json.load(open('data/raw/ocr_raw.json'))
FIX={('empty_out',7,4):73,('empty_out',7,5):82,('empty_out',10,4):59,('empty_out',10,5):61,
     ('laden_in',2,0):213,('laden_in',3,4):294,('laden_in',3,5):283,('laden_in',6,4):273,('laden_in',6,5):270,
     ('laden_out',7,3):325}
# 공컨 수입: 툴팁·로딩 화면이 많아 화면을 직접 보고 판독 (월별 10개년)
EMPTY_IN={1:[148,155,160,149,161,167,156,175,202,154],2:[145,165,152,192,154,166,187,185,172,160],
 3:[164,178,174,184,210,187,210,236,214,188],4:[183,165,171,155,171,161,192,186,194,160],
 5:[169,181,171,118,179,182,202,188,202,184],6:[174,159,154,110,166,137,176,193,184,181],
 7:[166,159,180,120,166,153,167,188,180,184],8:[170,156,161,124,146,168,179,178,162,187],
 9:[178,181,163,157,144,146,123,199,170,179],10:[168,169,166,165,159,179,158,180,168,147],
 11:[158,162,169,166,167,141,173,195,168,157],12:[154,157,151,156,172,162,167,196,185,185]}
rows=[]
def years(m): return list(range(2017,2027)) if m<=8 else list(range(2016,2026))
for r in R:
    s,m=r['series'],r['month']
    if s=='empty_in': continue
    for i,v in enumerate(r['vals']):
        val=FIX.get((s,m,i),v.get('val') if v.get('val') is not None else (int(v['white']) if v.get('white') else None))
        rows.append((years(m)[i],m,s,val))
for m,vs in EMPTY_IN.items():
    for i,v in enumerate(vs): rows.append((years(m)[i],m,'empty_in',v))
d=pd.DataFrame(rows,columns=['year','month','series','val']).pivot_table(index=['year','month'],columns='series',values='val').reset_index()
d['ym']=d.year*100+d.month
d=d.sort_values('ym')[['ym','year','month','empty_in','empty_out','laden_in','laden_out']]
assert d[['empty_in','empty_out','laden_in','laden_out']].notna().all().all(), d[d.isna().any(axis=1)]
d.to_csv('data/processed/busan_monthly_bpa.csv',index=False)
print(len(d), d.ym.min(), d.ym.max())
