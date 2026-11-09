"""KOSIS 컨테이너수송현황(XML 스프레드시트)에서 주요 항만의 연간 적·공컨테이너 수출입(환적 제외)을 만든다.
외항입항 = 수입, 외항출항 = 수출 (환적은 별도 항목이라 포함되지 않음)."""
import re, pandas as pd
SRC='data/raw/kosis_container_transport_2308_2608.xls'
t=open(SRC,'rb').read().decode('euc-kr',errors='replace')
P=[[re.sub(r'<[^>]+>','',c).strip() for c in re.findall(r'<Cell[^>]*>(.*?)</Cell>',r,re.S)] for r in re.findall(r'<Row[^>]*>(.*?)</Row>',t,re.S)]
hdr=P[1]; months=[h.replace(' 월','').replace('.','') for h in hdr[6:]]
rec=[]; port=None
for r in P[2:]:
    if len(r)<7: continue
    if r[0]: port=r[0]
    nat,io,kind,item,unit=r[1:6]
    if kind!='TEU' or io not in ('외항입항','외항출항') or item not in ('적','공'): continue
    for m,v in zip(months,r[6:]):
        v=v.replace(',','')
        rec.append((port,nat,io,item,int(m),float(v) if v not in ('','-') else 0.0))
d=pd.DataFrame(rec,columns=['port','nat','io','item','ym','teu'])
nats=d.nat.unique(); print('국적구분',nats)
d=d[d.nat!='계'] if '계' in nats else d
m=d.groupby(['port','io','item','ym']).teu.sum().unstack(['io','item'])
m.columns=[{'외항입항':'in','외항출항':'out'}[a]+'_'+{'적':'laden','공':'empty'}[b] for a,b in m.columns]
m=m.reset_index(); m['year']=m.ym//100
m.to_csv('data/processed/port_monthly_kosis.csv',index=False)
keep={'부산':'부산','인천':'인천','광양':'광양','평택·당진':'평택·당진','울산':'울산'}
a=m[m.port.isin(keep)&m.year.isin([2024,2025])].groupby(['year','port'])[['in_empty','out_empty','in_laden','out_laden']].sum()/1000
a=a.reset_index()
o=pd.DataFrame({'year':a.year,'port':a.port,'empty_in':a.in_empty.round(1),'empty_out':a.out_empty.round(1),
                'empty_net_inflow':(a.in_empty-a.out_empty).round(1),'laden_imb':(a.out_laden-a.in_laden).round(1)})
o['port_type']=o.empty_net_inflow.map(lambda x:'shortage' if x>0 else 'surplus')
order={'부산':0,'인천':1,'광양':2,'평택·당진':3,'울산':4}
o=o.sort_values(['year','port'],key=lambda s:s.map(order) if s.name=='port' else s)
o.to_csv('data/processed/port_empty_container_2024_2025.csv',index=False,encoding='utf-8-sig')
print(o.to_string(index=False)); print(o.groupby(['year','port_type']).empty_net_inflow.sum())
