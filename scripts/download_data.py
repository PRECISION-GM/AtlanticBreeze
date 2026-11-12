import os,sys,json,datetime,hashlib,time,shutil,subprocess,argparse,tomllib
from pathlib import Path
import zipfile
parser=argparse.ArgumentParser(description='Download ENA observations and regional reanalysis; credentials stay outside the repository.')
parser.add_argument('source', choices=['arm','era','merra'])
parser.add_argument('--case', type=Path, default=Path(__file__).resolve().parents[1]/'cases/march2016.toml')
parser.add_argument('--data-dir', type=Path, default=Path('data/march2016'))
args=parser.parse_args()
import requests,netCDF4 as nc,numpy as np
os.umask(0o077)
with args.case.open('rb') as f: config=tomllib.load(f)
ROOT=args.data_dir.resolve(); ROOT.mkdir(parents=True,exist_ok=True)
start,end=config['start'],config['end']
DAYS=[start+datetime.timedelta(days=i) for i in range((end-start).days+1)]
# Prioritize the focal day, then nearby days without changing the coverage window.
DAYS.sort(key=lambda d:abs((d-config['focus']).days))
def record(kind,**kw):
 with open(ROOT/(kind+'.jsonl'),'a') as f:f.write(json.dumps(kw)+'\n')
 print(kind,kw,flush=True)
def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def room():
 if shutil.disk_usage(ROOT).free<25*1024**3:raise RuntimeError('storage reserve reached')
def arm():
 user=os.environ['ARM_USERNAME']+':'+os.environ['ARM_TOKEN']; base='https://adc.arm.gov/armlive/livedata'
 streams=['enasondewnpnC1.b1','enamwrret2turnC1.c1','enamwr3cC1.b1','enatropoeC1.c1','enaarsclkazr1kolliasC1.c0','enaceilC1.b1','enainterpolatedsondeC1.c1','enametC1.b1','enaaosccn100C1.b1','enaaosuhsasC1.b1','enavisstgridm10minnisX1.c1','enavisstpx2dm10minnisX1.c1','enakazrgeC1.a1']
 for ds in streams:
  for d in DAYS:
   day=d.strftime('%Y%m%d')
   try:
    r=requests.get(base+'/query',params=dict(user=user,ds=ds,start=str(d),end=str(d+datetime.timedelta(days=1)),wt='json'),timeout=180);r.raise_for_status()
    files=[f for f in r.json().get('files',[]) if '.'+day+'.' in f]
    record('arm_inventory',stream=ds,date=str(d),files=files)
    for name in files:
     dst=ROOT/'arm'/ds/name;dst.parent.mkdir(parents=True,exist_ok=True)
     if dst.exists():continue
     room()
     with requests.get(base+'/saveData',params=dict(user=user,file=name),stream=True,timeout=600) as r:
      r.raise_for_status()
      with open(str(dst)+'.part','wb') as f:
       for b in r.iter_content(1048576):f.write(b)
     with nc.Dataset(str(dst)+'.part'):pass
     os.replace(str(dst)+'.part',dst)
     record('files',source='ARM',file=str(dst.relative_to(ROOT)),bytes=dst.stat().st_size,sha256=digest(dst))
   except Exception as e:record('errors',source='ARM',stream=ds,date=str(d),error=type(e).__name__)
def era():
 import cdsapi
 out=ROOT/'era5';out.mkdir(exist_ok=True)
 levels=['100','125','150','175','200','225','250','300','350','400','450','500','550','600','650','700','750','775','800','825','850','875','900','925','950','975','1000']
 for d in DAYS:
  for typ in ['single','pressure']:
   dst=out/f'era5_{typ}_{d:%Y%m%d}.nc'
   if dst.exists() or dst.with_suffix('.zip').exists():continue
   req=dict(product_type=['reanalysis'],year=[str(d.year)],month=[f'{d.month:02}'],day=[f'{d.day:02}'],time=[f'{h:02}:00' for h in range(24)],area=config['area'],data_format='netcdf',download_format='unarchived')
   if typ=='pressure':req.update(pressure_level=levels,variable=['temperature','specific_humidity','vertical_velocity','u_component_of_wind','v_component_of_wind','geopotential','specific_cloud_liquid_water_content','fraction_of_cloud_cover'])
   else:req.update(variable=['sea_surface_temperature','surface_pressure','mean_sea_level_pressure','surface_sensible_heat_flux','surface_latent_heat_flux','surface_net_solar_radiation','surface_net_thermal_radiation','total_column_cloud_liquid_water','total_cloud_cover','low_cloud_cover','total_precipitation','boundary_layer_height','10m_u_component_of_wind','10m_v_component_of_wind','2m_temperature','2m_dewpoint_temperature'])
   record('requests',source='ERA5',dataset=typ,request=req)
   try:
    room()
    part=Path(str(dst)+'.part')
    if not part.exists():cdsapi.Client(quiet=True,progress=False).retrieve('reanalysis-era5-'+typ+'-levels',req,str(part))
    if zipfile.is_zipfile(part):
     with zipfile.ZipFile(part) as z:
      for member in z.namelist():
       target=out/(dst.stem+'_'+Path(member).name)
       target.write_bytes(z.read(member))
       with nc.Dataset(target):pass
       record('files',source='ERA5',file=str(target.relative_to(ROOT)),sha256=digest(target),bytes=target.stat().st_size)
     os.replace(part,dst.with_suffix('.zip'))
    else:
     with nc.Dataset(part):pass
     os.replace(part,dst);record('files',source='ERA5',file=str(dst.relative_to(ROOT)),sha256=digest(dst),bytes=dst.stat().st_size)
   except Exception as e:record('errors',source='ERA5',date=str(d),dataset=typ,error=type(e).__name__)
def merra():
 out=ROOT/'merra2';out.mkdir(exist_ok=True)
 products=['M2T1NXSLV','M2T1NXFLX','M2T1NXRAD','M2I3NPASM']
 for d in DAYS:
  for product in products:
   try:
    j=requests.get('https://cmr.earthdata.nasa.gov/search/granules.json',params={'short_name':product,'temporal':f'{d}T00:00:00Z,{d}T23:59:59Z','page_size':10},timeout=120).json()
    for e in j['feed']['entry']:
     urls=[x['href'] for x in e['links'] if x['href'].startswith('https://data.gesdisc.earthdata.nasa.gov/data/') and x['href'].endswith('.nc4')]
     for url in urls:
      name=url.rsplit('/',1)[-1];dst=out/name.replace('.nc4','_ena.nc4')
      if dst.exists():continue
      room();temp=out/(name+'.part')
      p=subprocess.run(['curl','--fail','--silent','--show-error','--location','--netrc','--cookie-jar',str(ROOT/'earthdata.cookies'),'--cookie',str(ROOT/'earthdata.cookies'),'--retry','3','--max-time','1800','-o',str(temp),url],capture_output=True)
      if p.returncode:raise RuntimeError('download failed')
      rawhash=digest(temp)
      with nc.Dataset(temp) as src,nc.Dataset(str(dst)+'.part','w') as target:
       lat=np.asarray(src['lat'][:]);lon=np.asarray(src['lon'][:]);iy=np.where((lat>=config['area'][2])&(lat<=config['area'][0]))[0];ix=np.where((lon>=config['area'][1])&(lon<=config['area'][3]))[0]
       slices={'lat':slice(iy[0],iy[-1]+1),'lon':slice(ix[0],ix[-1]+1)}
       target.setncatts({a:src.getncattr(a) for a in src.ncattrs()});target.source_download_url=url;target.source_sha256=rawhash;target.subset_bounds=str(config['area'])+' (north, west, south, east); native times and vertical levels'
       for k,v in src.dimensions.items():target.createDimension(k,len(iy) if k=='lat' else len(ix) if k=='lon' else len(v))
       for k,v in src.variables.items():
        attrs={a:v.getncattr(a) for a in v.ncattrs()};fill=attrs.pop('_FillValue',None)
        w=target.createVariable(k,v.dtype,v.dimensions,zlib=True,fill_value=fill);w.setncatts(attrs);w[:]=v[tuple(slices.get(dim,slice(None)) for dim in v.dimensions)]
      os.replace(str(dst)+'.part',dst);temp.unlink();record('files',source='MERRA2',url=url,source_sha256=rawhash,file=str(dst.relative_to(ROOT)),sha256=digest(dst),bytes=dst.stat().st_size)
   except Exception as e:record('errors',source='MERRA2',product=product,date=str(d),error=type(e).__name__)
if __name__=='__main__':globals()[args.source]();record('finished',source=args.source,utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
