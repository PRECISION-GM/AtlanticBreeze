"""Prepare a single observed sounding interval; no plotting or fitted tendencies."""
import argparse,datetime,json,hashlib
from pathlib import Path
import numpy as np
import netCDF4 as nc
p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,default=Path('data/march2016'));p.add_argument('--output',type=Path,default=Path('data/interval'));a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
root=a.data_dir;used=[]
def load(path):used.append(path);return nc.Dataset(path)
def values(d,k):return np.asarray(np.ma.filled(d[k][:],np.nan),float)
def stamp(d):return values(d,'time')+float(d['base_time'][...])
faces=np.r_[np.arange(0,2500+1,25),np.arange(2600,8000+1,100)];z=(faces[:-1]+faces[1:])/2
np.savetxt(a.output/'faces.csv',faces,delimiter=',')
files=sorted((root/'arm/enasondewnpnC1.b1').glob('*20160304*'))
assert len(files)==2,'Expected exactly the two March 4 soundings; select another pair explicitly.'
starts=[]
for f,label in zip(files,['initial','target']):
 with load(f) as d:
  height=values(d,'alt');height-=height[0];pressure=100*values(d,'pres');T=values(d,'tdry')+273.15;dew=values(d,'dp');e=611.2*np.exp(17.67*dew/(dew+243.5));q=.622*e/(pressure-.378*e)
  data=np.c_[height,pressure,T*(1e5/pressure)**.2857,q,values(d,'u_wind'),values(d,'v_wind')]
  good=np.all(np.isfinite(data),axis=1)&(np.arange(len(height))<=np.nanargmax(height))
  for key in ['pres','tdry','dp','u_wind','v_wind']:good &= values(d,'qc_'+key)==0
  data=data[good];data=data[np.argsort(data[:,0])];data=data[np.r_[True,np.diff(data[:,0])>0]]
  assert data[-1,0]>8000 and data[0,0]<100
  # Average within +/-25 m, retaining the inversion; interpolate where no bin sample exists.
  result=np.column_stack([z]+[np.interp(z,data[:,0],data[:,i]) for i in range(1,6)])
  for k,zz in enumerate(z):
   nearby=abs(data[:,0]-zz)<=25
   if nearby.any():result[k,1:]=data[nearby,1:].mean(axis=0)
  np.savetxt(a.output/(label+'.csv'),result,delimiter=',',header='z_m,p_Pa,theta_d_K,qv_kgkg,u_ms,v_ms',comments='')
  np.savetxt(a.output/(label+'_raw.csv'),data,delimiter=',',header='z_m,p_Pa,theta_d_K,qv_kgkg,u_ms,v_ms',comments='')
  starts.append(float(stamp(d)[0]))
t0,t1=starts;duration=t1-t0
rows=[]
for day in ['20160304','20160305']:
 with load(root/f'era5/era5_pressure_{day}.nc') as d,load(root/f'era5/era5_single_{day}_data_stream-oper_stepType-instant.nc') as s:
  lat=values(d,'latitude');lon=values(d,'longitude');i=np.argmin(abs(lat-39.0916));j=np.argmin(abs(lon+28.0257));latr=np.deg2rad(lat);lonr=np.deg2rad(lon);R=6371000.;f=2*7.292115e-5*np.sin(latr[i]);pressure=values(d,'pressure_level')*100
  times=values(d,'valid_time');slat=values(s,'latitude');slon=values(s,'longitude');si=np.argmin(abs(slat-39.0916));sj=np.argmin(abs(slon+28.0257))
  for it,t in enumerate(times):
   if t<t0-3600 or t>t1+3600:continue
   T=values(d,'t')[it];q=values(d,'q')[it];u=values(d,'u')[it];v=values(d,'v')[it];phi=values(d,'z')[it];theta=T*(1e5/pressure[:,None,None])**.2857
   def dx(x):return (x[:,i,j+1]-x[:,i,j-1])/(R*np.cos(latr[i])*(lonr[j+1]-lonr[j-1]))
   def dy(x):return (x[:,i+1,j]-x[:,i-1,j])/(R*(latr[i+1]-latr[i-1]))
   advtheta=-u[:,i,j]*dx(theta)-v[:,i,j]*dy(theta);advq=-u[:,i,j]*dx(q)-v[:,i,j]*dy(q)
   # Explicit Exner conversion: thermal adapter takes dT/dt, not dtheta/dt.
   advT=advtheta*(pressure/1e5)**.2857
   rho=pressure/(287.05*T[:,i,j]*(1+.61*q[:,i,j]));w=-values(d,'w')[it,:,i,j]/(rho*9.80665)
   heights=phi[:,i,j]/9.80665;order=np.argsort(heights);h=heights[order]
   cols=[np.interp(z,h,x[order]) for x in [advT,advq,w,-dy(phi)/f,dx(phi)/f]]
   # Below the lowest pressure level, hold advection and geostrophic wind but taper w to zero at the surface.
   cols[2]=np.where(z<h[0],cols[2]*z/h[0],cols[2])
   sst=float(values(s,'sst')[it,si,sj]);assert np.isfinite(sst),'Nearest ERA5 point has no ocean SST'
   rows.extend(np.column_stack([np.full(len(z),t-t0),z]+cols+[np.full(len(z),sst)]))
assert min(r[0] for r in rows)<=0 and max(r[0] for r in rows)>=duration,'Forcing does not bracket the sounding interval'
np.savetxt(a.output/'forcing.csv',rows,delimiter=',',header='time_s,z_m,dT_Ks,dq_kgkgs,w_ms,ug_ms,vg_ms,SST_K',comments='')
with load(next((root/'arm/enamwrret2turnC1.c1').glob('*20160304*'))) as d:
 t=stamp(d)-t0;lwp=values(d,'phys_lwp');good=(values(d,'qc_phys_lwp')==0)&(values(d,'phys_qc_flag')==0)&(values(d,'converged')==1)&np.isfinite(lwp)
 bins=[]
 for lo in np.arange(0,duration,600):
  use=(t>=lo)&(t<min(lo+600,duration));ok=use&good
  bins.append([lo,min(lo+600,duration),np.mean(lwp[ok]) if ok.sum()>=10 and ok.sum()>=.5*use.sum() else np.nan,ok.sum(),use.sum()])
 np.savetxt(a.output/'lwp.csv',bins,delimiter=',',header='start_s,end_s,LWP_gm2,n_valid,n_total',comments='')
epoch=datetime.datetime.fromtimestamp(t0,datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
(a.output/'interval.toml').write_text(f'epoch = "{epoch}"\nduration = {duration}\nspinup = 3600.0\nnx = 128\ndx = 100.0\ndt = 1.0\n')
manifest={'sources':[{ 'file':str(f.relative_to(root)),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in dict.fromkeys(used)],'smoothing':'50 m centered average, linear interpolation in empty bins; no artificial mixed-layer replacement','humidity':'sonde dewpoint to specific humidity; initial condensate-free state','forcing':'ERA5 central horizontal differences on pressure surfaces; theta advection converted with Exner; omega converted to w; geostrophic wind from geopotential gradient; nearest ENA gridpoint','loss':'post-spinup QC-screened 10-minute LWP plus final sounding 0–2500m; sonde profile acquisition treated at launch time','known_limits':['No initial cloud-water retrieval','Below the lowest ERA5 level, advection and geostrophic wind hold the lowest value; w tapers linearly to zero at the surface','Paper physical LWP is not necessarily this ARM MWRRET retrieval','Small periodic domain is a candidate, not validated minimum']}
(a.output/'provenance.json').write_text(json.dumps(manifest,indent=2));print(epoch,duration,len(z),'levels;',len(rows)//len(z),'forcing knots')
