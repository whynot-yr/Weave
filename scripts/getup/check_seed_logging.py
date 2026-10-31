"""Two-process weighted seed statistics test; no simulator or GPU required."""
import json,sys,tempfile
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'source/g1_hoi_learning/g1_hoi_learning/tasks/getup'))
from seed_logging import reduce_seed_statistics,seed_metrics

def worker(rank,init_file):
 dist.init_process_group('gloo',init_method='file://'+init_file,rank=rank,world_size=2)
 x=torch.zeros(2,9,dtype=torch.float64)
 if rank==0:x[0]=torch.tensor([3,3,1,1,30,3,6,9,12])
 else:
  x[0]=torch.tensor([1,1,1,0,20,5,10,15,20]);x[1]=torch.tensor([2,2,2,1,50,8,10,12,14])
 sums=reduce_seed_statistics(x);metrics=seed_metrics(sums)
 assert metrics['Seed/supine/success_rate']==.5
 assert metrics['Seed/supine/return_task']==2
 assert metrics['Seed/prone/success_rate']==1
 assert metrics['Seed/prone/episode_count']==2
 assert metrics['Seed/supine/reset_fraction']==4/6
 if rank==0:print(json.dumps({'passed':True,'metrics':metrics}),flush=True)
 dist.destroy_process_group()

if __name__=='__main__':
 assert 'Seed/prone/success_rate' not in seed_metrics(torch.zeros(2,9))
 with tempfile.TemporaryDirectory() as d:mp.spawn(worker,args=(str(Path(d)/'init'),),nprocs=2,join=True)
