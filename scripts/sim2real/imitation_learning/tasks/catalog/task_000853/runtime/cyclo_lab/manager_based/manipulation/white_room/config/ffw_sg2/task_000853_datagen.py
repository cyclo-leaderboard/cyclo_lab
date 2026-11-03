"""Export based on the final supported state, not transient success mid-episode."""
import torch
import traceback
import sys
from isaaclab_mimic.datagen.data_generator import DataGenerator

class Task000853DataGenerator(DataGenerator):
    async def generate(self,env_id,success_term,**kwargs):
        export=kwargs.pop('export_demo',True)
        try:
            result=await super().generate(env_id,success_term,export_demo=False,**kwargs)
        except Exception:
            traceback.print_exc(file=sys.stderr);sys.stderr.flush()
            raise
        success=self.env.get_generation_success(env_id,success_term)
        result['success']=success
        ids=torch.tensor([env_id],device=self.env.device)
        self.env.recorder_manager.set_success_to_episodes(ids,torch.tensor([[success]],device=self.env.device))
        if export:self.env.recorder_manager.export_episodes(ids)
        return result
