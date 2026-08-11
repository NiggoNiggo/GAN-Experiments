import torch

class Evaluate:
    def __init__(self,
                 num_samples:int):
        self.eval_strats = []
        self.num_samples = self.num_samples

    #this is our notify
    def compute(self,gen):
        """compute as notify function to compute all desired metrics for evaluation

        Args:
            gen (torch.nn.Module): Generator network
        """
        #iterate through any metric
        for metric in self.eval_strats:
            #compute the metrics with alltogether with the same fake samples
            print(metric.compute(self.generate_fake(gen)))

    def attach(self,metric):
        self.eval_strats.append(metric)

    def generate_fake(self,gen)->torch.tensor:
        """generate fake data for evaluation

        Args:
            gen (Generator) : Generator of any GAN

        Returns:
            fake (torch.tensor) : fake samples (num_samples)
        """
        #without gradients 
        with torch.no_grad():
            #generate num samples in one tensor 
            noise = torch.randn(self.num_samples,100,1,1).cuda()
            #generate fakes (num_samples,c,h,w)
            fakes = gen(noise)
            return fakes

        

            
        