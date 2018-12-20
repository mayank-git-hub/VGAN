import torch
import torch.nn as nn

class generic_model(nn.Module):

	def __init__(self):

		super(generic_model, self).__init__()

	def save(self, no=0, epoch_i=0, info={}, is_best=False, filename='checkpoint.pth.tar', best={}):
		
		if is_best:
			torch.save({'epoch': epoch_i,
					'state_dict': self.state_dict(),
					'optimizer': self.opt.state_dict(),
					'seed': self.config['seed'],
					'best': best},self.config['dir']['Model_Output_Best']+'/'+str(no)+'_'+str(epoch_i)+'_'+filename)
			torch.save(info, self.config['dir']['Model_Output_Best']+'/'+str(no)+'_'+str(epoch_i)+'_'+'info_'+filename)
		else:
			torch.save({'epoch': epoch_i,
					'state_dict': self.state_dict(),
					'optimizer': self.opt.state_dict(),
					'seed': self.config['seed'],
					'best': best},self.config['dir']['Model_Output']+'/'+str(no)+'_'+str(epoch_i)+'_'+filename)
			torch.save(info, self.config['dir']['Model_Output']+'/'+str(no)+'_'+str(epoch_i)+'_'+'info_'+filename)


	def load(self, path, path_info):

		checkpoint = torch.load(path)

		self.load_state_dict(checkpoint['state_dict'])

		if not self.config['optimizer_new']:
			self.opt.load_state_dict(checkpoint['optimizer'])
		
		return checkpoint['epoch'], torch.load(path_info)

class Generator_UNet(generic_model):

	def __init__(self):

		super(Generator_UNet, self).__init__()
		n_features = 100
		n_out = 784
		
		self.hidden0 = nn.Sequential(
			nn.Linear(n_features, 256),
			nn.LeakyReLU(0.2)
		)
		self.hidden1 = nn.Sequential(			
			nn.Linear(256, 512),
			nn.LeakyReLU(0.2)
		)
		self.hidden2 = nn.Sequential(
			nn.Linear(512, 1024),
			nn.LeakyReLU(0.2)
		)
		
		self.out = nn.Sequential(
			nn.Linear(1024, n_out),
			nn.Tanh()
		)

	def forward(self, x):
		x = self.hidden0(x)
		x = self.hidden1(x)
		x = self.hidden2(x)
		x = self.out(x)
		return x

class Discriminator(generic_model):
	"""
	A three hidden-layer discriminative neural network
	"""
	def __init__(self):
		super(Discriminator, self).__init__()
		n_features = 784
		n_out = 1
		
		self.hidden0 = nn.Sequential( 
			nn.Linear(n_features, 1024),
			nn.LeakyReLU(0.2),
			nn.Dropout(0.3)
		)
		self.hidden1 = nn.Sequential(
			nn.Linear(1024, 512),
			nn.LeakyReLU(0.2),
			nn.Dropout(0.3)
		)
		self.hidden2 = nn.Sequential(
			nn.Linear(512, 256),
			nn.LeakyReLU(0.2),
			nn.Dropout(0.3)
		)
		self.out = nn.Sequential(
			torch.nn.Linear(256, n_out),
			torch.nn.Sigmoid()
		)

	def forward(self, x):
		x = self.hidden0(x)
		x = self.hidden1(x)
		x = self.hidden2(x)
		x = self.out(x)
		return x