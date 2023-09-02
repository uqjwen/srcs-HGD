import numpy as np 
import glob

datasets = ['Musical_Instruments','Grocery_and_Gourmet_Food', 'Video_Games', 'Sports_and_Outdoors']

# Musical_Instruments
num = 1
def get_scalar():
	scalars = [str(float(i*2)) for i in range(6)]
	fr = open('scalar.txt', 'w')

	first_line = format('scalar','<25')
	for scalar in scalars:
		first_line += '\t'+format(scalar,'<6')
	fr.write(first_line+'\n')

	for dataset in datasets:
		filenames = ['tradeoff_1.0_scalar_'+scalar+'_hislen_20' for scalar in scalars]
		# fr.write(format('scalar','<25')+'\t'+'\t'.join(scalars)+'\n')
		str_scalar = ''
		for filename in filenames:
			path = '../ckpt/'+dataset+'/'+filename+'/result.txt'
			res  = read_file(path)
			mae = res[:,2]
			temp = np.mean(np.sort(mae)[:num])
			str_scalar += '\t'+format('%.3f'%temp,'<6')

		fr.write(format(dataset,'<25')+str_scalar+'\n')
	fr.close()


def get_tradeoff():
	
	fr = open('tradeoff.txt', 'w')
	tradeoffs = [str(i*2./10) for i in range(6)]
	first_line = format('tradeoff','<25')
	for tradeoff in tradeoffs:
		first_line += '\t'+format(tradeoff,'<6')
	fr.write(first_line+'\n')

	for dataset in datasets:
		filenames = ['tradeoff_'+tradeoff+'_scalar_5.0_hislen_20' for tradeoff in tradeoffs]
		# fr.write(format('scalar','<25')+'\t'+'\t'.join(scalars)+'\n')
		str_scalar = ''
		for filename in filenames:
			path = '../ckpt/'+dataset+'/'+filename+'/result.txt'
			res  = read_file(path)
			mae = res[:,2]
			temp = np.mean(np.sort(mae)[:num])
			str_scalar += '\t'+format('%.3f'%temp,'<6')

		fr.write(format(dataset,'<25')+str_scalar+'\n')
	fr.close()

def get_hislen():
	hislens = [str((i+1)*5) for i in range(6)]
	fr = open('hislen.txt', 'w')

	first_line = format('hislen','<25')
	for hislen in hislens:
		first_line += '\t'+format(hislen,'<6')
	fr.write(first_line+'\n')

	for dataset in datasets:
		filenames = ['tradeoff_1.0_scalar_5.0_hislen_'+hislen for hislen in hislens]
		# fr.write(format('scalar','<25')+'\t'+'\t'.join(scalars)+'\n')
		str_scalar = ''
		for filename in filenames:
			path = '../ckpt/'+dataset+'/'+filename+'/result.txt'
			res  = read_file(path)
			mae = res[:,2]
			temp = np.mean(np.sort(mae)[:num])
			str_scalar += '\t'+format('%.3f'%temp,'<6')

		fr.write(format(dataset,'<25')+str_scalar+'\n')
	fr.close()


def get_mae():
	fr = open('mae.txt', 'w')
	for dataset in datasets:
		path = '../ckpt/'+dataset+'/*'
		dirs = glob.glob(path)
		res_mae = 10.0
		for test_dir in dirs:
			res = read_file(test_dir+'/result.txt')
			mae = res[:,2]
			temp = np.mean(np.sort(mae)[:10])
			if temp < res_mae:
				res_mae = temp 
		fr.write(format(dataset,'<25')+'\t: %.3f'%(res_mae)+'\n')

def read_file(filename):
	fr = open(filename)
	res = []
	for line in fr.readlines():
		line = line.strip()
		line = line.split('\t')
		line = [float(item) for item in line]
		res.append(line)
	return np.array(res)


if __name__ == '__main__':
	# main()
	get_mae()
	get_scalar()
	get_tradeoff()
	get_hislen()