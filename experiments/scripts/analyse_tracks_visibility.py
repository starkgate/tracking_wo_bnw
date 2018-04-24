# Script that analyses the visibiities of persons in the dataset

import _init_paths

import os.path as osp
import os
import numpy as np
import yaml
import time
import csv
import matplotlib.pyplot as plt

from sacred import Experiment
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from tracker.config import get_output_dir
from tracker.mot_tracks import MOT_Tracks

ex = Experiment()

test = ["MOT17-01", "MOT17-03", "MOT17-06", "MOT17-07", "MOT17-08", "MOT17-12", "MOT17-14"]
train = ["MOT17-13", "MOT17-11", "MOT17-10", "MOT17-09", "MOT17-05", "MOT17-04", "MOT17-02", ]
sequences = train

track_dict = {
	'track_len': 7,
	'mult_transitions': 4,
	'max_trail_dead': 6,
	'sample_transitions_backwards': True,
	'min_alive_start': 1,
	'active_thresh': 0.5,
}

@ex.automain
def my_main(_config):
	print(_config)

	# save sacred config to experiment
	output_dir = osp.join(get_output_dir('dataset'), 'visibility_tracks')
	
	if not osp.exists(output_dir):
		os.makedirs(output_dir)
	
	##################
	# Begin analysis #
	##################
	print("[*] Beginning visibility analysis ..")

	num_range = np.arange(0.0,1.0,0.1)

	res = {}
	res['ges'] = []

	# keep track of how many transition tracks there are
	trans = 0
	active = 0

	for s in sequences:
		res[s] = []

		print("[*] Analysing: {}".format(s))
		db = MOT_Tracks(s, generate_blobs=False, **track_dict)
		sampler = WeightedRandomSampler(db.weights, len(db), True)
		dl = DataLoader(db, batch_size=1, shuffle=False, sampler=sampler)

		trans_track = 0
		active_track = 0

		for i,track in enumerate(dl,1):
			for t in track:
				vis = t['vis'][0]
				res['ges'].append(vis)
				res[s].append(vis)
			if track[-1]['active'][0] == True:
				active_track += 1
			else:
				trans_track += 1

		trans += trans_track
		active += active_track

		print("Completelty active: {}/{} tracks".format(active_track, active_track+trans_track))
		print("In Percent: {:.2f}%".format(active_track/(active_track+trans_track)*100))

	print("[*] Whole Dataset")
	print("Completelty active: {}/{} tracks".format(active, active+trans))
	print("In Percent: {:.2f}%".format(active/(active+trans)*100))


	with open(osp.join(output_dir, 'results.csv'), 'w', newline='') as csvfile:
		fieldnames = ['seq_name', '0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
		for k,v in res.items():
			line = {fieldnames[0]: k}
			for i in range(1,len(fieldnames)):
				line[fieldnames[i]] = 0
			for i in v:
				ind = np.where(i >= num_range)[0][-1]
				line[fieldnames[ind+1]] += 1
			writer.writerow(line)


	for k,v in res.items():
		plt.hist(v,bins=10, rwidth=0.95)
		plt.title("Visibility Histogram "+k)
		plt.xlabel("Value")
		plt.ylabel("Occurence")
		plt.draw()
		plt.savefig(osp.join(output_dir, k+'.jpg'), dpi=300)
		plt.close()

