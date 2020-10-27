import configparser
import csv
import os
import os.path as osp

from torch.utils.data import Dataset

class CustomSequence(Dataset):
    """Multiple Object Tracking Dataset.

    This dataloader is designed so that it can handle only one sequence, if more have to be
    handled one should inherit from this class.
    It is also only meant for testing, not training. Hence the lack of groundtruth data and labels.
    """

    def __init__(self, cfg):
        """
        Args:
            seq_name (string): Sequence to take
            vis_threshold (float): Threshold of visibility of persons above which they are selected
        """
        self.folder = cfg['data_path']
        self.no_gt = True

        assert os.path.exists(self.folder), 'Image set does not exist: {}'.format(self.folder)
        self.data = self._sequence()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Return the ith image converted to blob"""
        return self.data[idx]

    def _sequence(self):
        config_file = osp.join(self.folder, 'seqinfo.ini')

        assert osp.exists(config_file), \
            'Config file does not exist: {}'.format(config_file)

        config = configparser.ConfigParser()
        config.read(config_file)
        seqLength = int(config['Sequence']['seqLength'])
        imDir = config['Sequence']['imDir']
        imExt = config['Sequence']['imExt']

        imDir = osp.join(self.folder, imDir)
        total = []

        # frames start at 0
        for i in range(1, seqLength + 1):
            im_path = osp.join(imDir, "{:06d}{:s}".format(i, imExt))
            sample = {'img_path': im_path}
            total.append(sample)

        return total

    def write_results(self, all_tracks, output_dir):
        """Write the tracks in the format for MOT16/MOT17 sumbission

        all_tracks: dictionary with 1 dictionary for every track with {..., i:np.array([x1,y1,x2,y2]), ...} at key track_num

        Each file contains these lines:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        """

        # format_str = "{}, -1, {}, {}, {}, {}, {}, -1, -1, -1"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file = osp.join(output_dir, 'output.txt')

        with open(file, "w") as of:
            writer = csv.writer(of, delimiter=',')
            for i, track in all_tracks.items():
                for frame, bb in track.items():
                    x1 = bb[0]
                    y1 = bb[1]
                    x2 = bb[2]
                    y2 = bb[3]
                    writer.writerow([frame + 1, i + 1, x1 + 1, y1 + 1, x2 - x1 + 1, y2 - y1 + 1, -1, -1, -1, -1])