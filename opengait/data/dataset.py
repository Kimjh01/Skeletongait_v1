import os
import pickle
import os.path as osp
import torch.utils.data as tordata
import json
from utils import get_msg_mgr


class DataSet(tordata.Dataset):
    def __init__(self, data_cfg, training):
        self.cache = data_cfg['cache']
        self.dataset_root = data_cfg['dataset_root']
        self.training = training
        
        self.__dataset_parser()
        
        self.label_list = [seq_info[0] for seq_info in self.seqs_info]
        self.types_list = [seq_info[1] for seq_info in self.seqs_info]
        self.views_list = [seq_info[2] for seq_info in self.seqs_info]

        self.label_set = sorted(list(set(self.label_list)))
        self.types_set = sorted(list(set(self.types_list)))
        self.views_set = sorted(list(set(self.views_list)))

        self.seqs_data = [None] * len(self)
        self.indices_dict = {label: [] for label in self.label_set}
        for i, seq_info in enumerate(self.seqs_info):
            self.indices_dict[seq_info[0]].append(i)

        if self.cache:
            self.__load_all_data()

    def __len__(self):
        return len(self.seqs_info)

    def __loader__(self, paths):
        paths = sorted(paths)
        data_list = []
        for pth in paths:
            if pth.endswith('.pkl'):
                with open(pth, 'rb') as f:
                    _ = pickle.load(f)
                data_list.append(_)
            else:
                raise ValueError('- Loader - just support .pkl !!!')

        for idx, data in enumerate(data_list):
            if len(data) != len(data_list[0]):
                raise ValueError('Each input data({}) should have the same length.'.format(paths[idx]))
            if len(data) == 0:
                raise ValueError('Each input data({}) should have at least one element.'.format(paths[idx]))

        return data_list

    def __getitem__(self, idx):
        if not self.cache:
            data_list = self.__loader__(self.seqs_info[idx][-1])
        elif self.seqs_data[idx] is None:
            data_list = self.__loader__(self.seqs_info[idx][-1])
            self.seqs_data[idx] = data_list
        else:
            data_list = self.seqs_data[idx]
        seq_info = self.seqs_info[idx]

        print(f"getitem idx={idx}, type(data_list)={type(data_list)}, data_list keys/length: "
          f"{list(data_list.keys()) if isinstance(data_list, dict) else len(data_list)}")
        
        return data_list, seq_info

    def __load_all_data(self):
        for idx in range(len(self)):
            self.__getitem__(idx)

    def __dataset_parser(self):
        msg_mgr = get_msg_mgr()
        dataset_root = self.dataset_root

        all_pids = sorted([d for d in os.listdir(dataset_root) if osp.isdir(osp.join(dataset_root, d))])
        total_count = len(all_pids)
        split_idx = int(total_count * 0.6)  # 예: 60% train, 40% test
        train_set = all_pids[:split_idx]
        test_set = all_pids[split_idx:]

        if self.training:
            selected_pids = train_set
            msg_mgr.log_info(f"Training PIDs ({len(selected_pids)}): {selected_pids[:3]} ... {selected_pids[-3:]}")
        else:
            selected_pids = test_set
            msg_mgr.log_info(f"Testing PIDs ({len(selected_pids)}): {selected_pids[:3]} ... {selected_pids[-3:]}")

        def get_seqs_info_list(label_set):
            seqs_info_list = []
            for lab in label_set:
                lab_path = osp.join(dataset_root, lab)
                if not osp.exists(lab_path):
                    continue
                for typ in sorted(os.listdir(lab_path)):
                    typ_path = osp.join(lab_path, typ)
                    if not osp.exists(typ_path):
                        continue
                    for vie in sorted(os.listdir(typ_path)):
                        vie_path = osp.join(typ_path, vie)
                        if not osp.exists(vie_path):
                            continue
                        seq_info = [lab, typ, vie]

                        # 디렉터리면 하위 파일 리스트로, 파일이면 단일 리스트로
                        if osp.isdir(vie_path):
                            seq_dirs = sorted(os.listdir(vie_path))
                            seq_dirs = [osp.join(vie_path, d) for d in seq_dirs]
                        elif vie_path.endswith('.pkl'):
                            seq_dirs = [vie_path]
                        else:
                            continue

                        if seq_dirs:
                            seqs_info_list.append([*seq_info, seq_dirs])
                        else:
                            msg_mgr.log_debug(f'Find no .pkl file in {lab}-{typ}-{vie}.')
            return seqs_info_list

        self.seqs_info = get_seqs_info_list(selected_pids)

    def __dataset_parser_by_pid(self, data_config, training):
        dataset_root = data_config['dataset_root']
        with open(data_config['dataset_partition'], "r") as f:
            partition = json.load(f)

        pid_list = sorted(list(partition.keys()))
        if training:
            msg_mgr = get_msg_mgr()
            msg_mgr.log_info("-------- Train Pid List --------")
            if len(pid_list) >= 3:
                msg_mgr.log_info('[%s, %s, ..., %s]' % (pid_list[0], pid_list[1], pid_list[-1]))
            else:
                msg_mgr.log_info(pid_list)

        def get_seqs_info_list(label_set):
            seqs_info_list = []
            for lab in label_set:
                lab_path = osp.join(dataset_root, lab)
                if not osp.exists(lab_path):
                    continue
                for typ in sorted(os.listdir(lab_path)):
                    typ_path = osp.join(lab_path, typ)
                    if not osp.exists(typ_path):
                        continue
                    for vie in sorted(os.listdir(typ_path)):
                        vie_path = osp.join(typ_path, vie)
                        if not osp.exists(vie_path):
                            continue
                        seq_info = [lab, typ, vie]

                        if osp.isdir(vie_path):
                            seq_dirs = sorted(os.listdir(vie_path))
                            seq_dirs = [osp.join(vie_path, d) for d in seq_dirs]
                        elif vie_path.endswith('.pkl'):
                            seq_dirs = [vie_path]
                        else:
                            continue

                        if seq_dirs:
                            seqs_info_list.append([*seq_info, seq_dirs])
                        else:
                            msg_mgr.log_debug(f'Find no .pkl file in {lab}-{typ}-{vie}.')
            return seqs_info_list

        self.seqs_info = get_seqs_info_list(pid_list)
