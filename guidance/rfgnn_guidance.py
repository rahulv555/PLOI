import pickle
import os
import numpy as np
from torch.utils.data import DataLoader
import torch.optim
import torch.nn
import torch
import pddlgym
from pddlgym.structs import Predicate
from gnn.gnn import setup_graph_net
from gnn.gnn_dataset import GraphDictDataset, graph_batch_collate
from gnn.gnn_utils import train_model, get_single_model_prediction
from guidance import BaseSearchGuidance
from planning import PlanningTimeout, PlanningFailure


class RF_GNNSearchGuidance():

    def __init__(self, training_planner, num_train_problems, num_epochs,
                 criterion_name, bce_pos_weight, load_from_file,
                 load_dataset_from_file, dataset_file_prefix,
                 save_model_prefix, is_strips_domain, seed):
        super().__init__()
        self._planner = training_planner
        self._num_train_problems = num_train_problems
        self._num_epochs = num_epochs
        self._criterion_name = criterion_name
        self._bce_pos_weight = bce_pos_weight
        self._load_from_file = load_from_file
        self._load_dataset_from_file = load_dataset_from_file
        self._dataset_file_prefix = dataset_file_prefix
        self._save_model_prefix = save_model_prefix
        self._is_strips_domain = is_strips_domain
        self._seed = seed
        # Initialize other instance variables.
        self._num_of_S = 10

    def train(self, train_env_name):
        model_outfile = "rf"+self._save_model_prefix + \
            "_{}.pt".format(train_env_name)
        print("Training search guidance {} in domain {}...".format(
            self.__class__.__name__, train_env_name))
        # Collect raw training data. Inputs are States, outputs are objects.
        training_data = self._collect_training_data(train_env_name)

        # Convert training data to graphs
        graphs_input, graphs_target = self._create_graph_dataset(training_data)
        # Use 10% for validation
        num_validation = max(1, int(len(graphs_input)*0.1))
        train_graphs_input = graphs_input[num_validation:]
        train_graphs_target = graphs_target[num_validation:]
        valid_graphs_input = graphs_input[:num_validation]
        valid_graphs_target = graphs_target[:num_validation]
        # Set up dataloaders
        graph_dataset = GraphDictDataset(train_graphs_input,
                                         train_graphs_target)
        graph_dataset_val = GraphDictDataset(valid_graphs_input,
                                             valid_graphs_target)
        dataloader = DataLoader(graph_dataset, batch_size=16, shuffle=False,
                                num_workers=3, collate_fn=graph_batch_collate)
        dataloader_val = DataLoader(graph_dataset_val, batch_size=16,
                                    shuffle=False, num_workers=3,
                                    collate_fn=graph_batch_collate)
        dataloaders = {"train": dataloader, "val": dataloader_val}


    


    def seed(self, seed):
        torch.manual_seed(seed)



    def _collect_training_data(self, train_env_name):
        """Returns X, Y where X are States and Y are sets of objects
        """
        outfile = self._dataset_file_prefix + "_{}.pkl".format(train_env_name)
        if not self._load_dataset_from_file or not os.path.exists(outfile):
            inputs = []
            outputs = []
            env = pddlgym.make("PDDLEnv{}-v0".format(train_env_name))
            assert env.operators_as_actions
            for idx in range(min(self._num_train_problems, len(env.problems))):
                print("Collecting training data problem {}".format(idx),
                      flush=True)
                env.fix_problem_index(idx)
                state, _ = env.reset()
                
                try:
                    plan = self._planner(env.domain, state, timeout=500)
                except (PlanningTimeout, PlanningFailure):
                    print("Warning: planning failed, skipping: {}".format(
                        env.problems[idx].problem_fname))
                    continue
                inputs.append(state)
                objects_in_plan = {o for act in plan for o in act.variables}
                outputs.append(objects_in_plan)
            training_data = (inputs, outputs)

            with open(outfile, "wb") as f:
                pickle.dump(training_data, f)

        with open(outfile, "rb") as f:
            training_data = pickle.load(f)

        return training_data