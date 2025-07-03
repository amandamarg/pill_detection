
import colorama
import logging
import numpy as np
import os
import torch

from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from typing import List, Tuple
from pytorch_metric_learning import losses, miners

from config.json_config import json_config_selector
from config.networks_paths_selector import substream_paths
from config.nlp_paths_selector import nlp_configs
from stream_network_models.stream_network_selector import StreamNetworkFactory
from loss_functions.dynamic_margin_triplet_loss_stream import DynamicMarginTripletLoss
from dataloader_stream_network import DataLoaderStreamNet
from utils.utils import (create_dataset, create_timestamp, get_embedded_text_matrix, measure_execution_time,
                         use_gpu_if_available, setup_logger, load_config_json)
from train_stream_network import TrainModel


cfg = (
        load_config_json(
            json_schema_filename=json_config_selector("stream_net").get("schema"),
            json_filename=json_config_selector("stream_net").get("config")
        )
    )

class ExpirimentModel(TrainModel):
    def __init__(self, dataset, stream_type, loss_type, epochs, batch_size, weight_decay, step_size, learning_rate, gamma, margin, mining_type, upper_norm_limit=None):
        # Set up logger
        self.logger = setup_logger()

        # Set up tqdm colour
        colorama.init()

        # Select the GPU if possible
        self.device = use_gpu_if_available()

        self.type_of_net = cfg.get("type_of_net")

        substream_network_cfg = cfg.get("streams").get(stream_type)
        backbone_network_cfg = cfg.get("networks").get(self.type_of_net)

        # Load model and upload it to the GPU
        self.model = StreamNetworkFactory.create_network(self.type_of_net, substream_network_cfg)
        self.model = self.model.to(self.device)

        # Print model configuration
        summary(
            model=self.model,
            input_size=(
                substream_network_cfg.get('channels')[0],
                backbone_network_cfg.get("image_size"),
                backbone_network_cfg.get("image_size")
            )
        )

        self.mapping = dataset.reference_encoding_map
        self.train_data_loader, self.valid_data_loader = (
            create_dataset(
                dataset=dataset,
                train_valid_ratio=cfg.get("train_valid_ratio"),
                batch_size=batch_size
            )
        )

        # Set up loss and mining functions
        if loss_type == "dmtl":
            path_to_excel_file = nlp_configs().get("vector_distances")
            df = get_embedded_text_matrix(path_to_excel_file)

            #TODO: raise error if no upper_norm_limit passed

            self.criterion = (
                DynamicMarginTripletLoss(
                    margin=margin,
                    triplets_per_anchor="all",
                    euc_dist_mtx=df,
                    upper_norm_limit=upper_norm_limit,
                    mapping_table=self.mapping
                )
            )

        elif loss_type == "hmtl":
            self.criterion = (
                losses.TripletMarginLoss(
                    margin=self.cfg.get("margin")
                )
            )
        else:
            raise ValueError(f"Wrong loss function: {loss_type}")
        
        self.mining_func = (
            miners.TripletMarginMiner(
                margin=margin,
                type_of_triplets=mining_type
            )
        )

        # Specify optimizer
        self.optimizer = (
            torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate
            )
        )


        # LR scheduler
        self.scheduler = (
            StepLR(
                optimizer=self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
        )

        



class Estimator:
    def __init__(self, epochs, batch_size, weight_decay, step_size, learning_rate, gamma, margin, mining_type, upper_norm_limit,):
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.step_size = step_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.margin = margin
        self.mining_type = mining_type
        self.upper_norm_limit = upper_norm_limit

        # Select the GPU if possible
        self.device = use_gpu_if_available()


    def fit(self, X, y=None):
        
        None
    
    def predict(self, X):
        None
