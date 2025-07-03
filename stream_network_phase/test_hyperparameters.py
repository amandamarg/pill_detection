
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
                         use_gpu_if_available, setup_logger, load_config_json, find_latest_file_in_latest_directory)
from train_stream_network import TrainModel
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Subset
from train_stream_network import PredictStreamNetwork




class ExpirimentModel(TrainModel):
    def __init__(self, cfg, mapping, train_data_loader, valid_data_loader):
        self.cfg = cfg

        # Set up logger
        self.logger = setup_logger()

        # Set up tqdm colour
        colorama.init()

        # Create time stamp
        self.timestamp = create_timestamp()

        # Select the GPU if possible
        self.device = use_gpu_if_available()


        # Setup network config
        self.dataset_type = self.cfg.get("dataset_type")

        stream_type = self.cfg.get("type_of_stream")
        self.type_of_net = self.cfg.get("type_of_net")

        substream_network_cfg = self.cfg.get("streams").get(stream_type)
        backbone_network_cfg = self.cfg.get("networks").get(self.type_of_net)

        # Loss type
        loss_type = self.cfg.get("type_of_loss_func")

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

        self.mapping = mapping

        self.train_data_loader = train_data_loader

        self.valid_data_loader = valid_data_loader

        # Set up loss and mining functions
        if loss_type == "dmtl":
            path_to_excel_file = nlp_configs().get("vector_distances")
            df = get_embedded_text_matrix(path_to_excel_file)

            self.criterion = (
                DynamicMarginTripletLoss(
                    margin=self.cfg.get("margin"),
                    triplets_per_anchor="all",
                    euc_dist_mtx=df,
                    upper_norm_limit=self.cfg.get("upper_norm_limit"),
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
                margin=self.cfg.get("margin"),
                type_of_triplets=self.cfg.get("mining_type")
            )
        )

        # Specify optimizer
        self.optimizer = (
            torch.optim.Adam(
                self.model.parameters(),
                lr=backbone_network_cfg.get("learning_rate")
            )
        )

        # LR scheduler
        self.scheduler = (
            StepLR(
                optimizer=self.optimizer,
                step_size=self.cfg.get("step_size"),
                gamma=self.cfg.get("gamma")
            )
        )

        # Tensorboard
        tensorboard_log_dir = (
            self.create_save_dirs(
                network_cfg=substream_paths().get(stream_type),
                subdir="logs_dir",
                loss=loss_type
            )
        )
        self.writer = (
            SummaryWriter(
                log_dir=tensorboard_log_dir
            )
        )

        # Create save directory for model weights
        self.save_path = (
            self.create_save_dirs(
                network_cfg=substream_paths().get(stream_type),
                subdir="model_weights_dir",
                loss=loss_type
            )
        )

        # Create save directory for hard samples
        self.hard_samples_path = (
            self.create_save_dirs(
                network_cfg=substream_paths().get(stream_type),
                subdir="hardest_samples",
                loss=loss_type
            )
        )

        # Variables to save only the best weights and model
        self.best_valid_loss = float('inf')
        self.best_model_path = None
    
    def create_save_dirs(self, network_cfg, subdir, loss) -> str:

        directory_path = network_cfg.get(self.dataset_type).get(self.type_of_net).get(subdir).get(loss)
        directory_to_create = (
            os.path.join(directory_path, "expiriments", f"{self.timestamp}")
        )
        os.makedirs(directory_to_create, exist_ok=True)
        return directory_to_create


class Estimator:
    def __init__(self, epochs, batch_size, weight_decay, step_size, learning_rate, gamma, margin, mining_type, upper_norm_limit=None):
        self.cfg = (
            load_config_json(
                json_schema_filename=json_config_selector("stream_net").get("schema"),
                json_filename=json_config_selector("stream_net").get("config")
            )
        )
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.step_size = step_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.margin = margin
        self.mining_type = mining_type
        self.upper_norm_limit = upper_norm_limit
            

    def fit(self, X, y=None):
        
        for key, val in self.get_params():
            self.cfg[key] = val

        for stream_type in ["Contour", "LBP", "RGB", "Texture"]:
            self.cfg["type_of_stream"] = stream_type
        
            try:
                em = ExpirimentModel(self.cfg, mapping, train_data_loader, valid_data_loader)
                try:
                    em.training()
                except torch.cuda.OutOfMemoryError:
                    logging.error('Detected OutOfMemoryError!')
                    torch.cuda.empty_cache()
            except KeyboardInterrupt as kbe:
                logging.error("Keyboard interrupt, program has been shut down!")

        self.is_fitted_ = True
        return True


    def predict(self, X):
        dataset_type = self.cfg.get(dataset_type)
        network_type = self.cfg.get(network_type)
        loss_type = self.cfg.get("type_of_loss_func")

        #load networks
        for stream_type in ["Contour", "LBP", "RGB", "Texture"]:
            self.cfg["type_of_stream"] = stream_type
            weight_file_path = substream_paths().get(stream_type).get(dataset_type).get(network_type).get("model_weights_dir").get(loss_type)
            latest_pt_file = find_latest_file_in_latest_directory(
                path=weight_file_path
            )
            network = StreamNetworkFactory.create_network(network_type, self.cfg)



