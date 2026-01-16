"""
MLP Classifier with Contrastive Learning Regularization

This module extends the base MLPClassifier to include a contrastive learning
regularization term. The model learns representations where:
- Samples with the same prediction are close in the embedding space
- Samples with different predictions are far apart

The embedding space is defined by the activations of the penultimate layer.
"""

from typing import Union, Tuple
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from .utils import array_to_tensor, bootstrap_data


# ==============================================================================
# CONTRASTIVE LOSS CONFIGURATION
# ==============================================================================

class ContrastiveLossConfig:
    """Configuration for the contrastive loss term."""
    
    # Weight of the contrastive loss term (lambda)
    # Total loss = BCE_loss + lambda_contrastive * contrastive_loss
    lambda_contrastive: float = 0.1
    
    # Margin for contrastive loss (minimum distance between different classes)
    margin: float = 1.0
    
    # Temperature for supervised contrastive loss (if using InfoNCE-style loss)
    temperature: float = 0.5
    
    # Type of contrastive loss: 'simple', 'supervised', 'triplet'
    loss_type: str = 'supervised'


def contrastive_loss_simple(embeddings: torch.Tensor, labels: torch.Tensor, 
                            margin: float = 1.0) -> torch.Tensor:
    """
    Simple contrastive loss (pairwise).
    
    For pairs with same label: minimize distance
    For pairs with different labels: maximize distance (up to margin)
    
    Parameters:
    -----------
    embeddings : torch.Tensor
        Embeddings from penultimate layer, shape (batch_size, embedding_dim)
    labels : torch.Tensor
        Binary labels, shape (batch_size,)
    margin : float
        Margin for dissimilar pairs
        
    Returns:
    --------
    torch.Tensor
        Scalar contrastive loss
    """
    batch_size = embeddings.shape[0]
    if batch_size < 2:
        return torch.tensor(0.0, device=embeddings.device)
    
    # Compute pairwise distances
    distances = torch.cdist(embeddings, embeddings, p=2)  # (batch_size, batch_size)
    
    # Create label similarity matrix (1 if same label, 0 if different)
    labels = labels.view(-1, 1)
    label_match = (labels == labels.T).float()
    
    # Mask diagonal (self-comparisons)
    mask = 1 - torch.eye(batch_size, device=embeddings.device)
    
    # Loss for similar pairs: minimize distance
    similar_loss = (label_match * mask * distances.pow(2)).sum()
    
    # Loss for dissimilar pairs: maximize distance up to margin
    dissimilar_loss = ((1 - label_match) * mask * F.relu(margin - distances).pow(2)).sum()
    
    # Normalize by number of pairs
    num_pairs = mask.sum()
    total_loss = (similar_loss + dissimilar_loss) / (num_pairs + 1e-8)
    
    return total_loss


def supervised_contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor,
                                 temperature: float = 0.5) -> torch.Tensor:
    """
    Supervised Contrastive Loss (SupCon).
    
    Based on: "Supervised Contrastive Learning" (Khosla et al., 2020)
    
    Parameters:
    -----------
    embeddings : torch.Tensor
        Embeddings from penultimate layer, shape (batch_size, embedding_dim)
    labels : torch.Tensor
        Binary labels, shape (batch_size,)
    temperature : float
        Temperature scaling parameter
        
    Returns:
    --------
    torch.Tensor
        Scalar supervised contrastive loss
    """
    batch_size = embeddings.shape[0]
    if batch_size < 2:
        return torch.tensor(0.0, device=embeddings.device)
    
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Compute similarity matrix
    similarity = torch.matmul(embeddings, embeddings.T) / temperature
    
    # Create mask for positive pairs (same label, excluding self)
    labels = labels.view(-1, 1)
    mask_positive = (labels == labels.T).float()
    mask_self = torch.eye(batch_size, device=embeddings.device)
    mask_positive = mask_positive - mask_self  # Exclude self
    
    # For numerical stability
    logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
    logits = similarity - logits_max.detach()
    
    # Compute log_prob
    exp_logits = torch.exp(logits) * (1 - mask_self)  # Exclude self
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
    
    # Compute mean of log-likelihood over positive pairs
    num_positives = mask_positive.sum(dim=1)
    mean_log_prob_pos = (mask_positive * log_prob).sum(dim=1) / (num_positives + 1e-8)
    
    # Only consider samples that have at least one positive pair
    valid_samples = num_positives > 0
    if valid_samples.sum() == 0:
        return torch.tensor(0.0, device=embeddings.device)
    
    loss = -mean_log_prob_pos[valid_samples].mean()
    
    return loss


def triplet_loss(embeddings: torch.Tensor, labels: torch.Tensor,
                 margin: float = 1.0) -> torch.Tensor:
    """
    Triplet Loss with online hard mining.
    
    Parameters:
    -----------
    embeddings : torch.Tensor
        Embeddings from penultimate layer, shape (batch_size, embedding_dim)
    labels : torch.Tensor
        Binary labels, shape (batch_size,)
    margin : float
        Margin for triplet loss
        
    Returns:
    --------
    torch.Tensor
        Scalar triplet loss
    """
    batch_size = embeddings.shape[0]
    if batch_size < 2:
        return torch.tensor(0.0, device=embeddings.device)
    
    # Compute pairwise distances
    distances = torch.cdist(embeddings, embeddings, p=2)
    
    # Create masks for positive and negative pairs
    labels = labels.view(-1, 1)
    mask_positive = (labels == labels.T).float() - torch.eye(batch_size, device=embeddings.device)
    mask_negative = (labels != labels.T).float()
    
    # For each anchor, find hardest positive and hardest negative
    # Hardest positive: furthest sample with same label
    hardest_positive_dist = (distances * mask_positive).max(dim=1)[0]
    
    # Hardest negative: closest sample with different label
    # Set positive pairs to large distance so they're not selected
    distances_neg = distances + mask_positive * 1e9
    hardest_negative_dist = distances_neg.min(dim=1)[0]
    
    # Triplet loss
    losses = F.relu(hardest_positive_dist - hardest_negative_dist + margin)
    
    # Only consider valid triplets (samples that have both pos and neg)
    valid = (mask_positive.sum(dim=1) > 0) & (mask_negative.sum(dim=1) > 0)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=embeddings.device)
    
    return losses[valid].mean()


def get_contrastive_loss(loss_type: str = 'supervised'):
    """
    Get the contrastive loss function based on type.
    
    Parameters:
    -----------
    loss_type : str
        Type of contrastive loss: 'simple', 'supervised', 'triplet'
        
    Returns:
    --------
    callable
        The contrastive loss function
    """
    if loss_type == 'simple':
        return contrastive_loss_simple
    elif loss_type == 'supervised':
        return supervised_contrastive_loss
    elif loss_type == 'triplet':
        return triplet_loss
    else:
        raise ValueError(f"Unknown contrastive loss type: {loss_type}")


# ==============================================================================
# MLP CLASSIFIER WITH CONTRASTIVE REGULARIZATION
# ==============================================================================


class MLPClassifierContrastive(nn.Module):
    """
    MLP Classifier with Contrastive Learning Regularization.
    
    This model learns representations where samples with the same prediction
    are close together and samples with different predictions are far apart
    in the embedding space (penultimate layer activations).
    
    Loss = BCE_loss + lambda_contrastive * contrastive_loss
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_layers: int,
        activation: str,
        neurons_per_layer: int,
        seed: int,
        dropout: float = 0.2,
        lambda_contrastive: float = 0.1,
        contrastive_margin: float = 1.0,
        contrastive_temperature: float = 0.5,
        contrastive_loss_type: str = 'supervised',
    ) -> None:
        """
        Parameters:
        -----------
        input_dim: int
            Input dimension
        hidden_layers: int
            Number of hidden layers
        activation: str
            Activation function ('relu', 'tanh', 'sigmoid')
        neurons_per_layer: int
            Number of neurons per hidden layer
        seed: int
            Random seed for reproducibility
        dropout: float
            Dropout rate (default: 0.2)
        lambda_contrastive: float
            Weight of contrastive loss term (default: 0.1)
        contrastive_margin: float
            Margin for contrastive/triplet loss (default: 1.0)
        contrastive_temperature: float
            Temperature for supervised contrastive loss (default: 0.5)
        contrastive_loss_type: str
            Type of contrastive loss: 'simple', 'supervised', 'triplet'
        """
        super(MLPClassifierContrastive, self).__init__()

        self.input_dim = input_dim
        self.output_dim = 1
        self.activation = activation
        self.dropout = dropout
        self.hidden_layers = [neurons_per_layer for _ in range(hidden_layers)]
        
        # Contrastive learning parameters
        self.lambda_contrastive = lambda_contrastive
        self.contrastive_margin = contrastive_margin
        self.contrastive_temperature = contrastive_temperature
        self.contrastive_loss_type = contrastive_loss_type

        # Layers will be split into:
        # - feature_layers: all layers up to and including the penultimate layer
        # - output_layer: final linear + sigmoid
        self.feature_layers = nn.ModuleList()
        self.output_layer = nn.Sequential()

        torch.manual_seed(seed)

        self.build_model()
        
        # Get the contrastive loss function
        self.contrastive_loss_fn = get_contrastive_loss(contrastive_loss_type)

    def build_model(self):
        """Build the model architecture with separate feature and output layers."""
        input_dim = self.input_dim
        
        # Build all hidden layers (these will produce the embeddings)
        for i, hidden_dim in enumerate(self.hidden_layers):
            self.feature_layers.append(nn.Linear(input_dim, hidden_dim))
            
            if self.activation == "relu":
                self.feature_layers.append(nn.ReLU())
            elif self.activation == "tanh":
                self.feature_layers.append(nn.Tanh())
            elif self.activation == "sigmoid":
                self.feature_layers.append(nn.Sigmoid())
            else:
                raise ValueError("Invalid activation function")
            
            if self.dropout > 0:
                self.feature_layers.append(nn.Dropout(self.dropout))
            
            input_dim = hidden_dim
        
        # Output layer (from penultimate to output)
        self.output_layer = nn.Sequential(
            nn.Linear(input_dim, self.output_dim),
            nn.Sigmoid()
        )
        
        # Store embedding dimension for reference
        self.embedding_dim = input_dim

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the embeddings (activations of the penultimate layer).
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor, shape (batch_size, input_dim)
            
        Returns:
        --------
        torch.Tensor
            Embeddings, shape (batch_size, embedding_dim)
        """
        for layer in self.feature_layers:
            x = layer(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning only predictions.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor
            
        Returns:
        --------
        torch.Tensor
            Predictions, shape (batch_size,)
        """
        embeddings = self.get_embeddings(x)
        output = self.output_layer(embeddings)
        return output.flatten()
    
    def forward_with_embeddings(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both predictions and embeddings.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor
            
        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor]
            (predictions, embeddings)
        """
        embeddings = self.get_embeddings(x)
        output = self.output_layer(embeddings)
        return output.flatten(), embeddings

    def predict_proba(self, x) -> np.ndarray:
        """Predict probabilities."""
        self.eval()
        x = array_to_tensor(x)
        with torch.no_grad():
            return self.forward(x).detach().numpy().flatten()

    def predict_crisp(self, x, threshold=0.5) -> np.ndarray:
        """Predict crisp class labels."""
        self.eval()
        x = array_to_tensor(x)
        with torch.no_grad():
            pred = self.predict_proba(x)
            if isinstance(pred, np.ndarray):
                return (pred > threshold).astype(int).flatten()
            return (pred > threshold).int().detach().numpy().flatten()
    
    def get_embedding_numpy(self, x) -> np.ndarray:
        """
        Get embeddings as numpy array.
        
        Parameters:
        -----------
        x : array-like
            Input data
            
        Returns:
        --------
        np.ndarray
            Embeddings, shape (batch_size, embedding_dim)
        """
        self.eval()
        x = array_to_tensor(x)
        with torch.no_grad():
            return self.get_embeddings(x).detach().numpy()

    def compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, 
                     embeddings: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute the total loss: BCE + contrastive regularization.
        
        Parameters:
        -----------
        y_pred : torch.Tensor
            Predicted probabilities
        y_true : torch.Tensor
            Ground truth labels
        embeddings : torch.Tensor
            Embeddings from penultimate layer
            
        Returns:
        --------
        Tuple[torch.Tensor, dict]
            Total loss and dictionary with individual loss components
        """
        # BCE Loss
        bce_loss = nn.BCELoss()(y_pred, y_true)
        
        # Contrastive Loss
        if self.contrastive_loss_type == 'supervised':
            contrastive_loss = self.contrastive_loss_fn(
                embeddings, y_true, temperature=self.contrastive_temperature
            )
        else:
            contrastive_loss = self.contrastive_loss_fn(
                embeddings, y_true, margin=self.contrastive_margin
            )
        
        # Total loss
        total_loss = bce_loss + self.lambda_contrastive * contrastive_loss
        
        loss_dict = {
            'total': total_loss.item(),
            'bce': bce_loss.item(),
            'contrastive': contrastive_loss.item(),
        }
        
        return total_loss, loss_dict

    def fit(
        self,
        X_train: Union[np.ndarray, torch.Tensor],
        y_train: Union[np.ndarray, torch.Tensor],
        X_val: Union[np.ndarray, torch.Tensor] = None,
        y_val: Union[np.ndarray, torch.Tensor] = None,
        epochs: int = 100,
        lr: float = 0.002,
        batch_size: int = 256,
        verbose: bool = True,
        early_stopping: bool = True,
        optimizer: str = "adam",
        device: str = "cpu",
    ) -> dict:
        """
        Train the model with BCE + Contrastive Loss.
        
        Returns:
        --------
        dict
            Training history with loss values
        """
        if optimizer == "adam":
            optim = torch.optim.Adam(self.parameters(), lr=lr)
        elif optimizer == "sgd":
            optim = torch.optim.SGD(self.parameters(), lr=lr)
        else:
            raise ValueError("Invalid optimizer")

        # Reshape y_train
        if len(y_train.shape) == 2:
            y_train = y_train.flatten()
            if y_val is not None:
                y_val = y_val.flatten()

        # Convert to tensors
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        if X_val is not None:
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            if isinstance(y_val, pd.Series):
                y_val = y_val.values

        X_train = array_to_tensor(X_train, device=device, dtype=torch.float32)
        y_train = array_to_tensor(y_train, device=device, dtype=torch.float32)
        if X_val is not None:
            X_val = array_to_tensor(X_val, device=device, dtype=torch.float32)
            y_val = array_to_tensor(y_val, device=device, dtype=torch.float32)

        history = {'train_loss': [], 'train_bce': [], 'train_contrastive': [],
                   'val_loss': [], 'val_bce': [], 'val_contrastive': []}
        
        early_stopping_patience = 5
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            self.train()
            epoch_losses = {'total': [], 'bce': [], 'contrastive': []}
            
            # Shuffle data
            perm = torch.randperm(len(X_train))
            X_train_shuffled = X_train[perm]
            y_train_shuffled = y_train[perm]
            
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train_shuffled[i:i + batch_size]
                y_batch = y_train_shuffled[i:i + batch_size]
                
                optim.zero_grad()
                
                # Forward pass with embeddings
                y_pred, embeddings = self.forward_with_embeddings(X_batch)
                
                # Compute loss
                loss, loss_dict = self.compute_loss(y_pred, y_batch, embeddings)
                
                loss.backward()
                optim.step()
                
                epoch_losses['total'].append(loss_dict['total'])
                epoch_losses['bce'].append(loss_dict['bce'])
                epoch_losses['contrastive'].append(loss_dict['contrastive'])

            # Record training losses
            history['train_loss'].append(np.mean(epoch_losses['total']))
            history['train_bce'].append(np.mean(epoch_losses['bce']))
            history['train_contrastive'].append(np.mean(epoch_losses['contrastive']))

            if verbose and epoch % 10 == 0:
                logging.info(
                    f"Epoch {epoch}: Loss={history['train_loss'][-1]:.4f}, "
                    f"BCE={history['train_bce'][-1]:.4f}, "
                    f"Contrastive={history['train_contrastive'][-1]:.4f}"
                )

            # Validation
            if early_stopping and X_val is not None:
                self.eval()
                with torch.no_grad():
                    y_pred_val, embeddings_val = self.forward_with_embeddings(X_val)
                    val_loss, val_loss_dict = self.compute_loss(y_pred_val, y_val, embeddings_val)
                    
                    history['val_loss'].append(val_loss_dict['total'])
                    history['val_bce'].append(val_loss_dict['bce'])
                    history['val_contrastive'].append(val_loss_dict['contrastive'])

                    if verbose and epoch % 5 == 0:
                        logging.debug(
                            f"Epoch {epoch}: Val Loss={val_loss_dict['total']:.4f}, "
                            f"Val BCE={val_loss_dict['bce']:.4f}, "
                            f"Val Contrastive={val_loss_dict['contrastive']:.4f}"
                        )

                    if val_loss_dict['total'] < best_val_loss:
                        best_val_loss = val_loss_dict['total']
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= early_stopping_patience:
                        logging.debug("Early stopping due to validation loss not improving.")
                        break
        
        return history

    def evaluate(
        self,
        X_test: Union[np.ndarray, torch.Tensor],
        y_test: Union[np.ndarray, torch.Tensor],
        device: str = "cpu",
    ) -> dict:
        """Evaluate the model on test data."""
        self.eval()
        X_test = array_to_tensor(X_test, device=device)
        y_test = array_to_tensor(y_test, device=device)
        y_pred = self.predict_crisp(X_test)
        
        accuracy = accuracy_score(y_test.numpy(), y_pred)
        recall = recall_score(y_test.numpy(), y_pred, average="binary")
        precision = precision_score(y_test.numpy(), y_pred, average="binary")
        f1 = f1_score(y_test.numpy(), y_pred, average="binary")

        return {
            "accuracy": accuracy,
            "recall": recall,
            "precision": precision,
            "f1": f1,
        }


def train_neural_network_contrastive(
    X_train, y_train, seed: int, hparams: dict, split: float = 0.8,
    lambda_contrastive: float = 0.1,
    contrastive_loss_type: str = 'supervised',
    contrastive_margin: float = 1.0,
    contrastive_temperature: float = 0.5,
) -> tuple:
    """
    Train a neural network with contrastive regularization.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    seed : int
        Random seed
    hparams : dict
        Hyperparameters for the model
    split : float
        Train/validation split ratio
    lambda_contrastive : float
        Weight of contrastive loss term
    contrastive_loss_type : str
        Type of contrastive loss: 'simple', 'supervised', 'triplet'
    contrastive_margin : float
        Margin for contrastive/triplet loss
    contrastive_temperature : float
        Temperature for supervised contrastive loss
    
    Returns:
    --------
    tuple
        (model, predict_proba_fn, predict_crisp_fn)
    """

    logging.debug("Training torch model with contrastive regularization")

    # Initialize the model
    model = MLPClassifierContrastive(
        input_dim=X_train.shape[1],
        hidden_layers=hparams["hidden_layers"],
        activation=hparams["activation"],
        dropout=hparams["dropout"],
        neurons_per_layer=hparams["neurons_per_layer"],
        seed=seed,
        lambda_contrastive=lambda_contrastive,
        contrastive_loss_type=contrastive_loss_type,
        contrastive_margin=contrastive_margin,
        contrastive_temperature=contrastive_temperature,
    )

    # Create a validation set internally
    X_train1, X_val1, y_train1, y_val1 = train_test_split(
        X_train, y_train, test_size=1 - split, random_state=seed
    )

    # Train the model
    history = model.fit(
        X_train1,
        y_train1,
        X_val=X_val1,
        y_val=y_val1,
        epochs=hparams["epochs"],
        lr=hparams["lr"],
        batch_size=hparams["batch_size"],
        verbose=hparams["verbose"],
        early_stopping=hparams["early_stopping"],
    )

    ret = model.evaluate(X_val1, y_val1)
    logging.debug(f"Validation set metrics: {ret}")

    predict_fn_1 = lambda x: model.predict_proba(x)
    predict_fn_1_crisp = lambda x: model.predict_crisp(
        x, threshold=hparams["classification_threshold"]
    )

    return model, predict_fn_1, predict_fn_1_crisp


def train_K_mlps_contrastive(
    X_train,
    y_train,
    X_test,
    y_test,
    hparams_list: list,
    seeds: list,
    bootstrap_seeds: list,
    K: int = 5,
    lambda_contrastive: float = 0.1,
    contrastive_loss_type: str = 'supervised',
) -> dict:
    """
    Train K MLP models with contrastive regularization.
    
    Parameters:
    -----------
    X_train : np.array
        Training data
    y_train : np.array
        Training labels
    X_test : np.array
        Test data
    y_test : np.array
        Test labels
    hparams_list : list[dict]
        List of hyperparameters for each model
    seeds : list[int]
        List of seeds for each model
    bootstrap_seeds : list[int]
        List of bootstrap seeds for each model
    K : int
        Number of models to train
    lambda_contrastive : float
        Weight of contrastive loss
    contrastive_loss_type : str
        Type of contrastive loss
        
    Returns:
    --------
    dict
        Dictionary with models and metrics
    """

    accuracies = []
    recalls = []
    precisions = []
    f1s = []
    models = []

    for k in range(K):
        seed = seeds[k]
        bootstrap_seed = bootstrap_seeds[k]
        hparams = hparams_list[k]

        np.random.seed(bootstrap_seed)
        X_train_k, y_train_k = bootstrap_data(X_train, y_train)

        np.random.seed(seed)
        torch.manual_seed(seed)

        mlp = MLPClassifierContrastive(
            input_dim=X_train_k.shape[1],
            hidden_layers=hparams["hidden_layers"],
            neurons_per_layer=hparams["neurons_per_layer"],
            activation=hparams["activation"],
            dropout=hparams["dropout"],
            seed=seed,
            lambda_contrastive=lambda_contrastive,
            contrastive_loss_type=contrastive_loss_type,
        )

        print(f"Model {k}: seed={seed}, bootstrap_seed={bootstrap_seed}")

        mlp.fit(
            X_train_k,
            y_train_k,
            X_val=X_test,
            y_val=y_test,
            verbose=hparams["verbose"],
            early_stopping=hparams["early_stopping"],
            lr=hparams["lr"],
            batch_size=hparams["batch_size"],
            epochs=hparams["epochs"],
            optimizer=hparams["optimizer"],
        )

        d = mlp.evaluate(X_test, y_test)
        accuracies.append(d["accuracy"])
        recalls.append(d["recall"])
        precisions.append(d["precision"])
        f1s.append(d["f1"])
        models.append(mlp)

    print(
        f"Ensemble (Contrastive): Avg accuracy={np.mean(accuracies):.4f}, "
        f"Avg recall={np.mean(recalls):.4f}, Avg precision={np.mean(precisions):.4f}, "
        f"Avg f1={np.mean(f1s):.4f}"
    )
    
    return {
        "models": models,
        "accuracies": accuracies,
        "recalls": recalls,
        "precisions": precisions,
        "f1s": f1s,
    }


def ensemble_predict_proba_contrastive(
    models: list, X: Union[np.ndarray, torch.Tensor]
) -> np.ndarray:
    """
    Predict probabilities using an ensemble of contrastive models.
    """
    assert len(models) > 0, "No models to predict"
    if len(X.shape) == 1:
        X = X.reshape(1, -1)
    if len(X.shape) == 2 and X.shape[0] != 1:
        pass  # Allow batch predictions
    
    predictions = []
    for model in models:
        predictions.append(model.predict_proba(X))
    predictions = np.array(predictions)

    if len(predictions.shape) == 2 and predictions.shape[1] == 1:
        predictions = predictions.flatten()

    return predictions


def ensemble_predict_crisp_contrastive(
    sample: np.ndarray, models: list, class_threshold: float = 0.5
) -> np.ndarray:
    """
    Predict crisp classes using an ensemble of contrastive models.
    """
    assert len(models) > 0, "No models to predict"
    if len(sample.shape) == 1:
        sample = sample.reshape(1, -1)

    predictions = []
    for model in models:
        predictions.append(model.predict_crisp(sample, threshold=class_threshold))
    predictions = np.array(predictions)

    if len(predictions.shape) == 2:
        predictions = predictions.flatten()

    return predictions


def ensemble_get_embeddings(
    models: list, X: Union[np.ndarray, torch.Tensor]
) -> np.ndarray:
    """
    Get embeddings from all models in the ensemble.
    
    Parameters:
    -----------
    models : list
        List of MLPClassifierContrastive models
    X : array-like
        Input data, shape (1, n_features) or (n_features,)
        
    Returns:
    --------
    np.ndarray
        Embeddings from all models, shape (n_models, embedding_dim)
    """
    assert len(models) > 0, "No models to predict"
    if len(X.shape) == 1:
        X = X.reshape(1, -1)
    
    embeddings = []
    for model in models:
        emb = model.get_embedding_numpy(X)
        embeddings.append(emb.flatten())
    
    return np.array(embeddings)


def train_K_mlps_contrastive_in_parallel(
    X_train,
    y_train,
    X_test,
    y_test,
    hparamsB,
    bootstrapB,
    seedB,
    hparams_base: dict,
    K: int,
    n_jobs: int = 4,
    lambda_contrastive: float = 0.1,
    contrastive_loss_type: str = 'supervised',
) -> list:
    """
    Train K MLP models with contrastive regularization in parallel.
    
    Parameters:
    -----------
    X_train : np.array
        Training data
    y_train : np.array
        Training labels
    X_test : np.array
        Test data
    y_test : np.array
        Test labels
    hparamsB : list[dict]
        List of hyperparameters for each model
    bootstrapB : list[int]
        List of bootstrap seeds
    seedB : list[int]
        List of seeds for each model
    hparams_base : dict
        Base hyperparameters
    K : int
        Number of models to train
    n_jobs : int
        Number of parallel jobs
    lambda_contrastive : float
        Weight of contrastive loss
    contrastive_loss_type : str
        Type of contrastive loss
        
    Returns:
    --------
    list
        List of dictionaries with models and metrics
    """
    k_for_each_job = K // n_jobs
    
    # Append the base hyperparameters
    hparamsB = [hparams_base | h for h in hparamsB]
    
    partitioned_hparams = np.array_split(hparamsB, n_jobs)
    partitioned_bootstrap = np.array_split(bootstrapB, n_jobs)
    partitioned_seed = np.array_split(seedB, n_jobs)
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(train_K_mlps_contrastive)(
            X_train,
            y_train,
            X_test,
            y_test,
            hparams_list=list(hparams),
            seeds=list(seeds),
            bootstrap_seeds=list(bootstrap_seeds),
            K=k_for_each_job,
            lambda_contrastive=lambda_contrastive,
            contrastive_loss_type=contrastive_loss_type,
        )
        for hparams, seeds, bootstrap_seeds in zip(
            partitioned_hparams, partitioned_seed, partitioned_bootstrap
        )
    )
    return results
