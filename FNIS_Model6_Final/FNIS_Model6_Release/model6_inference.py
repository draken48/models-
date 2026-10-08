
import json
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_v2_s


class FNISModel6:

    def __init__(
        self,
        model_path="model5_efficientnetv2s_best.pth",
        config_path="model6_config.json",
        classes_path="classes.txt",
        device=None
    ):

        self.device = torch.device(
            device
            if device is not None
            else (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        with open(
            config_path,
            "r"
        ) as f:
            self.config = json.load(f)

        self.classes = {}

        with open(
            classes_path,
            "r"
        ) as f:

            for line in f:

                if line.strip():

                    idx, name = (
                        line.rstrip("\n")
                        .split("\t", 1)
                    )

                    self.classes[int(idx)] = name

        self.num_classes = len(
            self.classes
        )

        # ----------------------------------------------------
        # Rebuild frozen Model 5 architecture.
        # ----------------------------------------------------

        self.model = efficientnet_v2_s(
            weights=None
        )

        in_features = (
            self.model.classifier[-1].in_features
        )

        self.model.classifier[-1] = nn.Sequential(
            nn.Dropout(
                p=0.35
            ),
            nn.Linear(
                in_features,
                self.num_classes
            )
        )

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=False
        )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.model = self.model.to(
            self.device
        )

        self.image_transform = transforms.Compose([
            transforms.Resize(
                (400, 400),
                antialias=True
            ),
            transforms.CenterCrop(
                384
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406
                ],
                std=[
                    0.229,
                    0.224,
                    0.225
                ]
            )
        ])

        self.temperature = float(
            self.config[
                "temperature_scaling"
            ][
                "temperature"
            ]
        )

        self.mc_samples = int(
            self.config[
                "mc_samples"
            ]
        )

        self.threshold = float(
            self.config[
                "reliability_threshold"
            ]
        )

    def _enable_mc_dropout(self):

        self.model.eval()

        for module in self.model.modules():

            if isinstance(
                module,
                (
                    nn.Dropout,
                    nn.Dropout2d,
                    nn.Dropout3d
                )
            ):

                module.train()

    @staticmethod
    def _entropy(probabilities):

        probabilities = np.clip(
            probabilities,
            1e-8,
            1.0
        )

        return float(
            -np.sum(
                probabilities
                * np.log(
                    probabilities
                )
            )
        )

    def _preprocess(self, image):

        if not isinstance(
            image,
            Image.Image
        ):
            image = Image.open(
                image
            )

        image = image.convert(
            "RGB"
        )

        return self.image_transform(
            image
        ).unsqueeze(0).to(
            self.device
        )

    @torch.no_grad()
    def predict(
        self,
        image
    ):

        self._enable_mc_dropout()

        samples = []

        for _ in range(
            self.mc_samples
        ):

            logits = self.model(
                self._preprocess(image)
            )

            logits = (
                logits /
                self.temperature
            )

            probabilities = torch.softmax(
                logits.float(),
                dim=1
            )[0]

            samples.append(
                probabilities
                .cpu()
                .numpy()
            )

        samples = np.asarray(
            samples
        )

        mean_probability = (
            samples.mean(
                axis=0
            )
        )

        entropy = self._entropy(
            mean_probability
        )

        max_entropy = np.log(
            self.num_classes
        )

        normalized_entropy = (
            entropy /
            max_entropy
        )

        predictions = np.argmax(
            samples,
            axis=1
        )

        counts = np.bincount(
            predictions,
            minlength=self.num_classes
        )

        majority_class = int(
            np.argmax(counts)
        )

        stability = float(
            counts[
                majority_class
            ] /
            self.mc_samples
        )

        predicted_class = int(
            np.argmax(
                mean_probability
            )
        )

        confidence = float(
            mean_probability[
                predicted_class
            ]
        )

        if (
            normalized_entropy
            <= self.threshold
        ):

            reliability = "HIGH"
            needs_review = False

        elif (
            confidence >= 0.55
            and stability >= 0.50
        ):

            reliability = "MEDIUM"
            needs_review = True

        else:

            reliability = "LOW"
            needs_review = True

        return {
            "predicted_class":
                self.classes[
                    predicted_class
                ],

            "class_index":
                predicted_class,

            "confidence":
                confidence,

            "predictive_entropy":
                normalized_entropy,

            "stability":
                stability,

            "reliability":
                reliability,

            "needs_review":
                needs_review,

            "probabilities": {
                self.classes[i]:
                    float(
                        mean_probability[i]
                    )
                for i in range(
                    self.num_classes
                )
            }
        }
